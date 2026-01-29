import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch_geometric.nn import GCNConv, global_mean_pool
from alfa_beta_range_predictor import generate_simple_fault_tree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definizione bucket
IS_BUCKETS = [10000, 25000, 50000, 100000, 250000, 500000]
MC_BUCKETS = [20000, 50000, 100000, 250000, 500000, 1000000]


def get_bucket_index(n, buckets):
    for i, b in enumerate(buckets):
        if n <= b:
            return i
    return len(buckets) - 1


def get_bucket_value(idx, buckets):
    idx = max(0, min(idx, len(buckets) - 1))
    return buckets[idx]


def get_samples_heuristic(T, n_components, n_AND, n_OR, T_max=500):
    # 1. Fattore T: scaling inverso (T piccolo → più samples)
    T_factor = math.sqrt(T_max / max(T, 1))

    # 2. Fattore struttura: più AND → P più bassa → più samples
    total_gates = n_AND + n_OR
    if total_gates > 0:
        and_ratio = n_AND / total_gates
    else:
        and_ratio = 0.5

    # and_ratio = 0 → structure_factor = 1
    # and_ratio = 0.5 → structure_factor = 2
    # and_ratio = 1 → structure_factor = 4
    structure_factor = 1 + 3 * and_ratio

    # 3. Fattore componenti
    comp_factor = math.sqrt(n_components / 10)

    # 4. Base samples
    base_is = 30000
    base_mc = 60000

    # 5. Calcola samples finali
    n_is = int(base_is * T_factor * structure_factor * comp_factor)
    n_mc = int(base_mc * T_factor * structure_factor * comp_factor)

    # 6. Clamp ai limiti
    n_is = max(10000, min(500000, n_is))
    n_mc = max(20000, min(1000000, n_mc))

    # 7. MC deve avere almeno tanti samples quanti IS
    n_mc = max(n_mc, n_is)

    return n_is, n_mc


class SamplePredictor(nn.Module):
    def __init__(self, node_features=5, hidden_dim=64):
        super().__init__()

        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # +7 features globali: n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu, T_normalized
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Head per IS: 6 classi (bucket)
        self.is_head = nn.Linear(32, len(IS_BUCKETS))

        # Head per MC: 6 classi (bucket)
        self.mc_head = nn.Linear(32, len(MC_BUCKETS))

    def compute_global_features(self, data, T_normalized):
        x = data.x

        n_comp = x[:, 2].sum().item()
        n_AND = x[:, 3].sum().item()
        n_OR = x[:, 4].sum().item()

        comp_mask = x[:, 2] == 1
        if comp_mask.sum() > 0:
            avg_lambda = x[comp_mask, 0].mean().item()
            avg_mu = x[comp_mask, 1].mean().item()
        else:
            avg_lambda = 0
            avg_mu = 0

        depth = n_AND + n_OR

        return torch.tensor([[n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu, T_normalized]],
                            dtype=torch.float, device=x.device)

    def forward(self, data, T=100.0, T_max=500.0):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Global pooling
        embedding = global_mean_pool(x, batch)

        # Aggiungi features globali
        T_normalized = T / T_max
        global_features = self.compute_global_features(data, T_normalized)
        embedding = torch.cat([embedding, global_features], dim=1)

        # FC layers
        features = self.fc(embedding)

        # Classification heads
        is_logits = self.is_head(features)
        mc_logits = self.mc_head(features)

        return is_logits, mc_logits

    def predict_buckets(self, data, T=100.0, T_max=500.0):
        is_logits, mc_logits = self.forward(data, T, T_max)

        is_idx = torch.argmax(is_logits, dim=1).item()
        mc_idx = torch.argmax(mc_logits, dim=1).item()

        return is_idx, mc_idx

    def predict_samples(self, data, T=100.0, T_max=500.0):
        is_idx, mc_idx = self.predict_buckets(data, T, T_max)

        n_is = IS_BUCKETS[is_idx]
        n_mc = MC_BUCKETS[mc_idx]

        # MC deve essere >= IS
        n_mc = max(n_mc, n_is)

        return n_is, n_mc


def train_sample_predictor(sample_model, n_iterations, T_range=(10, 500),
                           comp_range=(5, 45),
                           pretrained_model=None, verbose=True):
    if pretrained_model is not None:
        model = pretrained_model.to(device)
        print(f"[Fine-tuning] Partendo da modello pre-addestrato")
    else:
        model = SamplePredictor().to(device)
        print(f"[Training] Nuovo modello")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    sample_model.eval()

    T_min, T_max = T_range

    print("=" * 60)
    print(f"TRAINING SAMPLE PREDICTOR v4 (BUCKET CLASSIFICATION)")
    print(f"Range T: [{T_min}, {T_max}]")
    print(f"Range componenti: {comp_range}")
    print(f"Ground Truth: EURISTICA")
    print(f"Bucket IS: {IS_BUCKETS}")
    print(f"Bucket MC: {MC_BUCKETS}")
    print("=" * 60)

    running_loss = 0.0
    correct_is = 0
    correct_mc = 0
    total = 0

    for iteration in range(n_iterations):
        # Genera fault tree
        ft_data = generate_simple_fault_tree(comp_range)
        pyg_data = ft_data['graph'].to_pyg_data().to(device)

        lambda_ = ft_data['lambda_']
        comps = list(lambda_.keys())
        n_comps = len(comps)

        # Estrai info struttura
        x = pyg_data.x
        n_AND = int(x[:, 3].sum().item())
        n_OR = int(x[:, 4].sum().item())

        # Campiona T
        T = random.uniform(T_min, T_max)

        try:
            # Ground truth dall'EURISTICA
            n_is_real, n_mc_real = get_samples_heuristic(T, n_comps, n_AND, n_OR, T_max)

            # Converti in indice bucket
            is_target = get_bucket_index(n_is_real, IS_BUCKETS)
            mc_target = get_bucket_index(n_mc_real, MC_BUCKETS)

            # Forward pass
            model.train()
            is_logits, mc_logits = model(pyg_data, T, T_max)

            # Loss
            is_target_tensor = torch.tensor([is_target], device=device)
            mc_target_tensor = torch.tensor([mc_target], device=device)

            loss_is = criterion(is_logits, is_target_tensor)
            loss_mc = criterion(mc_logits, mc_target_tensor)
            loss = loss_is + loss_mc

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Stats
            running_loss += loss.item()

            is_pred = torch.argmax(is_logits, dim=1).item()
            mc_pred = torch.argmax(mc_logits, dim=1).item()

            if is_pred == is_target:
                correct_is += 1
            if mc_pred == mc_target:
                correct_mc += 1
            total += 1

        except Exception as e:
            print(f"Errore iter {iteration}: {e}")
            continue

        scheduler.step()

        if verbose and iteration % 50 == 0 and total > 0:
            avg_loss = running_loss / total
            acc_is = 100 * correct_is / total
            acc_mc = 100 * correct_mc / total

            n_is_pred = IS_BUCKETS[is_pred]
            n_mc_pred = MC_BUCKETS[mc_pred]

            print(f"Iter {iteration:4d} | T={T:5.0f} | comp={n_comps:2d} | "
                  f"IS: {n_is_pred:6,} vs {n_is_real:6,} (acc={acc_is:.1f}%) | "
                  f"MC: {n_mc_pred:7,} vs {n_mc_real:7,} (acc={acc_mc:.1f}%) | "
                  f"Loss: {avg_loss:.3f}")

            running_loss = 0.0
            correct_is = 0
            correct_mc = 0
            total = 0

    return model


def get_predicted_samples(model, pyg_data, T=100.0, T_max=500.0):
    model.eval()

    if not hasattr(pyg_data, 'batch'):
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=pyg_data.x.device)

    with torch.no_grad():
        n_is, n_mc = model.predict_samples(pyg_data, T, T_max)

    return n_is, n_mc
