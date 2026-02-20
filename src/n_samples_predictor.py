import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool

from src.fault_tree_generator import generate_rare_event_fault_tree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 6 bucket per IS: da 10^5 a 10^10
IS_BUCKETS = [
    int(1e5),  # 0: 10^5 = 100,000
    int(1e6),  # 1: 10^6 = 1,000,000
    int(1e7),  # 2: 10^7 = 10,000,000
    int(1e8),  # 3: 10^8 = 100,000,000
    int(1e9),  # 4: 10^9 = 1,000,000,000
    int(1e10),  # 5: 10^10 = 10,000,000,000
]

# 6 bucket per MC: da 10^5 a 10^10
MC_BUCKETS = [
    int(1e5),  # 0: 10^5
    int(1e6),  # 1: 10^6
    int(1e7),  # 2: 10^7
    int(1e8),  # 3: 10^8
    int(1e9),  # 4: 10^9
    int(1e10),  # 5: 10^10
]


def get_bucket_index(n, buckets):
    """Trova l'indice del bucket appropriato."""
    for i, b in enumerate(buckets):
        if n <= b:
            return i
    return len(buckets) - 1


def get_bucket_value(idx, buckets):
    """Restituisce il valore del bucket dato l'indice."""
    idx = max(0, min(idx, len(buckets) - 1))
    return buckets[idx]


def get_samples_heuristic(T, n_components, n_AND, n_OR, T_max=500):
    """
    Euristica basata su ordini di grandezza.

    Stima l'ordine di P e calcola samples necessari:
    - MC: deve avere abbastanza samples per ~10 hit → 10^(-P_order + 1)
    - IS: più efficiente, ~2 ordini di grandezza in meno
    """
    T_ratio = max(0.01, min(1.0, T / T_max))

    # Stima ordine di grandezza di P
    # Base: sistema semplice a T=T_max → P ~ 10^-3
    base_p_order = -3

    # Effetto AND: ogni AND aumenta la rarità
    # Più AND = probabilità congiunta più bassa
    and_effect = -0.4 * n_AND  # es: 10 AND → -4 ordini

    # Effetto T: T piccolo = evento più raro
    # A T=0, l'evento è molto più raro che a T=T_max
    t_effect = -4 * (1 - T_ratio)  # Fino a -4 ordini per T piccolo

    # Effetto componenti: più componenti = sistema più complesso
    comp_effect = -0.2 * math.log10(max(1, n_components / 5))

    # Stima finale dell'ordine di P
    estimated_p_order = base_p_order + and_effect + t_effect + comp_effect
    estimated_p_order = max(-10, min(-2, estimated_p_order))  # Clamp tra 10^-10 e 10^-2

    # MC: per avere ~10 hit, servono 10^(-P_order + 1) samples
    # Es: P ~ 10^-6 → servono ~10^7 samples MC
    # LIMITI PRATICI: max 10^7 per MC, 10^6 per IS
    mc_order = int(-estimated_p_order + 1)
    mc_order = max(5, min(7, mc_order))  # Clamp tra 10^5 e 10^7

    # IS: più efficiente di ~2 ordini di grandezza
    is_order = mc_order - 2
    is_order = max(5, min(6, is_order))  # Clamp tra 10^5 e 10^6

    n_is = int(10 ** is_order)
    n_mc = int(10 ** mc_order)

    # MC deve essere >= IS
    n_mc = max(n_mc, n_is)

    return n_is, n_mc


class SamplePredictor(nn.Module):
    def __init__(self, node_features=5, hidden_dim=64):
        super().__init__()

        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.is_head = nn.Linear(32, len(IS_BUCKETS))
        self.mc_head = nn.Linear(32, len(MC_BUCKETS))

    def compute_global_features(self, data, T, T_max):
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
        T_normalized = T / T_max
        T_factor = 1.0 + 6.0 * ((1.0 - T_normalized) ** 1.5)

        return torch.tensor([[n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu, T_normalized, T_factor]],
                            dtype=torch.float, device=x.device)

    def forward(self, data, T=100.0, T_max=500.0):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        embedding = global_mean_pool(x, batch)

        global_features = self.compute_global_features(data, T, T_max)
        embedding = torch.cat([embedding, global_features], dim=1)

        features = self.fc(embedding)

        is_logits = self.is_head(features)
        mc_logits = self.mc_head(features)

        return is_logits, mc_logits

    def predict_buckets(self, data, T=100.0, T_max=500.0):

        self.eval()
        with torch.no_grad():
            is_logits, mc_logits = self.forward(data, T, T_max)

        is_idx = torch.argmax(is_logits, dim=1).item()
        mc_idx = torch.argmax(mc_logits, dim=1).item()

        return is_idx, mc_idx

    def predict_samples(self, data, T=100.0, T_max=500.0):

        is_idx, mc_idx = self.predict_buckets(data, T, T_max)

        n_is = IS_BUCKETS[is_idx]
        n_mc = MC_BUCKETS[mc_idx]

        n_mc = max(n_mc, n_is)

        return n_is, n_mc


def train_sample_predictor(
        n_iterations,
        T_range=(10, 500),
        comp_range=(5, 45),
        pretrained_model=None,
        learning_rate=1e-3,
        verbose=True
):
    if pretrained_model is not None:
        model = pretrained_model.to(device)

        lr = learning_rate * 0.5
        print(f"[SamplePredictor] Fine-tuning (lr={lr:.0e})")
    else:
        model = SamplePredictor().to(device)
        lr = learning_rate
        print(f"[SamplePredictor] Training da zero (lr={lr:.0e})")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    T_min, T_max = T_range

    print("=" * 70)
    print(f"TRAINING SAMPLE PREDICTOR (ordini di grandezza 10^5 - 10^10)")
    print(f"Range T: [{T_min}, {T_max}]")
    print(f"Range componenti: {comp_range}")
    print(f"Bucket IS: {len(IS_BUCKETS)} classi (10^5 - 10^10)")
    print(f"Bucket MC: {len(MC_BUCKETS)} classi (10^5 - 10^10)")
    print("=" * 70)

    running_loss = 0.0
    correct_is = 0
    correct_mc = 0
    total = 0

    for iteration in range(n_iterations):

        ft_data = generate_rare_event_fault_tree(comp_range, target_p_order=-7)
        pyg_data = ft_data['graph'].to_pyg_data().to(device)

        lambda_ = ft_data['lambda_']
        n_comps = len(lambda_)

        x = pyg_data.x
        n_AND = int(x[:, 3].sum().item())
        n_OR = int(x[:, 4].sum().item())

        T = random.uniform(T_min, T_max)

        try:

            n_is_real, n_mc_real = get_samples_heuristic(T, n_comps, n_AND, n_OR, T_max)

            is_target = get_bucket_index(n_is_real, IS_BUCKETS)
            mc_target = get_bucket_index(n_mc_real, MC_BUCKETS)

            model.train()
            is_logits, mc_logits = model(pyg_data, T, T_max)

            is_target_tensor = torch.tensor([is_target], device=device)
            mc_target_tensor = torch.tensor([mc_target], device=device)

            loss_is = criterion(is_logits, is_target_tensor)
            loss_mc = criterion(mc_logits, mc_target_tensor)
            loss = loss_is + loss_mc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            is_pred = torch.argmax(is_logits, dim=1).item()
            mc_pred = torch.argmax(mc_logits, dim=1).item()

            if abs(is_pred - is_target) <= 1:
                correct_is += 1
            if abs(mc_pred - mc_target) <= 1:
                correct_mc += 1
            total += 1

        except Exception as e:
            if verbose:
                print(f"Errore iter {iteration}: {e}")
            continue

        scheduler.step()

        if verbose and iteration % 100 == 0 and total > 0:
            avg_loss = running_loss / total
            acc_is = 100 * correct_is / total
            acc_mc = 100 * correct_mc / total

            # Mostra ordini di grandezza
            is_order_pred = 5 + is_pred
            is_order_target = 5 + is_target
            mc_order_pred = 5 + mc_pred
            mc_order_target = 5 + mc_target

            print(f"Iter {iteration:4d} | T={T:5.0f} | {n_comps:2d}C {n_AND:2d}AND | "
                  f"IS: 10^{is_order_pred} vs 10^{is_order_target} | "
                  f"MC: 10^{mc_order_pred} vs 10^{mc_order_target} | "
                  f"Acc: IS={acc_is:.0f}% MC={acc_mc:.0f}% | "
                  f"Loss: {avg_loss:.3f}")

            running_loss = 0.0
            correct_is = 0
            correct_mc = 0
            total = 0

    return model


def train_sample_predictor_incremental(
        stages,
        n_iterations_per_stage=500,
        T_range=(10, 500),
        verbose=True
):
    model = None

    for i, comp_range in enumerate(stages):
        print(f"\n{'=' * 70}")
        print(f"STAGE {i + 1}/{len(stages)}: comp_range = {comp_range}")
        print(f"{'=' * 70}")

        lr = 1e-3 * (0.7 ** i)

        model = train_sample_predictor(
            n_iterations=n_iterations_per_stage,
            T_range=T_range,
            comp_range=comp_range,
            pretrained_model=model,
            learning_rate=lr,
            verbose=verbose
        )

    return model


def get_predicted_samples(model, pyg_data, T=100.0, T_max=500.0):
    """Wrapper per ottenere samples predetti dal modello."""
    model.eval()

    if not hasattr(pyg_data, 'batch'):
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=pyg_data.x.device)

    with torch.no_grad():
        n_is, n_mc = model.predict_samples(pyg_data, T, T_max)

    n_is = 250000

    return n_is, n_mc