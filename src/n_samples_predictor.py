"""
SamplePredictor v2: Predice il numero ottimale di samples per IS e MC.

Miglioramenti rispetto a v1:
- 12 bucket invece di 6 (granularità doppia)
- Gap più piccoli tra bucket (max 1.5x invece di 2x)
- Supporto esplicito per training incrementale
- Euristica migliorata
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch_geometric.nn import GCNConv, global_mean_pool

from src.fault_tree_generator import generate_rare_event_fault_tree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# BUCKET DEFINITIONS - GRANULARITÀ AUMENTATA
# =============================================================================

# 12 bucket per IS: da 5k a 750k
IS_BUCKETS = [
    5000,  # 0: molto semplice
    10000,  # 1
    15000,  # 2
    25000,  # 3
    40000,  # 4
    60000,  # 5
    100000,  # 6
    150000,  # 7
    250000,  # 8
    400000,  # 9
    600000,  # 10
    750000,  # 11: molto complesso
]

# 12 bucket per MC: da 10k a 1.5M
MC_BUCKETS = [
    10000,  # 0
    20000,  # 1
    35000,  # 2
    50000,  # 3
    80000,  # 4
    120000,  # 5
    200000,  # 6
    350000,  # 7
    500000,  # 8
    750000,  # 9
    1000000,  # 10
    1500000,  # 11
]


def get_bucket_index(n, buckets):
    """Trova l'indice del bucket più vicino."""
    for i, b in enumerate(buckets):
        if n <= b:
            return i
    return len(buckets) - 1


def get_bucket_value(idx, buckets):
    """Restituisce il valore del bucket dato l'indice."""
    idx = max(0, min(idx, len(buckets) - 1))
    return buckets[idx]


# =============================================================================
# EURISTICA MIGLIORATA
# =============================================================================

def get_samples_heuristic(T, n_components, n_AND, n_OR, T_max=500):
    """
    Versione ottimizzata: Riduce la saturazione per permettere alla GNN
    di variare i bucket durante la scansione della CDF.
    """
    T_ratio = max(0.01, min(1.0, T / T_max))

    # 1. Fattore T: Usiamo una potenza più dolce (0.7 invece di lineare/1.5)
    # Vogliamo che cali sensibilmente quando T > 100
    T_factor = 1.0 + 2.0 * math.pow(1.0 - T_ratio, 0.7)

    # 2. Fattore AND: Logaritmico invece che a scaglioni lineari.
    # Gli scaglioni creano "salti" che confondono il gradiente della GNN.
    # Con 8 AND: factor ~= 1 + log(9)*0.8 ~= 2.7 (prima era più alto)
    and_factor = 1.0 + math.log1p(n_AND) * 0.8

    # 3. Fattore ratio AND/OR: Più bilanciato
    total_gates = n_AND + n_OR
    and_ratio = n_AND / total_gates if total_gates > 0 else 0.5
    # La dominanza di AND è già catturata da and_factor, qui scaliamo meno
    ratio_factor = 1.0 + (and_ratio * 0.5)

    # 4. Fattore componenti: logaritmico
    # 33 comp: log(3.3)*1.2 + 1 ~= 2.4
    comp_factor = 1.0 + math.log1p(n_components / 10.0) * 1.2

    # 5. Calcolo samples base
    # Abbassiamo leggermente la base per dare spazio ai moltiplicatori
    base_is = 12000

    # Sommiamo i fattori invece di moltiplicarli tutti?
    # No, ma limitiamo il prodotto totale (Damping)
    total_factor = T_factor * and_factor * ratio_factor * comp_factor

    # Applichiamo una compressione (soft-clamping) prima dei bucket
    # Impedisce a sistemi complessi di finire sempre a 750k
    n_is = int(base_is * math.pow(total_factor, 0.9))

    # 6. MC dinamico rispetto alla rarità (T)
    # Se T è piccolo (evento raro), MC deve essere molto più grande di IS
    # Se T è grande, MC può avvicinarsi a IS
    mc_multiplier = 4.0 * (1.0 - T_ratio) + 1.5
    n_mc = int(n_is * mc_multiplier)

    # 7. Clamp finale ai bucket definiti nel file
    n_is = max(IS_BUCKETS[0], min(IS_BUCKETS[-1], n_is))
    n_mc = max(MC_BUCKETS[0], min(MC_BUCKETS[-1], n_mc))

    return n_is, n_mc


# =============================================================================
# SAMPLE PREDICTOR MODEL
# =============================================================================

class SamplePredictor(nn.Module):
    """
    GNN che predice il bucket ottimale di samples per IS e MC.

    Output: 
    - IS bucket index (0-11)
    - MC bucket index (0-11)
    """

    def __init__(self, node_features=5, hidden_dim=64):
        super().__init__()

        # GNN layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # +8 features globali: n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu, T_normalized, T_factor
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Classification heads (12 classi ciascuno)
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

        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Global pooling
        embedding = global_mean_pool(x, batch)

        # Aggiungi features globali
        global_features = self.compute_global_features(data, T, T_max)
        embedding = torch.cat([embedding, global_features], dim=1)

        # FC layers
        features = self.fc(embedding)

        # Classification heads
        is_logits = self.is_head(features)
        mc_logits = self.mc_head(features)

        return is_logits, mc_logits

    def predict_buckets(self, data, T=100.0, T_max=500.0):
        """Predice gli indici dei bucket."""
        self.eval()
        with torch.no_grad():
            is_logits, mc_logits = self.forward(data, T, T_max)

        is_idx = torch.argmax(is_logits, dim=1).item()
        mc_idx = torch.argmax(mc_logits, dim=1).item()

        return is_idx, mc_idx

    def predict_samples(self, data, T=100.0, T_max=500.0):
        """Predice il numero di samples."""
        is_idx, mc_idx = self.predict_buckets(data, T, T_max)

        n_is = IS_BUCKETS[is_idx]
        n_mc = MC_BUCKETS[mc_idx]

        # MC >= IS
        n_mc = max(n_mc, n_is)

        return n_is, n_mc


# =============================================================================
# TRAINING CON SUPPORTO INCREMENTALE
# =============================================================================

def train_sample_predictor(
        n_iterations,
        T_range=(10, 500),
        comp_range=(5, 45),
        pretrained_model=None,
        learning_rate=1e-3,
        verbose=True
):
    """
    Addestra il SamplePredictor.

    Supporta training incrementale:
    - pretrained_model: modello pre-addestrato da cui partire
    - Utile per: (2,15) → (15,30) → (30,45)

    Args:
        n_iterations: numero di iterazioni
        T_range: (T_min, T_max)
        comp_range: (min_components, max_components)
        pretrained_model: modello pre-addestrato (opzionale)
        learning_rate: lr iniziale (ridotto per fine-tuning)
        verbose: stampa progress

    Returns:
        modello addestrato
    """

    if pretrained_model is not None:
        model = pretrained_model.to(device)
        # Learning rate ridotto per fine-tuning
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
    print(f"TRAINING SAMPLE PREDICTOR v2")
    print(f"Range T: [{T_min}, {T_max}]")
    print(f"Range componenti: {comp_range}")
    print(f"Bucket IS: {len(IS_BUCKETS)} classi ({IS_BUCKETS[0]:,} - {IS_BUCKETS[-1]:,})")
    print(f"Bucket MC: {len(MC_BUCKETS)} classi ({MC_BUCKETS[0]:,} - {MC_BUCKETS[-1]:,})")
    print("=" * 70)

    running_loss = 0.0
    correct_is = 0
    correct_mc = 0
    total = 0

    for iteration in range(n_iterations):
        # Genera fault tree
        ft_data = generate_rare_event_fault_tree(comp_range, target_p_order=-5)
        pyg_data = ft_data['graph'].to_pyg_data().to(device)

        lambda_ = ft_data['lambda_']
        n_comps = len(lambda_)

        # Estrai struttura
        x = pyg_data.x
        n_AND = int(x[:, 3].sum().item())
        n_OR = int(x[:, 4].sum().item())

        # Campiona T
        T = random.uniform(T_min, T_max)

        try:
            # Ground truth dall'euristica
            n_is_real, n_mc_real = get_samples_heuristic(T, n_comps, n_AND, n_OR, T_max)

            # Converti in bucket index
            is_target = get_bucket_index(n_is_real, IS_BUCKETS)
            mc_target = get_bucket_index(n_mc_real, MC_BUCKETS)

            # Forward
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

            # Accuracy (anche ±1 bucket è ok)
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

            n_is_pred = IS_BUCKETS[is_pred]
            n_mc_pred = MC_BUCKETS[mc_pred]

            print(f"Iter {iteration:4d} | T={T:5.0f} | {n_comps:2d}C {n_AND:2d}AND | "
                  f"IS: {n_is_pred:7,} vs {n_is_real:7,} | "
                  f"MC: {n_mc_pred:8,} vs {n_mc_real:8,} | "
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
    """
    Training incrementale su più stage di complessità.

    Args:
        stages: lista di comp_range, es. [(2,15), (10,25), (20,40)]
        n_iterations_per_stage: iterazioni per ogni stage
        T_range: range temporale
        verbose: stampa progress

    Returns:
        modello finale
    """
    model = None

    for i, comp_range in enumerate(stages):
        print(f"\n{'=' * 70}")
        print(f"STAGE {i + 1}/{len(stages)}: comp_range = {comp_range}")
        print(f"{'=' * 70}")

        # Learning rate decresce per ogni stage
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


# =============================================================================
# UTILITY
# =============================================================================

def get_predicted_samples(model, pyg_data, T=100.0, T_max=500.0):
    """Wrapper per compatibilità."""
    model.eval()

    if not hasattr(pyg_data, 'batch'):
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=pyg_data.x.device)

    with torch.no_grad():
        n_is, n_mc = model.predict_samples(pyg_data, T, T_max)

    return n_is, n_mc
