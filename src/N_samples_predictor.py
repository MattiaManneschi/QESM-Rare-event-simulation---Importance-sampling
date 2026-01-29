"""
N_samples_predictor.py - VERSIONE CON T COME INPUT

Il SamplePredictor ora considera T per predire il numero di samples:
- T piccolo → P bassa → servono più samples
- T grande → P alta → bastano meno samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch_geometric.nn import GCNConv, global_mean_pool
from alfa_beta_range_predictor import generate_simple_fault_tree
from is_optimizer_evaluator import simulate_CTMC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SamplePredictor(nn.Module):
    """
    Predice il numero ottimale di samples IS e MC.

    NOVITÀ: Riceve T come input per adattare la predizione.
    - T piccolo → P bassa → più samples necessari
    - T grande → P alta → meno samples necessari
    """

    def __init__(self, node_features=5, hidden_dim=64):
        super().__init__()

        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # +7 per features globali: n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu, T_normalized
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + 7, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.log_std = nn.Parameter(torch.zeros(2))

    def compute_global_features(self, data, T_normalized):
        """Calcola features globali del grafo + T normalizzato."""
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

        # Aggiungi T_normalized come 7a feature
        return torch.tensor([[n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu, T_normalized]],
                            dtype=torch.float, device=x.device)

    def forward(self, data, T=100.0, T_max=500.0):
        """
        Forward pass con T come parametro.

        Args:
            data: PyG Data object
            T: tempo di missione corrente
            T_max: tempo massimo per normalizzazione

        Returns:
            log_n: [log10(n_is), log10(n_mc)]
        """
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        embedding = global_mean_pool(x, batch)

        # Normalizza T
        T_normalized = T / T_max

        # Aggiungi features globali con T
        global_features = self.compute_global_features(data, T_normalized)
        embedding = torch.cat([embedding, global_features], dim=1)

        raw = self.predictor(embedding)

        # Output range [2, 6] → N tra 100 e 1.000.000
        # Per T piccoli servono più samples, per T grandi meno
        # La rete impara questo pattern
        log_n = 2.0 + torch.sigmoid(raw) * 4.0

        return log_n

    def sample_prediction(self, log_n):
        """Campiona con rumore per esplorazione durante il training."""
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(log_n, std)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled).sum(dim=-1)
        return sampled, log_prob


def find_required_samples_is(lambda_, mu_, alpha, beta, T, fault_tree,
                             target_cv=0.3, max_n=500000, batch_size=1000):
    """
    Trova quanti campioni IS servono per raggiungere un CV target.

    Args:
        target_cv: coefficiente di variazione target (default 0.3 = 30%)
        max_n: massimo numero di samples
        batch_size: samples per la stima iniziale
    """
    results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
               for _ in range(batch_size)]

    weights = [math.exp(r['log_w']) if r['top'] else 0.0 for r in results]
    n_top = sum(1 for r in results if r['top'])

    if n_top == 0:
        return max_n

    # Stima CV attuale
    p_is = sum(weights) / batch_size
    if p_is <= 0:
        return max_n

    var_is = sum((w - p_is) ** 2 for w in weights) / batch_size
    std_is = math.sqrt(var_is / batch_size)
    cv_current = std_is / p_is if p_is > 0 else float('inf')

    # N necessario per raggiungere target_cv
    # CV scala con 1/sqrt(N), quindi N_new = N_old * (CV_old / CV_target)^2
    if cv_current <= target_cv:
        return batch_size

    scale_factor = (cv_current / target_cv) ** 2
    estimated_n = int(batch_size * scale_factor)

    return min(max(estimated_n, batch_size), max_n)


def find_required_samples_mc(lambda_, mu_, T, fault_tree,
                             target_cv=0.3, max_n=1000000, batch_size=2000):
    """
    Trova quanti campioni MC servono per raggiungere un CV target.
    """
    comps = list(lambda_.keys())
    alpha = {c: 1.0 for c in comps}
    beta = {c: 1.0 for c in comps}

    results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
               for _ in range(batch_size)]

    hits = [1.0 if r['top'] else 0.0 for r in results]
    n_top = sum(hits)

    if n_top == 0:
        return max_n

    p_mc = n_top / batch_size

    # Per Bernoulli: CV = sqrt((1-p)/(p*N))
    # Per target_cv: N = (1-p) / (p * target_cv^2)
    estimated_n = int((1 - p_mc) / (p_mc * target_cv ** 2))

    return min(max(estimated_n, batch_size), max_n)


def train_sample_predictor(range_model, n_iterations, T_range=(10, 500),
                           target_cv=0.3, comp_range=(2, 15),
                           pretrained_model=None, verbose=True):
    """
    Allena il SamplePredictor con T VARIABILE.

    Args:
        range_model: modello RangePredictor già addestrato
        n_iterations: numero di iterazioni di training
        T_range: (T_min, T_max) per campionare T casuali
        target_cv: CV target per le stime
        comp_range: (min_comp, max_comp) per generare fault tree
        pretrained_model: modello pre-addestrato per fine-tuning
        verbose: stampa progresso
    """
    if pretrained_model is not None:
        model = pretrained_model.to(device)
        print(f"[Fine-tuning] Partendo da modello pre-addestrato")
    else:
        model = SamplePredictor().to(device)
        print(f"[Training] Partendo da zero")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    range_model.eval()

    T_min, T_max = T_range

    print("=" * 60)
    print(f"TRAINING GNN - PREDIZIONE SAMPLES (T VARIABILE)")
    print(f"Range T: [{T_min}, {T_max}]")
    print(f"Range componenti: {comp_range}")
    print(f"Target CV: {target_cv}")
    print("=" * 60)

    for iteration in range(n_iterations):
        # Genera fault tree
        ft_data = generate_simple_fault_tree(comp_range)
        pyg_data = ft_data['graph'].to_pyg_data().to(device)

        lambda_ = ft_data['lambda_']
        mu_ = ft_data['mu_']
        fault_tree = ft_data['fault_tree']
        comps = list(lambda_.keys())
        n_comps = len(comps)

        # Campiona T casuale
        T = random.uniform(T_min, T_max)

        # 1. Ottieni biasing dal range_model (con T!)
        with torch.no_grad():
            ranges, _ = range_model(pyg_data, T=T, T_max=T_max)
            r = ranges[0].cpu().numpy()
            alpha_val = (r[0] + r[1]) / 2
            beta_val = (r[2] + r[3]) / 2

        alpha_dict = {c: alpha_val for c in comps}
        beta_dict = {c: beta_val for c in comps}

        # 2. Predici N (log10) con T
        log_n_pred = model(pyg_data, T=T, T_max=T_max)
        log_n_sampled, log_prob = model.sample_prediction(log_n_pred)

        n_is_pred = int(10 ** log_n_sampled[0, 0].item())
        n_mc_pred = int(10 ** log_n_sampled[0, 1].item())

        # 3. Calcola Ground Truth
        try:
            n_is_real = find_required_samples_is(
                lambda_, mu_, alpha_dict, beta_dict, T, fault_tree,
                target_cv=target_cv, max_n=500000, batch_size=500
            )
            n_mc_real = find_required_samples_mc(
                lambda_, mu_, T, fault_tree,
                target_cv=target_cv, max_n=1000000, batch_size=1000
            )

            # 4. REWARD
            log_n_is_real = math.log10(max(100, n_is_real))
            log_n_mc_real = math.log10(max(100, n_mc_real))

            diff_is = log_n_sampled[0, 0].item() - log_n_is_real
            diff_mc = log_n_sampled[0, 1].item() - log_n_mc_real

            # Penalità asimmetrica: sottostimare è peggio
            error_is = abs(diff_is) if diff_is >= 0 else abs(diff_is) * 3.0
            error_mc = abs(diff_mc) if diff_mc >= 0 else abs(diff_mc) * 3.0

            reward = -(error_is + error_mc)

            # Bonus se MC > IS (corretto per rare events)
            if n_mc_pred > n_is_pred:
                reward += 1.0
            else:
                reward -= 2.0

        except Exception as e:
            reward = -10.0
            n_is_real, n_mc_real = n_is_pred, n_mc_pred

        # 5. Backpropagation
        loss = -reward * log_prob
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if verbose and iteration % 20 == 0:
            print(f"Iter {iteration:3d} | T={T:5.0f} | comp={n_comps:2d} | "
                  f"IS: {n_is_pred:6d}/{n_is_real:6d} | MC: {n_mc_pred:7d}/{n_mc_real:7d} | "
                  f"Rew: {reward:.2f}")

    return model


def get_predicted_samples(model, pyg_data, T=100.0, T_max=500.0):
    """
    Ritorna il numero di campioni suggeriti.

    NOVITÀ: Richiede T come parametro!

    Args:
        model: SamplePredictor
        pyg_data: grafo PyG
        T: tempo di missione corrente
        T_max: tempo massimo per normalizzazione

    Returns:
        (n_is, n_mc)
    """
    model.eval()

    if not hasattr(pyg_data, 'batch'):
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=pyg_data.x.device)

    with torch.no_grad():
        log_n = model(pyg_data, T=T, T_max=T_max)

        n_is = int(10 ** log_n[0, 0].item())
        n_mc = int(10 ** log_n[0, 1].item())

        # Clamp ai limiti
        n_is = max(5000, min(500000, n_is))
        n_mc = max(10000, min(1000000, n_mc))

        # MC deve avere almeno tanti samples quanti IS
        n_mc = max(n_mc, n_is)

        return n_is, n_mc