import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch_geometric.nn import GCNConv, global_mean_pool

from range_predictor import generate_simple_fault_tree, simulate_CTMC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SamplePredictor(nn.Module):
    """
    GNN che predice il numero di samples necessari per IS e MC.

    Input: grafo del fault tree
    Output: [log10(N_is), log10(N_mc)]
    """

    def __init__(self, node_features=5, hidden_dim=64):
        super().__init__()

        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [log10(N_is), log10(N_mc)]
        )

        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        embedding = global_mean_pool(x, batch)

        # Output: log10 del numero di samples
        # Range: [2, 5] → N ∈ [100, 100000]
        raw = self.predictor(embedding)
        log_n = 2.0 + torch.sigmoid(raw) * 3.0  # [2, 5]

        return log_n

    def sample_prediction(self, log_n):
        """Campiona con rumore per esplorazione."""
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(log_n, std)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled).sum(dim=-1)
        return sampled, log_prob

def find_required_samples_is(lambda_, mu_, alpha, beta, T, fault_tree, target_cv=0.10, max_n=50000, batch_size=500):
    """
    Trova il numero di samples IS necessari per raggiungere target_cv.
    """
    samples = []
    n = 0

    while n < max_n:
        results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
                   for _ in range(batch_size)]
        weights = [math.exp(r['log_w']) if r['top'] else 0.0 for r in results]
        samples.extend(weights)
        n += batch_size

        if n >= 1000:
            avg = np.mean(samples)
            if avg > 0:
                std_err = np.std(samples) / np.sqrt(n)
                cv = std_err / avg
                if cv <= target_cv:
                    return n

    return max_n

def find_required_samples_mc(lambda_, mu_, T, fault_tree, target_cv=0.10, max_n=100000, batch_size=500):
    """
    Trova il numero di samples MC (alpha=beta=1) necessari per target_cv.
    """
    comps = list(lambda_.keys())
    alpha = {c: 1.0 for c in comps}
    beta = {c: 1.0 for c in comps}

    hits = []
    n = 0

    while n < max_n:
        results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
                   for _ in range(batch_size)]
        batch_hits = [1.0 if r['top'] else 0.0 for r in results]
        hits.extend(batch_hits)
        n += batch_size

        if n >= 1000:
            p = np.mean(hits)
            if p > 0:
                std_err = np.sqrt(p * (1 - p) / n)
                cv = std_err / p
                if cv <= target_cv:
                    return n

    return max_n

def train_sample_predictor(n_iterations=200, T=100, target_cv=0.10, verbose=True):
    """
    Training self-supervised del SamplePredictor.
    """

    model = SamplePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = []

    print("=" * 60)
    print("TRAINING GNN - PREDIZIONE NUMERO SAMPLES")
    print("=" * 60)

    for iteration in range(n_iterations):
        # 1. Genera fault tree
        ft_data = generate_simple_fault_tree()
        pyg_data = ft_data['graph'].to_pyg_data().to(device)

        lambda_ = ft_data['lambda_']
        mu_ = ft_data['mu_']
        fault_tree = ft_data['fault_tree']
        comps = list(lambda_.keys())

        # 2. Predici N
        log_n_pred = model(pyg_data)
        log_n_sampled, log_prob = model.sample_prediction(log_n_pred)

        n_is_pred = int(10 ** log_n_sampled[0, 0].item())
        n_mc_pred = int(10 ** log_n_sampled[0, 1].item())

        # 3. Trova N reale
        alpha = {c: 3.0 for c in comps}
        beta = {c: 0.5 for c in comps}

        try:
            n_is_real = find_required_samples_is(
                lambda_, mu_, alpha, beta, T, fault_tree,
                target_cv=target_cv, max_n=20000, batch_size=200
            )
            n_mc_real = find_required_samples_mc(
                lambda_, mu_, T, fault_tree,
                target_cv=target_cv, max_n=50000, batch_size=500
            )

            # 4. Reward
            log_n_is_real = math.log10(max(100, n_is_real))
            log_n_mc_real = math.log10(max(100, n_mc_real))

            error_is = abs(log_n_sampled[0, 0].item() - log_n_is_real)
            error_mc = abs(log_n_sampled[0, 1].item() - log_n_mc_real)

            reward = -(error_is + error_mc)

            if error_is < 0.3 and error_mc < 0.3:
                reward += 1.0

        except Exception as e:
            print(f"  Errore: {e}")
            reward = -5.0
            n_is_real = n_is_pred
            n_mc_real = n_mc_pred

        # 5. Policy Gradient update
        loss = -reward * log_prob

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        history.append({
            'iteration': iteration,
            'structure': ft_data['structure'],
            'n_is_pred': n_is_pred,
            'n_is_real': n_is_real,
            'n_mc_pred': n_mc_pred,
            'n_mc_real': n_mc_real,
            'reward': reward
        })

        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration:3d} | {ft_data['structure']:8s} | "
                  f"IS: {n_is_pred:5d} vs {n_is_real:5d} | "
                  f"MC: {n_mc_pred:5d} vs {n_mc_real:5d}")

    return model

def get_predicted_samples(model, pyg_data):
    """
    Usa il modello addestrato per predire N_is e N_mc.
    """
    model.eval()

    if not hasattr(pyg_data, 'batch'):
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=pyg_data.x.device)

    with torch.no_grad():
        log_n = model(pyg_data)

        n_is = int(10 ** log_n[0, 0].item())
        n_mc = int(10 ** log_n[0, 1].item())

        n_is = max(500, min(100000, n_is))
        n_mc = max(500, min(100000, n_mc))

        return n_is, n_mc