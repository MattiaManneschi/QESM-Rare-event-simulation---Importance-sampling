import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch_geometric.nn import GCNConv, global_mean_pool
from range_predictor import generate_simple_fault_tree
from range_tester import simulate_CTMC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SamplePredictor(nn.Module):
    def __init__(self, node_features=5, hidden_dim=64):
        super().__init__()

        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # +6 per features globali: n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + 6, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.log_std = nn.Parameter(torch.zeros(2))

    def compute_global_features(self, data):
        """Calcola features globali del grafo per aiutare la regressione."""
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

        return torch.tensor([[n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu]],
                            dtype=torch.float, device=x.device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        embedding = global_mean_pool(x, batch)

        # Aggiungi features globali
        global_features = self.compute_global_features(data)
        embedding = torch.cat([embedding, global_features], dim=1)

        raw = self.predictor(embedding)
        # log_n in scala log10. Output range [2, 5] significa N tra 100 e 100.000
        log_n = 2.0 + torch.sigmoid(raw) * 3.0

        return log_n

    def sample_prediction(self, log_n):
        """Campiona con rumore per esplorazione durante il training."""
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(log_n, std)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled).sum(dim=-1)
        return sampled, log_prob

def find_required_samples_is(lambda_, mu_, alpha, beta, T, fault_tree, target_top_events=100, max_n=50000, batch_size=500):
    """Trova quanti campioni IS servono per vedere 'target_top_events' eventi."""
    results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
               for _ in range(batch_size)]
    n_top = sum(1 for r in results if r['top'])

    if n_top == 0:
        return max_n

    top_rate = n_top / batch_size
    estimated_n = int(target_top_events / top_rate)

    return min(max(estimated_n, batch_size), max_n)

def find_required_samples_mc(lambda_, mu_, T, fault_tree, target_top_events=100, max_n=100000, batch_size=500):
    """Trova quanti campioni MC servono per vedere 'target_top_events' eventi."""
    comps = list(lambda_.keys())
    alpha = {c: 1.0 for c in comps}
    beta = {c: 1.0 for c in comps}

    results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
               for _ in range(batch_size)]
    n_top = sum(1 for r in results if r['top'])

    if n_top == 0:
        return max_n

    top_rate = n_top / batch_size
    estimated_n = int(target_top_events / top_rate)

    return min(max(estimated_n, batch_size), max_n)

def train_sample_predictor(range_model, n_iterations=200, T=100, target_top_events=100, verbose=True):
    """
    Allena il SamplePredictor con Reward Asimmetrico:
    Punisce severamente la sottostima dei campioni necessaria per la precisione statistica.
    """
    model = SamplePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    range_model.eval()

    print("=" * 60)
    print(f"TRAINING GNN - PREDIZIONE SAMPLES (Target Events: {target_top_events})")
    print("=" * 60)

    for iteration in range(n_iterations):
        ft_data = generate_simple_fault_tree()
        pyg_data = ft_data['graph'].to_pyg_data().to(device)

        lambda_ = ft_data['lambda_']
        mu_ = ft_data['mu_']
        fault_tree = ft_data['fault_tree']
        comps = list(lambda_.keys())

        # 1. Ottieni biasing suggerito dal range_model
        with torch.no_grad():
            ranges, _ = range_model(pyg_data)
            r = ranges[0].cpu().numpy()
            alpha_val = (r[0] + r[1]) / 2
            beta_val = (r[2] + r[3]) / 2

        alpha_dict = {c: alpha_val for c in comps}
        beta_dict = {c: beta_val for c in comps}

        # 2. Predici N (log10)
        log_n_pred = model(pyg_data)
        log_n_sampled, log_prob = model.sample_prediction(log_n_pred)

        n_is_pred = int(10 ** log_n_sampled[0, 0].item())
        n_mc_pred = int(10 ** log_n_sampled[0, 1].item())

        # 3. Calcola il Ground Truth (N reale richiesto per stabilità)
        try:
            n_is_real = find_required_samples_is(
                lambda_, mu_, alpha_dict, beta_dict, T, fault_tree,
                target_top_events=target_top_events, max_n=20000, batch_size=200
            )
            n_mc_real = find_required_samples_mc(
                lambda_, mu_, T, fault_tree,
                target_top_events=target_top_events, max_n=50000, batch_size=500
            )

            # 4. REWARD ASIMMETRICO
            log_n_is_real = math.log10(max(100, n_is_real))
            log_n_mc_real = math.log10(max(100, n_mc_real))

            diff_is = log_n_sampled[0, 0].item() - log_n_is_real
            diff_mc = log_n_sampled[0, 1].item() - log_n_mc_real

            # Se diff < 0 (sottostima), moltiplichiamo l'errore per 5.0
            error_is = abs(diff_is) if diff_is >= 0 else abs(diff_is) * 5.0
            error_mc = abs(diff_mc) if diff_mc >= 0 else abs(diff_mc) * 5.0

            reward = -(error_is + error_mc)

            # Bonus se IS è più efficiente di MC, MA solo se entrambi hanno campioni sufficienti
            # (usiamo diff > -0.1 come soglia di tolleranza per la sottostima logaritmica)
            if n_is_pred < n_mc_pred and diff_is > -0.1 and diff_mc > -0.1:
                reward += 3.0

            if n_is_pred > n_mc_pred:
                reward -= 4.0

        except Exception as e:
            reward = -10.0
            n_is_real, n_mc_real = n_is_pred, n_mc_pred

        # 5. Backpropagation
        loss = -reward * log_prob
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration:3d} | {ft_data['structure']:15s} | "
                  f"IS: {n_is_pred:5d}/{n_is_real:5d} | MC: {n_mc_pred:5d}/{n_mc_real:5d} | Rew: {reward:.2f}")

    return model

def get_predicted_samples(model, pyg_data):
    """Ritorna il numero di campioni suggeriti con floor dinamici e liberalizzati."""
    model.eval()

    if not hasattr(pyg_data, 'batch'):
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=pyg_data.x.device)

    with torch.no_grad():
        log_n = model(pyg_data)

        n_is = int(10 ** log_n[0, 0].item())
        n_mc = int(10 ** log_n[0, 1].item())

        # Liberalizziamo i limiti:
        # Per IS scendiamo a 200 (se l'albero è molto facile)
        # Per MC alziamo a 1000 (sotto i quali la stima MC è rumore puro)
        n_is = max(200, min(100000, n_is))
        n_mc = max(1000, min(100000, n_mc))

        return n_is, n_mc