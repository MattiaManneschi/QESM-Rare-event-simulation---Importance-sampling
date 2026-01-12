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
        """Calcola features globali del grafo."""
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
        log_n = 2.0 + torch.sigmoid(raw) * 3.0  # [2, 5] → N ∈ [100, 100000]

        return log_n

    def sample_prediction(self, log_n):
        """Campiona con rumore per esplorazione."""
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(log_n, std)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled).sum(dim=-1)
        return sampled, log_prob

def find_required_samples_is(lambda_, mu_, alpha, beta, T, fault_tree, target_top_events=50, max_n=50000, batch_size=500):

    # Esegui un batch iniziale per stimare il top rate
    results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
               for _ in range(batch_size)]
    n_top = sum(1 for r in results if r['top'])

    if n_top == 0:
        return max_n

    top_rate = n_top / batch_size
    estimated_n = int(target_top_events / top_rate)

    return min(max(estimated_n, batch_size), max_n)

def find_required_samples_mc(lambda_, mu_, T, fault_tree, target_top_events=50, max_n=100000, batch_size=500):
    comps = list(lambda_.keys())
    alpha = {c: 1.0 for c in comps}
    beta = {c: 1.0 for c in comps}

    # Esegui un batch iniziale per stimare il top rate
    results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
               for _ in range(batch_size)]
    n_top = sum(1 for r in results if r['top'])

    if n_top == 0:
        return max_n

    top_rate = n_top / batch_size
    estimated_n = int(target_top_events / top_rate)

    return min(max(estimated_n, batch_size), max_n)

def train_sample_predictor(range_model, n_iterations=200, T=100, target_top_events=50, verbose=True):

    model = SamplePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    range_model.eval()  # Non modificare range_model

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

        # 2. Ottieni α, β dal range_model
        with torch.no_grad():
            ranges, _ = range_model(pyg_data)
            r = ranges[0].cpu().numpy()
            alpha_val = (r[0] + r[1]) / 2  # Media del range
            beta_val = (r[2] + r[3]) / 2

        alpha_dict = {c: alpha_val for c in comps}
        beta_dict = {c: beta_val for c in comps}

        # 3. Predici N
        log_n_pred = model(pyg_data)
        log_n_sampled, log_prob = model.sample_prediction(log_n_pred)

        n_is_pred = int(10 ** log_n_sampled[0, 0].item())
        n_mc_pred = int(10 ** log_n_sampled[0, 1].item())

        # 4. Trova N reale
        try:
            n_is_real = find_required_samples_is(
                lambda_, mu_, alpha_dict, beta_dict, T, fault_tree,
                target_top_events=target_top_events, max_n=20000, batch_size=200
            )
            n_mc_real = find_required_samples_mc(
                lambda_, mu_, T, fault_tree,
                target_top_events=target_top_events, max_n=50000, batch_size=500
            )

            # 5. Reward
            log_n_is_real = math.log10(max(100, n_is_real))
            log_n_mc_real = math.log10(max(100, n_mc_real))

            error_is = abs(log_n_sampled[0, 0].item() - log_n_is_real)
            error_mc = abs(log_n_sampled[0, 1].item() - log_n_mc_real)

            reward = -(error_is + error_mc)

            # Bonus se IS < MC (comportamento corretto)
            if n_is_real < n_mc_real and n_is_pred < n_mc_pred:
                reward += 2.0

            # Penalità se predice IS > MC
            if n_is_pred > n_mc_pred:
                reward -= 3.0

        except Exception as e:
            print(f"  Errore: {e}")
            reward = -5.0
            n_is_real = n_is_pred
            n_mc_real = n_mc_pred

        # 6. Update
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