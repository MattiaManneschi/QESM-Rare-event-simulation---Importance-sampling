import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaultTreeGraph:
    """
    Rappresenta un fault tree come grafo per PyTorch Geometric.

    Nodi componente: features = [λ, μ, 1, 0, 0]
    Nodi AND gate:   features = [0, 0, 0, 1, 0]
    Nodi OR gate:    features = [0, 0, 0, 0, 1]

    Archi: figlio → padre (bottom-up)
    """

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_features = []
        self.components = {}

    def add_component(self, name, lambda_, mu_):
        """Aggiunge un nodo componente (foglia)."""
        idx = len(self.nodes)
        self.nodes.append({'type': 'component', 'name': name, 'idx': idx})
        self.node_features.append([lambda_, mu_, 1, 0, 0])
        self.components[name] = idx
        return idx

    def add_gate(self, gate_type, children_idx):
        """Aggiunge un gate AND/OR."""
        idx = len(self.nodes)
        self.nodes.append({'type': gate_type, 'idx': idx})

        if gate_type == 'AND':
            self.node_features.append([0, 0, 0, 1, 0])
        else:
            self.node_features.append([0, 0, 0, 0, 1])

        for child_idx in children_idx:
            self.edges.append((child_idx, idx))

        return idx

    def to_pyg_data(self):
        """Converte in formato PyTorch Geometric."""
        x = torch.tensor(self.node_features, dtype=torch.float)
        if self.edges:
            edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def get_lambda_mu(self):
        """Estrae dizionari lambda e mu."""
        lambda_ = {}
        mu_ = {}
        for node in self.nodes:
            if node['type'] == 'component':
                name = node['name']
                idx = node['idx']
                lambda_[name] = self.node_features[idx][0]
                mu_[name] = self.node_features[idx][1]
        return lambda_, mu_

    def get_logic_function(self):
        """Genera la funzione booleana del Fault Tree."""
        # La radice è l'ultimo nodo aggiunto
        root_idx = len(self.nodes) - 1

        def evaluate(node_idx, state):
            node = self.nodes[node_idx]

            # Se è un componente, restituisce il suo stato (1=guasto, 0=sano)
            if node['type'] == 'component':
                return state[node['name']] == 1

            # Se è un gate, valuta ricorsivamente i figli
            # Gli archi sono (figlio, padre), cerchiamo i figli del nodo corrente
            children = [src for src, dst in self.edges if dst == node_idx]
            results = [evaluate(c, state) for c in children]

            if node['type'] == 'AND':
                return all(results) if results else False
            if node['type'] == 'OR':
                return any(results) if results else False
            return False

        return lambda state: evaluate(root_idx, state)

def generate_simple_fault_tree():
    """
    Genera fault tree semplici con P ~ 10^-1 o 10^-2.

    Con λ/μ ~ 0.1:
    - OR di 2-3 componenti: P ≈ n * (λ/μ) ~ 0.1-0.3
    - AND di 2 componenti: P ≈ (λ/μ)^2 ~ 0.01
    - 2oo3: P ≈ 3 * (λ/μ)^2 ~ 0.03
    """

    base_mu = 0.1
    ratio = random.uniform(0.05, 0.15)
    base_lambda = base_mu * ratio

    def vary(val):
        return val * random.uniform(0.8, 1.2)

    structure_type = random.choice(['OR_2', 'OR_3', 'AND_2', 'AND_3', '2oo3', 'mixed'])

    graph = FaultTreeGraph()

    if structure_type == 'OR_2':
        idx_A = graph.add_component('A', vary(base_lambda), vary(base_mu))
        idx_B = graph.add_component('B', vary(base_lambda), vary(base_mu))
        graph.add_gate('OR', [idx_A, idx_B])

        def fault_tree(state):
            return state['A'] == 1 or state['B'] == 1

    elif structure_type == 'OR_3':
        idx_A = graph.add_component('A', vary(base_lambda), vary(base_mu))
        idx_B = graph.add_component('B', vary(base_lambda), vary(base_mu))
        idx_C = graph.add_component('C', vary(base_lambda), vary(base_mu))
        graph.add_gate('OR', [idx_A, idx_B, idx_C])

        def fault_tree(state):
            return state['A'] == 1 or state['B'] == 1 or state['C'] == 1

    elif structure_type == 'AND_2':
        idx_A = graph.add_component('A', vary(base_lambda), vary(base_mu))
        idx_B = graph.add_component('B', vary(base_lambda), vary(base_mu))
        graph.add_gate('AND', [idx_A, idx_B])

        def fault_tree(state):
            return state['A'] == 1 and state['B'] == 1

    elif structure_type == 'AND_3':
        idx_A = graph.add_component('A', vary(base_lambda), vary(base_mu))
        idx_B = graph.add_component('B', vary(base_lambda), vary(base_mu))
        idx_C = graph.add_component('C', vary(base_lambda), vary(base_mu))
        graph.add_gate('AND', [idx_A, idx_B, idx_C])

        def fault_tree(state):
            return state['A'] == 1 and state['B'] == 1 and state['C'] == 1

    elif structure_type == '2oo3':
        idx_A = graph.add_component('A', vary(base_lambda), vary(base_mu))
        idx_B = graph.add_component('B', vary(base_lambda), vary(base_mu))
        idx_C = graph.add_component('C', vary(base_lambda), vary(base_mu))
        idx_AB = graph.add_gate('AND', [idx_A, idx_B])
        idx_AC = graph.add_gate('AND', [idx_A, idx_C])
        idx_BC = graph.add_gate('AND', [idx_B, idx_C])
        graph.add_gate('OR', [idx_AB, idx_AC, idx_BC])

        def fault_tree(state):
            return state['A'] + state['B'] + state['C'] >= 2

    else:  # mixed: (A ∧ B) ∨ C
        idx_A = graph.add_component('A', vary(base_lambda), vary(base_mu))
        idx_B = graph.add_component('B', vary(base_lambda), vary(base_mu))
        idx_C = graph.add_component('C', vary(base_lambda), vary(base_mu))
        idx_AND = graph.add_gate('AND', [idx_A, idx_B])
        graph.add_gate('OR', [idx_AND, idx_C])

        def fault_tree(state):
            return (state['A'] == 1 and state['B'] == 1) or state['C'] == 1

    lambda_, mu_ = graph.get_lambda_mu()

    return {
        'graph': graph,
        'fault_tree': fault_tree,
        'lambda_': lambda_,
        'mu_': mu_,
        'structure': structure_type
    }

class RangePredictor(nn.Module):
    """
    GNN che predice i range ottimali di alpha e beta.

    Input: grafo del fault tree con features [λ, μ, is_comp, is_AND, is_OR]
    Output: [alpha_min, alpha_max, beta_min, beta_max]
    """

    def __init__(self, node_features=5, hidden_dim=64, embedding_dim=32):
        super().__init__()

        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        self.log_std = nn.Parameter(torch.zeros(4))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        embedding = global_mean_pool(x, batch)

        raw = self.predictor(embedding)

        val = F.softplus(raw)

        alpha_min = val[:, 0] + 1.0
        alpha_max = alpha_min + val[:, 1] + 1.0

        beta_min = val[:, 2] + 0.1
        beta_max = beta_min + val[:, 3] + 0.2

        out = torch.stack([alpha_min, alpha_max, beta_min, beta_max], dim=1)

        return out, embedding

    def get_ranges(self, raw):
        """Converte output raw in range vincolati."""
        alpha_min = 1.0 + torch.sigmoid(raw[:, 0]) * 4.0
        alpha_max = alpha_min + 1.0 + torch.sigmoid(raw[:, 1]) * 10.0
        beta_min = 0.1 + torch.sigmoid(raw[:, 2]) * 0.4
        beta_max = beta_min + 0.1 + torch.sigmoid(raw[:, 3]) * 0.4

        return alpha_min, alpha_max, beta_min, beta_max

    def sample_ranges(self, raw):
        """Campiona range con rumore per esplorazione."""
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(raw, std)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled).sum(dim=-1)

        return sampled, log_prob

def simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree):
    """Simula una traiettoria CTMC con Importance Sampling."""
    t = 0.0
    state = {i: 0 for i in lambda_}
    log_w = 0.0
    top_event_hit = False

    while t < T:
        rates_orig = {}
        rates_is = {}

        for i in lambda_:
            if state[i] == 0:
                rates_orig[i] = lambda_[i]
                rates_is[i] = lambda_[i] * alpha[i]
            else:
                rates_orig[i] = mu_[i]
                rates_is[i] = mu_[i] * beta[i]

        R_orig = sum(rates_orig.values())
        R_is = sum(rates_is.values())

        if R_is <= 0:
            break

        dt = random.expovariate(R_is)

        if t + dt > T:
            log_w += (R_is - R_orig) * (T - t)
            break

        log_w += (R_is - R_orig) * dt
        t += dt

        comps = list(lambda_.keys())
        p_is = [rates_is[c] / R_is for c in comps]
        chosen_comp = random.choices(comps, weights=p_is)[0]

        log_w += math.log((rates_orig[chosen_comp] / R_orig) / (rates_is[chosen_comp] / R_is))

        state[chosen_comp] = 1 - state[chosen_comp]

        if fault_tree(state):
            top_event_hit = True

    return {"top": top_event_hit, "log_w": log_w}

def evaluate_ranges(lambda_, mu_, T, fault_tree, alpha_min, alpha_max, beta_min, beta_max, n_eval=5000):
    """
    Valuta la qualità dei range predetti eseguendo IS.
    Ritorna CV della stima.
    """
    comps = list(lambda_.keys())

    # Usa valori medi dei range
    alpha = {c: (alpha_min + alpha_max) / 2 for c in comps}
    beta = {c: (beta_min + beta_max) / 2 for c in comps}

    results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
               for _ in range(n_eval)]

    weights = np.array([math.exp(r['log_w']) if r['top'] else 0.0 for r in results])
    n_top = sum(1 for r in results if r['top'])

    p_is = np.mean(weights)
    std_is = np.std(weights) / np.sqrt(n_eval)
    cv = std_is / p_is if p_is > 0 else float('inf')

    return {
        'p_is': p_is,
        'std_is': std_is,
        'cv': cv,
        'n_top': n_top,
        'top_rate': n_top / n_eval
    }

def train_range_predictor(n_iterations=200, T=100, verbose=True):

    model = RangePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = []

    print("=" * 60)
    print("TRAINING GNN - PREDIZIONE RANGE Î±, Î²")
    print("=" * 60)

    for iteration in range(n_iterations):
        # 1. Genera fault tree
        ft_data = generate_simple_fault_tree()
        pyg_data = ft_data['graph'].to_pyg_data().to(device)

        # 2. Predici range
        raw, _ = model(pyg_data)
        sampled_raw, log_prob = model.sample_ranges(raw)
        alpha_min, alpha_max, beta_min, beta_max = model.get_ranges(sampled_raw)

        a_min, a_max = alpha_min.item(), alpha_max.item()
        b_min, b_max = beta_min.item(), beta_max.item()

        # 3. Valuta
        try:
            eval_results = evaluate_ranges(
                ft_data['lambda_'], ft_data['mu_'], T, ft_data['fault_tree'],
                a_min, a_max, b_min, b_max, n_eval=3000
            )

            cv = eval_results['cv']
            top_rate = eval_results['top_rate']

            if cv == float('inf') or top_rate == 0:
                reward = -10.0
            else:
                reward = -cv
                if cv < 0.5:
                    reward += 1.0

        except Exception as e:
            reward = -10.0
            cv = float('inf')
            top_rate = 0

        # 4. Update
        loss = -reward * log_prob

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        history.append({
            'iteration': iteration,
            'structure': ft_data['structure'],
            'cv': cv if cv != float('inf') else 999,
            'top_rate': top_rate,
            'reward': reward,
            'ranges': [a_min, a_max, b_min, b_max]
        })

        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration:3d} | {ft_data['structure']:8s} | "
                  f"CV={cv:.3f} | Î±=[{a_min:.1f},{a_max:.1f}] Î²=[{b_min:.2f},{b_max:.2f}]")

    return model
