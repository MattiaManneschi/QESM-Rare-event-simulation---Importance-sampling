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
    Genera fault tree con varietà strutturale (P ~ 10^-1 / 10^-2).

    Strutture:
    - OR_2, OR_3, OR_4, OR_5: semplici OR
    - AND_2, AND_3, AND_4, AND_5: semplici AND (richiedono α diversi!)
    - 2oo3, 2oo4, 3oo5: voting
    - mixed_*: combinazioni AND/OR
    - hierarchical_*: strutture gerarchiche
    """

    base_mu = 0.1

    # Varia λ/μ per avere P diverse
    ratio = random.uniform(0.05, 0.20)
    base_lambda = base_mu * ratio

    def vary(val):
        return val * random.uniform(0.8, 1.2)

    def add_n_components(graph, n, prefix='C'):
        """Aggiunge n componenti e ritorna gli indici."""
        return [graph.add_component(f'{prefix}{i}', vary(base_lambda), vary(base_mu))
                for i in range(n)]

    # Scegli tipo di struttura con pesi
    structure_types = [
        # OR semplici (peso basso - facili)
        'OR_2', 'OR_3', 'OR_4', 'OR_5',
        # AND semplici (peso alto - interessanti per α)
        'AND_2', 'AND_2', 'AND_3', 'AND_3', 'AND_4', 'AND_5',
        # Voting
        '2oo3', '2oo4', '3oo5',
        # Mixed
        'mixed_AND_OR', 'mixed_OR_AND', 'mixed_complex',
        # Gerarchici
        'hier_AND_AND', 'hier_OR_OR', 'hier_AND_OR', 'hier_OR_AND'
    ]

    structure_type = random.choice(structure_types)
    graph = FaultTreeGraph()

    # -------------------------------------------------------------------------
    # OR semplici
    # -------------------------------------------------------------------------
    if structure_type == 'OR_2':
        nodes = add_n_components(graph, 2)
        graph.add_gate('OR', nodes)

    elif structure_type == 'OR_3':
        nodes = add_n_components(graph, 3)
        graph.add_gate('OR', nodes)

    elif structure_type == 'OR_4':
        nodes = add_n_components(graph, 4)
        graph.add_gate('OR', nodes)

    elif structure_type == 'OR_5':
        nodes = add_n_components(graph, 5)
        graph.add_gate('OR', nodes)

    # -------------------------------------------------------------------------
    # AND semplici (richiedono α diversi in base a n!)
    # -------------------------------------------------------------------------
    elif structure_type == 'AND_2':
        nodes = add_n_components(graph, 2)
        graph.add_gate('AND', nodes)

    elif structure_type == 'AND_3':
        nodes = add_n_components(graph, 3)
        graph.add_gate('AND', nodes)

    elif structure_type == 'AND_4':
        nodes = add_n_components(graph, 4)
        graph.add_gate('AND', nodes)

    elif structure_type == 'AND_5':
        nodes = add_n_components(graph, 5)
        graph.add_gate('AND', nodes)

    # -------------------------------------------------------------------------
    # Voting (k-out-of-n)
    # -------------------------------------------------------------------------
    elif structure_type == '2oo3':
        # 2 su 3: (A∧B) ∨ (A∧C) ∨ (B∧C)
        nodes = add_n_components(graph, 3)
        and_01 = graph.add_gate('AND', [nodes[0], nodes[1]])
        and_02 = graph.add_gate('AND', [nodes[0], nodes[2]])
        and_12 = graph.add_gate('AND', [nodes[1], nodes[2]])
        graph.add_gate('OR', [and_01, and_02, and_12])

    elif structure_type == '2oo4':
        # 2 su 4
        nodes = add_n_components(graph, 4)
        ands = []
        for i in range(4):
            for j in range(i + 1, 4):
                ands.append(graph.add_gate('AND', [nodes[i], nodes[j]]))
        graph.add_gate('OR', ands)

    elif structure_type == '3oo5':
        # 3 su 5 (semplificato: solo alcune combinazioni)
        nodes = add_n_components(graph, 5)
        ands = []
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    ands.append(graph.add_gate('AND', [nodes[i], nodes[j], nodes[k]]))
        graph.add_gate('OR', ands)

    # -------------------------------------------------------------------------
    # Mixed: combinazioni AND/OR
    # -------------------------------------------------------------------------
    elif structure_type == 'mixed_AND_OR':
        # (A ∧ B) ∨ C ∨ D
        nodes = add_n_components(graph, 4)
        and_gate = graph.add_gate('AND', [nodes[0], nodes[1]])
        graph.add_gate('OR', [and_gate, nodes[2], nodes[3]])

    elif structure_type == 'mixed_OR_AND':
        # (A ∨ B) ∧ (C ∨ D)
        nodes = add_n_components(graph, 4)
        or1 = graph.add_gate('OR', [nodes[0], nodes[1]])
        or2 = graph.add_gate('OR', [nodes[2], nodes[3]])
        graph.add_gate('AND', [or1, or2])

    elif structure_type == 'mixed_complex':
        # (A ∧ B ∧ C) ∨ (D ∧ E)
        nodes = add_n_components(graph, 5)
        and1 = graph.add_gate('AND', [nodes[0], nodes[1], nodes[2]])
        and2 = graph.add_gate('AND', [nodes[3], nodes[4]])
        graph.add_gate('OR', [and1, and2])

    # -------------------------------------------------------------------------
    # Gerarchici: strutture a più livelli
    # -------------------------------------------------------------------------
    elif structure_type == 'hier_AND_AND':
        # (A ∧ B) ∧ (C ∧ D) - profondità AND
        nodes = add_n_components(graph, 4)
        and1 = graph.add_gate('AND', [nodes[0], nodes[1]])
        and2 = graph.add_gate('AND', [nodes[2], nodes[3]])
        graph.add_gate('AND', [and1, and2])

    elif structure_type == 'hier_OR_OR':
        # (A ∨ B) ∨ (C ∨ D)
        nodes = add_n_components(graph, 4)
        or1 = graph.add_gate('OR', [nodes[0], nodes[1]])
        or2 = graph.add_gate('OR', [nodes[2], nodes[3]])
        graph.add_gate('OR', [or1, or2])

    elif structure_type == 'hier_AND_OR':
        # (A ∧ B) ∨ (C ∧ D) - AND sotto OR
        nodes = add_n_components(graph, 4)
        and1 = graph.add_gate('AND', [nodes[0], nodes[1]])
        and2 = graph.add_gate('AND', [nodes[2], nodes[3]])
        graph.add_gate('OR', [and1, and2])

    elif structure_type == 'hier_OR_AND':
        # (A ∨ B) ∧ (C ∨ D) - OR sotto AND
        nodes = add_n_components(graph, 4)
        or1 = graph.add_gate('OR', [nodes[0], nodes[1]])
        or2 = graph.add_gate('OR', [nodes[2], nodes[3]])
        graph.add_gate('AND', [or1, or2])

    lambda_, mu_ = graph.get_lambda_mu()
    fault_tree = graph.get_logic_function()

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

    Usa features globali del grafo per catturare la struttura.
    """

    def __init__(self, node_features=5, hidden_dim=64, embedding_dim=32):
        super().__init__()

        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)

        # +6 per features globali: n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + 6, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        self.log_std = nn.Parameter(torch.zeros(4))

    def compute_global_features(self, data):
        """Calcola features globali del grafo."""
        x = data.x

        # Conta tipi di nodi
        n_comp = x[:, 2].sum().item()  # is_component
        n_AND = x[:, 3].sum().item()  # is_AND
        n_OR = x[:, 4].sum().item()  # is_OR

        # Media lambda e mu (solo componenti)
        comp_mask = x[:, 2] == 1
        if comp_mask.sum() > 0:
            avg_lambda = x[comp_mask, 0].mean().item()
            avg_mu = x[comp_mask, 1].mean().item()
        else:
            avg_lambda = 0
            avg_mu = 0

        # Profondità approssimata (numero di gate)
        depth = n_AND + n_OR

        return torch.tensor([[n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu]],
                            dtype=torch.float, device=x.device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GNN message passing
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Pooling
        embedding = global_mean_pool(x, batch)

        # Aggiungi features globali
        global_features = self.compute_global_features(data)
        embedding = torch.cat([embedding, global_features], dim=1)

        raw = self.predictor(embedding)

        # Vincola output con softplus
        val = F.softplus(raw)

        alpha_min = val[:, 0] + 1.0
        alpha_max = alpha_min + val[:, 1] + 1.0

        beta_min = val[:, 2] + 0.1
        beta_max = beta_min + val[:, 3] + 0.1

        out = torch.stack([alpha_min, alpha_max, beta_min, beta_max], dim=1)

        return out, embedding

    def get_ranges(self, raw):
        # Usiamo il Sigmoid per mappare l'uscita tra 1.0 e 2.5 invece che 1.0 e 5.0
        # Questo impedisce alla rete di essere troppo "violenta"
        alpha_min = 1.0 + torch.sigmoid(raw[:, 0]) * 1.0  # Range: [1.0, 2.0]
        alpha_max = alpha_min + torch.sigmoid(raw[:, 1]) * 1.0  # Range: [alpha_min, alpha_min + 1.0]

        # Beta (riparazione) dovrebbe stare vicino a 1.0 se non strettamente necessario
        beta_min = 0.9 + torch.sigmoid(raw[:, 2]) * 0.2
        beta_max = beta_min + torch.sigmoid(raw[:, 3]) * 0.2

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
    var_is = np.var(results)

    results_mc = []
    for _ in range(n_eval):
        # Biasing neutro per MC
        a_mc = {c: 1.0 for c in lambda_}
        b_mc = {c: 1.0 for c in mu_}
        res = simulate_CTMC(lambda_, mu_, a_mc, b_mc, T, fault_tree.get_logic_function())
        results_mc.append(1.0 if res['top'] else 0.0)

    var_mc = np.var(results_mc)

    return {
        'p_is': p_is,
        'std_is': std_is,
        'cv': cv,
        'n_top': n_top,
        'top_rate': n_top / n_eval,
        'var_is': var_is,
        'var_mc': var_mc
    }

def train_range_predictor(n_iterations=200, T=100, verbose=True):

    model = RangePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = []

    print("=" * 60)
    print("TRAINING GNN - PREDIZIONE RANGE α e β")
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
            var_is = eval_results['var_is']
            var_mc = eval_results['var_mc']

            if cv == float('inf') or eval_results['top_rate'] == 0:
                reward = -20.0  # Penalità massima per fallimento totale
            else:
                # Il reward base è l'inverso del CV (vogliamo CV basso)
                reward = -cv

                # SEZIONE CRUCIALE: Confronto IS vs MC
                # Se la varianza IS è più alta di quella MC, l'IA sta facendo danni
                if var_is > var_mc:
                    reward = -5.0 * (var_is / (var_mc + 1e-9))
                else:
                    gain = var_mc / (var_is + 1e-9)
                    reward = 1.0 + math.log(gain + 1.0)

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
                  f"CV={cv:.3f} | α=[{a_min:.1f},{a_max:.1f}] β=[{b_min:.2f},{b_max:.2f}]")

    return model
