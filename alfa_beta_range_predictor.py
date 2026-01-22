import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from is_optimizer_evaluator import simulate_CTMC

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
        root_idx = len(self.nodes) - 1

        def evaluate(node_idx, state):
            node = self.nodes[node_idx]

            if node['type'] == 'component':
                return state[node['name']] == 1

            children = [src for src, dst in self.edges if dst == node_idx]
            results = [evaluate(c, state) for c in children]

            if node['type'] == 'AND':
                return all(results) if results else False
            if node['type'] == 'OR':
                return any(results) if results else False
            return False

        return lambda state: evaluate(root_idx, state)


def generate_simple_fault_tree(
        n_components_range=(15, 100),
        lambda_range=(1e-4, 1e-2),
        mu_range=(0.05, 0.2),
        max_children_per_gate=5,
        min_children_per_gate=2
):
    """
    Genera fault tree scalabili con 15-100 componenti.

    Struttura:
    - Genera n componenti con lambda/mu randomici nel range specificato
    - Costruisce una topologia gerarchica bottom-up
    - Usa gate AND, OR e KooN (k-out-of-n) in modo randomico

    Args:
        n_components_range: (min, max) numero di componenti
        lambda_range: (min, max) per failure rate
        mu_range: (min, max) per repair rate
        max_children_per_gate: massimo figli per gate
        min_children_per_gate: minimo figli per gate

    Returns:
        dict con graph, fault_tree, lambda_, mu_, structure
    """

    n_components = random.randint(*n_components_range)
    graph = FaultTreeGraph()

    # 1. Genera tutti i componenti con lambda/mu randomici
    component_indices = []
    for i in range(n_components):
        lambda_ = random.uniform(*lambda_range)
        mu_ = random.uniform(*mu_range)
        idx = graph.add_component(f'C{i}', lambda_, mu_)
        component_indices.append(idx)

    # 2. Costruisci la topologia bottom-up
    current_level = component_indices.copy()
    level = 0

    while len(current_level) > 1:
        next_level = []
        random.shuffle(current_level)

        i = 0
        while i < len(current_level):
            remaining = len(current_level) - i

            # Determina quanti figli per questo gate
            if remaining <= max_children_per_gate:
                n_children = remaining
            else:
                n_children = random.randint(
                    min_children_per_gate,
                    min(max_children_per_gate, remaining)
                )

            # Assicurati di avere almeno min_children
            n_children = max(min_children_per_gate, min(n_children, remaining))

            if n_children < min_children_per_gate:
                # Non abbastanza nodi, passali al livello successivo
                next_level.extend(current_level[i:])
                break

            children = current_level[i:i + n_children]

            # Scegli tipo di gate randomicamente
            gate_type = _choose_gate_type(n_children)

            if gate_type == 'KooN':
                # Implementa KooN come combinazione di AND e OR
                k = random.randint(2, n_children - 1)
                gate_idx = _add_koon_gate(graph, children, k)
            else:
                gate_idx = graph.add_gate(gate_type, children)

            next_level.append(gate_idx)
            i += n_children

        current_level = next_level
        level += 1

        # Safety: evita loop infiniti
        if level > 20:
            break

    # 3. Se rimane più di un nodo, crea root finale
    if len(current_level) > 1:
        gate_type = random.choice(['AND', 'OR'])
        graph.add_gate(gate_type, current_level)

    lambda_dict, mu_dict = graph.get_lambda_mu()
    fault_tree = graph.get_logic_function()

    # Calcola statistiche
    n_and = sum(1 for n in graph.nodes if n.get('type') == 'AND')
    n_or = sum(1 for n in graph.nodes if n.get('type') == 'OR')
    structure = f"random_{n_components}comp_{n_and}AND_{n_or}OR"

    return {
        'graph': graph,
        'fault_tree': fault_tree,
        'lambda_': lambda_dict,
        'mu_': mu_dict,
        'structure': structure
    }


def _choose_gate_type(n_children):
    """
    Sceglie il tipo di gate in base al numero di figli.

    - AND e OR sempre disponibili
    - KooN solo se n_children >= 3
    """
    if n_children >= 3:
        # 40% AND, 40% OR, 20% KooN
        r = random.random()
        if r < 0.4:
            return 'AND'
        elif r < 0.8:
            return 'OR'
        else:
            return 'KooN'
    else:
        # Solo AND o OR
        return random.choice(['AND', 'OR'])


def _add_koon_gate(graph, children, k):
    """
    Implementa un gate k-out-of-n usando combinazioni di AND e OR.

    k-out-of-n significa: almeno k dei n figli devono essere in fault.
    Equivale a: OR di tutti gli AND di k figli.

    Per evitare esplosione combinatorica con n grande, usa approssimazione:
    - Se combinazioni <= 10: implementazione esatta
    - Altrimenti: campiona subset di combinazioni
    """
    from itertools import combinations

    n = len(children)
    all_combos = list(combinations(range(n), k))

    # Limita il numero di combinazioni per evitare esplosione
    max_combos = 10
    if len(all_combos) > max_combos:
        selected_combos = random.sample(all_combos, max_combos)
    else:
        selected_combos = all_combos

    # Crea un AND per ogni combinazione
    and_gates = []
    for combo in selected_combos:
        combo_children = [children[i] for i in combo]
        and_idx = graph.add_gate('AND', combo_children)
        and_gates.append(and_idx)

    # OR finale di tutti gli AND
    if len(and_gates) == 1:
        return and_gates[0]
    else:
        return graph.add_gate('OR', and_gates)

class RangePredictor(nn.Module):
    """
    GNN che predice i range ottimali di alpha e beta.

    Input:
        - grafo del fault tree con features [λ, μ, is_comp, is_AND, is_OR]
        - T (tempo di missione) normalizzato
    Output: [alpha_min, alpha_max, beta_min, beta_max]

    NOVITÀ: T è un input! Questo permette alla rete di imparare che:
    - T piccolo → α alto (evento raro, serve biasing forte)
    - T grande → α → 1 (evento più probabile, meno biasing)
    """

    def __init__(self, node_features=5, hidden_dim=64, embedding_dim=32):
        super().__init__()

        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)

        # +7 per features globali: n_comp, n_AND, n_OR, depth, avg_lambda, avg_mu, T_normalized
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + 7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        self.log_std = nn.Parameter(torch.zeros(4))

    def compute_global_features(self, data, T_normalized):
        """Calcola features globali del grafo + T."""
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
        """
        Forward pass.

        Args:
            data: PyG Data object
            T: tempo di missione
            T_max: tempo massimo per normalizzazione (default 500)
        """
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

        # Normalizza T in [0, 1]
        T_normalized = T / T_max

        # Aggiungi features globali + T
        global_features = self.compute_global_features(data, T_normalized)
        embedding = torch.cat([embedding, global_features], dim=1)

        raw = self.predictor(embedding)

        # Vincola output con softplus
        val = F.softplus(raw)

        # Per T grande, vogliamo α e β più vicini a 1
        # Usiamo T_normalized per scalare i range
        # T piccolo (T_norm ~ 0) → range ampi (α alto, β variabile)
        # T grande (T_norm ~ 1) → range stretti verso 1
        scale_factor = 1.0 - 0.7 * T_normalized  # da 1.0 a 0.3

        # Alpha: range più contenuti (come prima)
        alpha_min = val[:, 0] + 1.0
        alpha_max = alpha_min + val[:, 1] + 1.0

        # Scala con T
        alpha_min = 1.0 + (alpha_min - 1.0) * scale_factor
        alpha_max = 1.0 + (alpha_max - 1.0) * scale_factor

        # Beta: parte da 1 e può salire (per T piccoli)
        # Per T grandi, converge a 1
        beta_min = 1.0 + val[:, 2] * scale_factor * 0.3  # parte da 1, può salire fino a ~1.3
        beta_max = beta_min + (val[:, 3] + 0.1) * scale_factor * 0.3

        out = torch.stack([alpha_min, alpha_max, beta_min, beta_max], dim=1)

        return out, embedding

    def get_ranges(self, raw, T_normalized=0.5):
        """Converte raw output in range."""
        scale_factor = 1.0 - 0.7 * T_normalized

        alpha_min = 1.0 + torch.sigmoid(raw[:, 0]) * 3.0 * scale_factor
        alpha_max = alpha_min + torch.sigmoid(raw[:, 1]) * 4.0 * scale_factor

        # Beta parte da 1 e può salire per T piccoli, converge a 1 per T grandi
        beta_min = 1.0 + torch.sigmoid(raw[:, 2]) * 0.5 * scale_factor
        beta_max = beta_min + torch.sigmoid(raw[:, 3]) * 0.3 * scale_factor

        return alpha_min, alpha_max, beta_min, beta_max

    def sample_ranges(self, raw):
        """Campiona range con rumore per esplorazione."""
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(raw, std)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled).sum(dim=-1)

        return sampled, log_prob


def evaluate_ranges(lambda_, mu_, T, fault_tree, alpha_min, alpha_max, beta_min, beta_max, n_eval):
    """
    Valuta la qualità dei range predetti eseguendo IS.
    """
    comps = list(lambda_.keys())

    alpha = {c: (alpha_min + alpha_max) / 2 for c in comps}
    beta = {c: (beta_min + beta_max) / 2 for c in comps}

    results = [simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree)
               for _ in range(n_eval)]

    weights = np.array([math.exp(r['log_w']) if r['top'] else 0.0 for r in results])
    n_top = sum(1 for r in results if r['top'])

    p_is = np.mean(weights)
    std_is = np.std(weights) / np.sqrt(n_eval)
    cv = std_is / p_is if p_is > 0 else float('inf')
    var_is = np.var(weights)

    # MC per confronto
    results_mc = []
    alpha_mc = {c: 1.0 for c in comps}
    beta_mc = {c: 1.0 for c in comps}
    for _ in range(n_eval):
        res = simulate_CTMC(lambda_, mu_, alpha_mc, beta_mc, T, fault_tree)
        results_mc.append(1.0 if res['top'] else 0.0)

    p_mc = np.mean(results_mc)
    var_mc = np.var(results_mc)

    return {
        'p_is': p_is,
        'p_mc': p_mc,
        'std_is': std_is,
        'cv': cv,
        'n_top': n_top,
        'top_rate': n_top / n_eval,
        'var_is': var_is,
        'var_mc': var_mc
    }


def train_range_predictor(n_iterations=300, T_range=(10, 500), verbose=True):
    """
    Addestra il RangePredictor con T VARIABILE.

    Args:
        n_iterations: numero di iterazioni di training
        T_range: (T_min, T_max) per campionare T casuali
        verbose: stampa progresso

    La rete impara che:
    - T piccolo → serve α alto (biasing forte)
    - T grande → α → 1 (poco biasing)
    """

    model = RangePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    T_min, T_max = T_range
    history = []

    print("=" * 60)
    print("TRAINING GNN - PREDIZIONE RANGE α e β (T VARIABILE)")
    print(f"Range T: [{T_min}, {T_max}]")
    print("=" * 60)

    for iteration in range(n_iterations):
        # 1. Genera fault tree
        ft_data = generate_simple_fault_tree()
        pyg_data = ft_data['graph'].to_pyg_data().to(device)

        # 2. Campiona T casuale
        T = random.uniform(T_min, T_max)

        # 3. Predici range con T
        raw, _ = model(pyg_data, T=T, T_max=T_max)
        sampled_raw, log_prob = model.sample_ranges(raw)

        T_normalized = T / T_max
        alpha_min, alpha_max, beta_min, beta_max = model.get_ranges(sampled_raw, T_normalized)

        a_min, a_max = alpha_min.item(), alpha_max.item()
        b_min, b_max = beta_min.item(), beta_max.item()

        # 4. Valuta
        try:
            eval_results = evaluate_ranges(
                ft_data['lambda_'], ft_data['mu_'], T, ft_data['fault_tree'],
                a_min, a_max, b_min, b_max, n_eval=1000
            )

            cv = eval_results['cv']
            top_rate = eval_results['top_rate']
            var_is = eval_results['var_is']
            var_mc = eval_results['var_mc']
            p_is = eval_results['p_is']
            p_mc = eval_results['p_mc']

            if cv == float('inf') or top_rate == 0:
                reward = -20.0
            else:
                # Reward base: vogliamo CV basso
                reward = -cv

                # Confronto IS vs MC
                if var_is > var_mc and var_mc > 0:
                    # IS peggiore di MC: penalità
                    reward = -5.0 * (var_is / (var_mc + 1e-9))
                elif var_mc > 0:
                    # IS migliore di MC: bonus
                    gain = var_mc / (var_is + 1e-9)
                    reward = 1.0 + math.log(gain + 1.0)

                # BONUS per α vicino a 1 quando T è grande
                # Questo incentiva la rete a imparare α → 1 per T grandi
                avg_alpha = (a_min + a_max) / 2
                avg_beta = (b_min + b_max) / 2
                if T_normalized > 0.5:  # T > metà del range
                    # Più T è grande, più premiamo α vicino a 1
                    alpha_penalty = (avg_alpha - 1.0) ** 2 * T_normalized
                    beta_penalty = (avg_beta - 1.0) ** 2 * T_normalized
                    reward -= alpha_penalty
                    reward -= beta_penalty

                # BONUS per accuratezza stima (IS vicino a MC)
                if p_mc > 0:
                    relative_error = abs(p_is - p_mc) / p_mc
                    if relative_error < 0.3:  # Errore < 30%
                        reward += 2.0
                    elif relative_error > 1.0:  # Errore > 100%
                        reward -= 2.0

        except Exception as e:
            reward = -10.0
            cv = float('inf')
            top_rate = 0
            a_min, a_max, b_min, b_max = 1.0, 2.0, 0.5, 1.0

        # 5. Update
        loss = -reward * log_prob

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        history.append({
            'iteration': iteration,
            'T': T,
            'structure': ft_data['structure'],
            'cv': cv if cv != float('inf') else 999,
            'top_rate': top_rate,
            'reward': reward,
            'ranges': [a_min, a_max, b_min, b_max]
        })

        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration:3d} | T={T:5.0f} | {ft_data['structure']:12s} | "
                  f"CV={cv:.3f} | α=[{a_min:.2f},{a_max:.2f}] β=[{b_min:.2f},{b_max:.2f}] | R={reward:.2f}")

    return model
