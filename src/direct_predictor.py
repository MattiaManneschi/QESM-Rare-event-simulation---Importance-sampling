import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaultTreeGraph:

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_features = []
        self.components = {}

    def add_component(self, name, lambda_, mu_):
        idx = len(self.nodes)
        self.nodes.append({'type': 'component', 'name': name, 'idx': idx})
        self.node_features.append([lambda_, mu_, 1, 0, 0])
        self.components[name] = idx
        return idx

    def add_gate(self, gate_type, children_idx):
        idx = len(self.nodes)
        self.nodes.append({'type': gate_type, 'idx': idx, 'inputs': children_idx})

        if gate_type == 'AND':
            self.node_features.append([0, 0, 0, 1, 0])
        else:
            self.node_features.append([0, 0, 0, 0, 1])

        for child_idx in children_idx:
            self.edges.append((child_idx, idx))

        return idx

    def to_pyg_data(self):
        x = torch.tensor(self.node_features, dtype=torch.float)
        if self.edges:
            edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def get_lambda_mu(self):
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

    def get_component_names(self):
        return [node['name'] for node in self.nodes if node['type'] == 'component']

    def get_component_indices(self):
        return [node['idx'] for node in self.nodes if node['type'] == 'component']

    def get_min_distance_to_top_event(self):
        adj_inv = {i: [] for i in range(len(self.nodes))}
        for child, parent in self.edges:
            adj_inv[parent].append(child)

        root_idx = len(self.nodes) - 1

        distances = {root_idx: 0}
        queue = [root_idx]

        while queue:
            curr = queue.pop(0)
            curr_dist = distances[curr]

            for child in adj_inv[curr]:
                if child not in distances:
                    distances[child] = curr_dist + 1
                    queue.append(child)

        component_depths = {}
        for node in self.nodes:
            if node['type'] == 'component':
                name = node['name']
                idx = node['idx']

                component_depths[name] = distances.get(idx, 5)

        return component_depths


def compute_component_criticality(graph):
    child_to_parents = defaultdict(list)
    for src, dst in graph.edges:
        child_to_parents[src].append(dst)

    root_idx = len(graph.nodes) - 1
    criticality = {}

    for node in graph.nodes:
        if node['type'] != 'component':
            continue

        comp_name = node['name']
        comp_idx = node['idx']

        n_and = 0
        n_or = 0
        current = comp_idx
        visited = set()

        while current != root_idx and current not in visited:
            visited.add(current)
            parents = child_to_parents.get(current, [])
            if not parents:
                break
            parent_idx = parents[0]
            parent_node = graph.nodes[parent_idx]

            if parent_node['type'] == 'AND':
                n_and += 1
            elif parent_node['type'] == 'OR':
                n_or += 1

            current = parent_idx

        if n_and + n_or > 0:
            criticality[comp_name] = n_and / (n_and + n_or)
        else:
            criticality[comp_name] = 0.5

    if criticality:
        min_crit = min(criticality.values())
        max_crit = max(criticality.values())
        if max_crit > min_crit:
            criticality = {c: (v - min_crit) / (max_crit - min_crit)
                          for c, v in criticality.items()}

    return criticality


def simulate_CTMC_simple(lambda_, mu_, alpha, beta, T, fault_tree):
    comps = list(lambda_.keys())
    state = {c: 0 for c in comps}

    t = 0.0
    log_w = 0.0
    top_event_hit = False
    n_transitions = 0
    max_transitions = 5000 

    while t < T and n_transitions < max_transitions:
        rates_orig = {}
        rates_is = {}

        for c in comps:
            if state[c] == 0:
                rates_orig[('fail', c)] = lambda_[c]
                rates_is[('fail', c)] = lambda_[c] * alpha[c]
            else:
                rates_orig[('repair', c)] = mu_[c]
                rates_is[('repair', c)] = mu_[c] / beta[c]

        R_orig = sum(rates_orig.values())
        R_is = sum(rates_is.values())

        if R_is <= 0: break

        dt = np.random.exponential(1.0 / R_is)

        if t + dt > T:
            log_w += (R_is - R_orig) * (T - t)
            break

        t += dt
        log_w += (R_is - R_orig) * dt + np.log(R_orig / R_is)

        r = np.random.random() * R_is
        cumsum = 0.0
        chosen_event = None
        for event, rate in rates_is.items():
            cumsum += rate
            if r <= cumsum:
                chosen_event = event
                break

        if chosen_event is None: break

        log_w += np.log(rates_orig[chosen_event] / rates_is[chosen_event]) + np.log(R_is / R_orig)

        event_type, comp = chosen_event
        state[comp] = 1 if event_type == 'fail' else 0

        if fault_tree(state):
            top_event_hit = True
            break

        n_transitions += 1

    return {
        'top': top_event_hit,
        'log_w': log_w,
        't_final': t
    }


class DirectPredictor(nn.Module):
    def __init__(self, node_features=5, hidden_dim=64, n_layers=4):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(node_features, hidden_dim, heads=4, concat=False))
        for _ in range(n_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))

        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.log_std_alpha = nn.Parameter(torch.ones(1) * 0.7)
        self.log_std_beta = nn.Parameter(torch.zeros(1))

    def forward(self, data, T, T_max=500.0, sample=False):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        T_normalized = T / T_max
        time_decay = (1.0 - T_normalized) ** 1.5

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.1, training=self.training)

        global_context = global_mean_pool(x, batch)
        global_context = self.global_encoder(global_context)

        if global_context.size(0) == 1:
            global_expanded = global_context.expand(x.size(0), -1)
        else:
            global_expanded = global_context[batch]

        node_features = torch.cat([x, global_expanded], dim=1)

        alpha_raw = self.alpha_head(node_features).squeeze(-1)
        beta_raw = self.beta_head(node_features).squeeze(-1)

        alpha_base = 1.0 + F.softplus(alpha_raw) * 15.0

        beta_mean = 1.0 + F.softplus(beta_raw) * 0.5

        alpha_mean = 1.0 + (alpha_base - 1.0) * time_decay

        if sample:

            std_alpha = (torch.exp(self.log_std_alpha) * (1.0 + alpha_mean * 0.1)).clamp(0.01, 30.0)
            std_beta = torch.exp(self.log_std_beta).clamp(0.01, 0.5)

            dist_alpha = torch.distributions.Normal(alpha_mean, std_alpha)
            dist_beta = torch.distributions.Normal(beta_mean, std_beta)

            alpha = dist_alpha.rsample().clamp(min=1.0)
            beta = dist_beta.rsample().clamp(min=1.0)

            log_prob = dist_alpha.log_prob(alpha).sum() + dist_beta.log_prob(beta).sum()

            return alpha, beta, log_prob
        else:
            return alpha_mean, beta_mean, None

    def predict(self, graph, T, T_max=500.0):
        self.eval()
        device = next(self.parameters()).device
        pyg_data = graph.to_pyg_data().to(device)
        component_names = graph.get_component_names()
        component_indices = graph.get_component_indices()

        with torch.no_grad():
            alpha_all, beta_all, _ = self.forward(pyg_data, T, T_max, sample=False)

        alpha_dict = {name: alpha_all[idx].item() for name, idx in zip(component_names, component_indices)}
        beta_dict = {name: beta_all[idx].item() for name, idx in zip(component_names, component_indices)}

        return alpha_dict, beta_dict


def compute_target_alpha_beta(graph, T, T_max):
    criticality = compute_component_criticality(graph)
    depths = graph.get_min_distance_to_top_event()

    component_names = graph.get_component_names()
    n_and = sum(1 for n in graph.nodes if n.get('type') == 'AND')

    T_normalized = T / T_max

    time_decay = (1.0 - T_normalized) ** 1.5

    base_alpha = 5.0 + n_and * 2.0

    target_alpha = {}
    target_beta = {}

    for name in component_names:
        crit = criticality.get(name, 0.5)
        d = depths.get(name, 1)

        depth_factor = 0.8 + (d * 0.2)

        alpha_base = base_alpha * crit * depth_factor
        alpha_val = 1.0 + alpha_base * time_decay

        target_alpha[name] = np.clip(alpha_val, 1.0, 250.0)

        target_beta[name] = 1.0 + (0.5 * crit * time_decay)
        target_beta[name] = max(1.0, target_beta[name])

    return target_alpha, target_beta


import numpy as np
import math


def compute_reward(alpha_dict, beta_dict, target_alpha, graph, fault_tree, T, n_simulations=500):
    lambda_, mu_ = graph.get_lambda_mu()
    component_names = graph.get_component_names()

    results = []
    for _ in range(n_simulations):
        r = simulate_CTMC_simple(lambda_, mu_, alpha_dict, beta_dict, T, fault_tree)
        results.append(r)

    all_weights = []
    for r in results:
        if r['top']:
            w = math.exp(r['log_w'])
            all_weights.append(w if not (math.isnan(w) or math.isinf(w)) else 0.0)
        else:
            all_weights.append(0.0)

    all_weights = np.array(all_weights)
    n_top = np.sum(all_weights > 0)
    top_rate = n_top / n_simulations

    if n_top == 0:
        mse_dist = np.mean([
            (alpha_dict[name] - target_alpha[name]) ** 2
            for name in component_names if name in target_alpha
        ])

        shaped_penalty = -100.0 + (90.0 / (1.0 + mse_dist))
        return shaped_penalty, float('inf'), 0.0

    mean_w = np.mean(all_weights)
    std_w = np.std(all_weights)

    if mean_w <= 1e-18:
        return -50.0, float('inf'), top_rate

    cv = std_w / mean_w

    target_cv = 0.2
    base_reward = 100.0 * np.exp(-cv / target_cv)

    if cv > 2.0:
        base_reward -= (cv * 5.0)

    if 0.1 <= top_rate <= 0.4:
        base_reward += 20.0
    elif top_rate > 0.6:
        base_reward -= 20.0  
    elif top_rate < 0.05:
        base_reward -= 10.0  

    return base_reward, cv, top_rate


def train_direct_predictor_hybrid(
    n_iterations_supervised=1500,
    n_iterations_rl=500,
    T_range=(10, 500),
    comp_range=(5, 45),
    pretrained_model=None,
    n_simulations_rl=2000,
    verbose=True
):
    from src.fault_tree_generator import generate_rare_event_fault_tree

    if pretrained_model is not None:
        model = pretrained_model.to(device)
        print("[Hybrid] Fine-tuning da modello pre-addestrato")
    else:
        model = DirectPredictor().to(device)
        print("[Hybrid] Training da zero")

    T_min, T_max = T_range

    print("\n" + "=" * 70)
    print("FASE 1: SUPERVISED LEARNING")
    print(f"Iterazioni: {n_iterations_supervised}")
    print("Loss: MSE verso target euristici")
    print("=" * 70)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    for iteration in range(n_iterations_supervised):
        model.train()

        ft_data = generate_rare_event_fault_tree(comp_range, target_p_order=-5)
        graph = ft_data['graph']
        pyg_data = graph.to_pyg_data().to(device)

        T = random.uniform(T_min, T_max)

        alpha_pred, beta_pred, _ = model.forward(pyg_data, T, T_max, sample=False)

        target_alpha, target_beta = compute_target_alpha_beta(graph, T, T_max)
        component_names = graph.get_component_names()
        component_indices = graph.get_component_indices()

        alpha_target = torch.tensor([target_alpha[name] for name in component_names],
                                    dtype=torch.float, device=device)
        beta_target = torch.tensor([target_beta[name] for name in component_names],
                                   dtype=torch.float, device=device)

        alpha_pred_comp = alpha_pred[component_indices]
        beta_pred_comp = beta_pred[component_indices]

        loss = F.mse_loss(torch.log(alpha_pred_comp), torch.log(alpha_target)) + F.mse_loss(torch.log(beta_pred_comp), torch.log(beta_target))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if verbose and iteration % 100 == 0:
            n_and = sum(1 for n in graph.nodes if n.get('type') == 'AND')
            avg_alpha = alpha_pred_comp.mean().item()
            avg_alpha_tgt = alpha_target.mean().item()
            avg_beta = beta_pred_comp.mean().item()
            avg_beta_tgt = beta_target.mean().item()

            print(f"[SUP] Iter {iteration:4d} | T={T:5.0f} | {n_and:2d}AND | "
                  f"α: {avg_alpha:.1f} vs {avg_alpha_tgt:.1f} | "
                  f"β: {avg_beta:.2f} vs {avg_beta_tgt:.2f} | "
                  f"Loss: {loss.item():.4f}")

    print("\n✓ Fase 1 completata!")

    print("\n" + "=" * 70)
    print("FASE 2: REINFORCEMENT LEARNING")
    print(f"Iterazioni: {n_iterations_rl}")
    print(f"Simulazioni per reward: {n_simulations_rl}")
    print("Reward: basato su CV reale")
    print("=" * 70)

    optimizer_rl = optim.Adam(model.parameters(), lr=1e-4)

    baseline = 0.0
    baseline_decay = 0.95

    for iteration in range(n_iterations_rl):
        model.train()

        ft_data = generate_rare_event_fault_tree(comp_range, target_p_order=-5)
        graph = ft_data['graph']
        fault_tree = ft_data['fault_tree']
        pyg_data = graph.to_pyg_data().to(device)

        if random.random() < 0.6:
            T = random.uniform(T_min, T_max * 0.3)  
        else:
            T = random.uniform(T_min, T_max)

        alpha_all, beta_all, log_prob = model.forward(pyg_data, T, T_max, sample=True)

        component_names = graph.get_component_names()
        component_indices = graph.get_component_indices()

        alpha_dict = {name: alpha_all[idx].item() for name, idx in zip(component_names, component_indices)}
        beta_dict = {name: beta_all[idx].item() for name, idx in zip(component_names, component_indices)}

        reward, cv, top_rate = compute_reward(
            alpha_dict, beta_dict, target_alpha, graph, fault_tree, T,
            n_simulations=n_simulations_rl
        )

        baseline = baseline_decay * baseline + (1 - baseline_decay) * reward

        advantage = reward - baseline
        loss = -advantage * log_prob

        optimizer_rl.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer_rl.step()

        if verbose and iteration % 20 == 0:
            n_and = sum(1 for n in graph.nodes if n.get('type') == 'AND')
            avg_alpha = np.mean(list(alpha_dict.values()))
            avg_beta = np.mean(list(beta_dict.values()))
            cv_str = f"{cv:.2f}" if cv != float('inf') else "inf"

            print(f"[RL] Iter {iteration:4d} | T={T:5.0f} | {n_and:2d}AND | "
                  f"ᾱ={avg_alpha:.1f} β̄={avg_beta:.2f} | "
                  f"CV={cv_str:>6s} top={top_rate:.0%} | "
                  f"R={reward:+.1f}")

    print("\n✓ Fase 2 completata!")
    print("=" * 70)

    return model


def train_direct_predictor(n_iterations, T_range, comp_range, pretrained_model=None):
    n_sup = int(n_iterations * 0.75)
    n_rl = n_iterations - n_sup

    return train_direct_predictor_hybrid(
        n_iterations_supervised=n_sup,
        n_iterations_rl=n_rl,
        T_range=T_range,
        comp_range=comp_range,
        pretrained_model=pretrained_model
    )


def train_direct_predictor_incremental(
    stages,
    n_iterations_per_stage=1000,
    supervised_ratio=0.75,
    T_range=(10, 500),
    n_simulations_rl=1000,
    verbose=True
):
    model = None

    for i, comp_range in enumerate(stages):
        print(f"\n{'#' * 70}")
        print(f"# STAGE {i + 1}/{len(stages)}: componenti {comp_range}")
        print(f"{'#' * 70}")

        n_sup = int(n_iterations_per_stage * supervised_ratio)
        n_rl = n_iterations_per_stage - n_sup

        n_sim = max(50, n_simulations_rl - i * 20)

        model = train_direct_predictor_hybrid(
            n_iterations_supervised=n_sup,
            n_iterations_rl=n_rl,
            T_range=T_range,
            comp_range=comp_range,
            pretrained_model=model,
            n_simulations_rl=n_sim,
            verbose=verbose
        )

        print(f"\n✓ Stage {i + 1} completato!")

    print(f"\n{'#' * 70}")
    print(f"# TRAINING INCREMENTALE COMPLETATO")
    print(f"# Stages: {stages}")
    print(f"{'#' * 70}")

    return model
