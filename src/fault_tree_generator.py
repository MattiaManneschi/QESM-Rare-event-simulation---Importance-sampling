import math
import random

from src.direct_predictor import FaultTreeGraph


def generate_rare_event_fault_tree(
        n_components_range,
        lambda_range=(1e-4, 5e-4),
        mu_range=(0.1, 0.5),
        target_p_order=-5,
        structure_type='auto'
):
    max_retries = 20
    best_ft = None
    min_error = float('inf')

    adj_lambda_min, adj_lambda_max = lambda_range
    if target_p_order > -3:
        adj_lambda_min *= 5
        adj_lambda_max *= 5

    for attempt in range(max_retries):
        n_components = random.randint(*n_components_range)

        if structure_type == 'auto':
            if target_p_order >= -3:
                st = random.choice(['mixed', 'hierarchical'])
                heavy_and = False
            else:
                st = random.choice(['series_and', 'deep_series', 'mixed'])
                heavy_and = (target_p_order < -6)
        else:
            st = structure_type
            heavy_and = (target_p_order < -6)

        if st == 'series_and':
            ft_data = _generate_series_and_tree(n_components, (adj_lambda_min, adj_lambda_max), mu_range)
        elif st == 'deep_series':
            ft_data = _generate_deep_series_tree(n_components, (adj_lambda_min, adj_lambda_max), mu_range,
                                                 heavy_and=heavy_and)
        elif st == 'mixed':
            ft_data = _generate_mixed_tree(n_components, (adj_lambda_min, adj_lambda_max), mu_range)
        else:
            ft_data = _generate_hierarchical_tree(n_components, (adj_lambda_min, adj_lambda_max), mu_range)

        log_p = _estimate_tree_log_prob(ft_data['graph'])
        error = abs(log_p - target_p_order)

        if error < min_error:
            min_error = error
            best_ft = ft_data

        current_tolerance = 1.5 + (attempt // 5) * 0.8

        if error <= current_tolerance:
            return ft_data

    return best_ft


def _estimate_tree_log_prob(graph):
    lambda_dict, mu_dict = graph.get_lambda_mu()
    nodes = graph.nodes
    memo = {}

    def get_prob(node_idx):
        if node_idx in memo: return memo[node_idx]
        
        node = nodes[node_idx]
        
        if node['type'] == 'component':
            lam = lambda_dict[node['name']]
            mu = mu_dict[node['name']]

            val = lam / (lam + mu) if (lam + mu) > 0 else 0
            memo[node_idx] = val
            return val
        
        inputs = [get_prob(i) for i in node['inputs']]
        
        if not inputs: return 0.0

        if node['type'] == 'OR':

            val = min(1.0, sum(inputs))
        elif node['type'] == 'AND':

            val = 1.0
            for p in inputs: val *= p
        else:
            val = 0.0
            
        memo[node_idx] = val
        return val

    try:
        top_prob = get_prob(len(nodes) - 1)
        if top_prob <= 0: return -100.0
        return math.log10(top_prob)
    except:
        return -100.0


def _generate_series_and_tree(n_components, lambda_range, mu_range):
    graph = FaultTreeGraph()

    n_subsystems = random.randint(2, max(3, n_components // 2))
    comps_per_subsystem = max(1, n_components // n_subsystems)
    
    subsystem_indices = []
    comp_idx = 0

    for ss in range(n_subsystems):

        ss_components = []

        current_n_comps = comps_per_subsystem + (1 if comp_idx + comps_per_subsystem < n_components and ss == n_subsystems-1 else 0)
        
        for _ in range(current_n_comps):
            if comp_idx >= n_components: break
            lambda_ = random.uniform(*lambda_range)
            mu_ = random.uniform(*mu_range)
            idx = graph.add_component(f'C{comp_idx}', lambda_, mu_)
            ss_components.append(idx)
            comp_idx += 1

        if not ss_components: continue

        if random.random() < 0.6: 
            gate = graph.add_gate('OR', ss_components)
        else:
            gate = graph.add_gate('AND', ss_components)
        subsystem_indices.append(gate)

    if len(subsystem_indices) > 1:
        graph.add_gate('AND', subsystem_indices)
    elif len(subsystem_indices) == 1:

        graph.add_gate('OR', subsystem_indices)

    return _finalize_tree(graph)


def _generate_hierarchical_tree(n_components, lambda_range, mu_range):
    graph = FaultTreeGraph()
    n_or_branches = random.randint(2, 4)
    comp_idx = 0
    or_branch_indices = []

    for _ in range(n_or_branches):
        n_and_in_branch = random.randint(1, 3)
        and_indices = []

        for _ in range(n_and_in_branch):

            comps_per_and = random.randint(1, 3)
            and_components = []
            
            for _ in range(comps_per_and):
                if comp_idx >= n_components: break
                lambda_ = random.uniform(*lambda_range)
                mu_ = random.uniform(*mu_range)
                idx = graph.add_component(f'C{comp_idx}', lambda_, mu_)
                and_components.append(idx)
                comp_idx += 1
            
            if len(and_components) > 1:
                and_indices.append(graph.add_gate('AND', and_components))
            elif len(and_components) == 1:
                and_indices.append(and_components[0])

        if len(and_indices) > 1:
            or_branch_indices.append(graph.add_gate('OR', and_indices))
        elif len(and_indices) == 1:
            or_branch_indices.append(and_indices[0])

    while comp_idx < n_components:
        lambda_ = random.uniform(*lambda_range)
        mu_ = random.uniform(*mu_range)
        idx = graph.add_component(f'C{comp_idx}', lambda_, mu_)
        or_branch_indices.append(idx)
        comp_idx += 1

    if len(or_branch_indices) > 1:
        graph.add_gate('AND', or_branch_indices)
    elif len(or_branch_indices) == 1:
        graph.add_gate('OR', or_branch_indices)

    return _finalize_tree(graph)


def _generate_mixed_tree(n_components, lambda_range, mu_range):
    graph = FaultTreeGraph()
    component_indices = []
    
    for i in range(n_components):
        lambda_ = random.uniform(*lambda_range)
        mu_ = random.uniform(*mu_range)
        idx = graph.add_component(f'C{i}', lambda_, mu_)
        component_indices.append(idx)

    random.shuffle(component_indices)

    level1 = []
    i = 0
    while i < len(component_indices):
        remaining = len(component_indices) - i

        if remaining < 2:
            group_size = 1
        else:
            group_size = random.randint(2, min(3, remaining))

        group = component_indices[i:i + group_size]
        i += group_size

        if len(group) >= 2:
            gtype = 'AND' if random.random() < 0.4 else 'OR'
            level1.append(graph.add_gate(gtype, group))
        else:
            level1.extend(group)

    random.shuffle(level1)
    level2 = []
    i = 0
    while i < len(level1):
        remaining = len(level1) - i
        if remaining < 2:
            group_size = 1
        else:
            group_size = random.randint(2, min(4, remaining))

        group = level1[i:i + group_size]
        i += group_size

        if len(group) >= 2:

            gtype = 'OR' if random.random() < 0.7 else 'AND'
            level2.append(graph.add_gate(gtype, group))
        else:
            level2.extend(group)

    if len(level2) >= 2:
        graph.add_gate('AND', level2)
    elif len(level2) == 1:
        graph.add_gate('OR', level2)

    return _finalize_tree(graph)


def _generate_deep_series_tree(n_components, lambda_range, mu_range, heavy_and=False):
    graph = FaultTreeGraph()
    component_indices = []
    for i in range(n_components):
        lambda_ = random.uniform(*lambda_range)
        mu_ = random.uniform(*mu_range)
        idx = graph.add_component(f'C{i}', lambda_, mu_)
        component_indices.append(idx)

    random.shuffle(component_indices)
    current_level = component_indices

    prob_and = 0.7 if heavy_and else 0.2

    while len(current_level) > 1:
        next_level = []
        i = 0
        while i < len(current_level):
            if i + 1 < len(current_level):

                if random.random() < prob_and:
                    gate = graph.add_gate('AND', [current_level[i], current_level[i+1]])
                else:
                    gate = graph.add_gate('OR', [current_level[i], current_level[i+1]])
                next_level.append(gate)
                i += 2
            else:
                next_level.append(current_level[i])
                i += 1
        current_level = next_level

    return _finalize_tree(graph)


def _finalize_tree(graph):
    lambda_dict, mu_dict = graph.get_lambda_mu()
    fault_tree = graph.get_logic_function()

    n_and = sum(1 for n in graph.nodes if n.get('type') == 'AND')
    n_or = sum(1 for n in graph.nodes if n.get('type') == 'OR')
    n_comp = len(lambda_dict)

    structure = f"rare_{n_comp}comp_{n_and}AND_{n_or}OR"

    return {
        'graph': graph,
        'fault_tree': fault_tree,
        'lambda_': lambda_dict,
        'mu_': mu_dict,
        'structure': structure
    }