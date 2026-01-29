import numpy as np
from collections import defaultdict


def compute_component_criticality(graph):
    # Costruisci struttura di adiacenza (child → parent)
    child_to_parents = defaultdict(list)
    for src, dst in graph.edges:
        child_to_parents[src].append(dst)

    # Trova la root (nodo senza genitori uscenti, cioè l'ultimo nodo)
    root_idx = len(graph.nodes) - 1

    # Per ogni componente, calcola metriche di criticità
    criticality = {}

    for node in graph.nodes:
        if node['type'] != 'component':
            continue

        comp_name = node['name']
        comp_idx = node['idx']

        # Conta AND e OR nel path verso la root
        n_and_above, n_or_above, depth = _count_gates_to_root(
            comp_idx, child_to_parents, graph.nodes, root_idx
        )

        # Formula di criticità:
        # - Più AND sopra = più critico (tutti devono fallire)
        # - Più OR sopra = meno critico (basta uno)
        # - Depth conta meno

        if n_and_above + n_or_above > 0:
            # Ratio AND / totale gates
            and_ratio = n_and_above / (n_and_above + n_or_above)
        else:
            and_ratio = 0.5  # Default se nessun gate (caso degenere)

        # Criticità base: quanto il componente è "necessario"
        # Se tutti AND sopra → criticità = 1 (deve fallire)
        # Se tutti OR sopra → criticità bassa (uno dei tanti)
        base_criticality = and_ratio

        # Bonus per depth (componenti più profondi leggermente più critici)
        depth_bonus = min(0.2, depth * 0.05)

        criticality[comp_name] = min(1.0, base_criticality + depth_bonus)

    # Normalizza tra 0 e 1
    if criticality:
        min_crit = min(criticality.values())
        max_crit = max(criticality.values())

        if max_crit > min_crit:
            criticality = {
                c: (v - min_crit) / (max_crit - min_crit)
                for c, v in criticality.items()
            }
        else:
            # Tutti uguali, assegna 0.5
            criticality = {c: 0.5 for c in criticality}

    return criticality


def _count_gates_to_root(start_idx, child_to_parents, nodes, root_idx):
    n_and = 0
    n_or = 0
    depth = 0

    current = start_idx
    visited = set()

    while current != root_idx and current not in visited:
        visited.add(current)

        parents = child_to_parents.get(current, [])
        if not parents:
            break

        # Prendi il primo parent (in un albero ce n'è uno solo)
        parent_idx = parents[0]
        parent_node = nodes[parent_idx]

        if parent_node['type'] == 'AND':
            n_and += 1
        elif parent_node['type'] == 'OR':
            n_or += 1

        depth += 1
        current = parent_idx

    return n_and, n_or, depth


def get_alpha_multipliers(criticality, base_alpha_min, base_alpha_max):
    alpha_ranges = {}

    # Range di scaling: componenti critici ottengono fino a 1.5x il base
    # Componenti non critici ottengono fino a 0.7x il base
    min_scale = 0.7
    max_scale = 1.5

    for comp, crit in criticality.items():
        # Scala lineare: crit=0 → min_scale, crit=1 → max_scale
        scale = min_scale + crit * (max_scale - min_scale)

        # Applica scaling
        alpha_min = max(1.0, base_alpha_min * scale)
        alpha_max = max(alpha_min + 0.5, base_alpha_max * scale)

        alpha_ranges[comp] = (alpha_min, alpha_max)

    return alpha_ranges


def get_beta_multipliers(criticality, base_beta_min, base_beta_max):
    beta_ranges = {}

    # Scaling più conservativo per beta
    min_scale = 0.8
    max_scale = 1.3

    for comp, crit in criticality.items():
        scale = min_scale + crit * (max_scale - min_scale)

        beta_min = max(1.0, base_beta_min * scale)
        beta_max = max(beta_min + 0.3, base_beta_max * scale)

        beta_ranges[comp] = (beta_min, beta_max)

    return beta_ranges

# Test standalone
if __name__ == "__main__":
    from alfa_beta_range_predictor import generate_simple_fault_tree

    # Genera un fault tree di test
    ft_data = generate_simple_fault_tree((10, 15))
    graph = ft_data['graph']

    print(f"Struttura: {ft_data['structure']}")

    # Calcola criticità
    criticality = compute_component_criticality(graph)

    # Test scaling
    base_alpha = (5.0, 10.0)
    alpha_ranges = get_alpha_multipliers(criticality, base_alpha[0], base_alpha[1])

    print("\n" + "=" * 50)
    print("ALPHA RANGES PER COMPONENTE")
    print("=" * 50)
    for comp, (a_min, a_max) in sorted(alpha_ranges.items()):
        crit = criticality[comp]
        print(f"{comp}: α=[{a_min:.2f}, {a_max:.2f}] (crit={crit:.2f})")