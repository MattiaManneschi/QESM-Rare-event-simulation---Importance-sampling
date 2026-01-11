import torch
from range_predictor import RangePredictor, FaultTreeGraph
from range_tester import run_overall_tester
from sample_predictor import SamplePredictor, get_predicted_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_test_trees():
    """Crea fault tree di test con strutture diverse."""

    trees = {}

    # Parametri fissi per confronto equo
    lam = 1e-2
    mu = 0.1

    # =========================================================================
    # 1. AND con numero crescente di componenti
    # =========================================================================

    # AND_2
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(2)]
    ft.add_gate('AND', nodes)
    trees['AND_2'] = ft

    # AND_3
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(3)]
    ft.add_gate('AND', nodes)
    trees['AND_3'] = ft

    # AND_4
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(4)]
    ft.add_gate('AND', nodes)
    trees['AND_4'] = ft

    # AND_5
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(5)]
    ft.add_gate('AND', nodes)
    trees['AND_5'] = ft

    # =========================================================================
    # 2. OR con numero crescente di componenti
    # =========================================================================

    # OR_2
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(2)]
    ft.add_gate('OR', nodes)
    trees['OR_2'] = ft

    # OR_3
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(3)]
    ft.add_gate('OR', nodes)
    trees['OR_3'] = ft

    # OR_5
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(5)]
    ft.add_gate('OR', nodes)
    trees['OR_5'] = ft

    # =========================================================================
    # 3. Gerarchici AND vs OR
    # =========================================================================

    # (A ∧ B) ∧ (C ∧ D) - 4 componenti tutti in AND
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(4)]
    and1 = ft.add_gate('AND', [nodes[0], nodes[1]])
    and2 = ft.add_gate('AND', [nodes[2], nodes[3]])
    ft.add_gate('AND', [and1, and2])
    trees['hier_AND_AND'] = ft

    # (A ∨ B) ∨ (C ∨ D) - 4 componenti tutti in OR
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(4)]
    or1 = ft.add_gate('OR', [nodes[0], nodes[1]])
    or2 = ft.add_gate('OR', [nodes[2], nodes[3]])
    ft.add_gate('OR', [or1, or2])
    trees['hier_OR_OR'] = ft

    # (A ∧ B) ∨ (C ∧ D) - AND sotto OR
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(4)]
    and1 = ft.add_gate('AND', [nodes[0], nodes[1]])
    and2 = ft.add_gate('AND', [nodes[2], nodes[3]])
    ft.add_gate('OR', [and1, and2])
    trees['hier_AND_OR'] = ft

    # (A ∨ B) ∧ (C ∨ D) - OR sotto AND
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(4)]
    or1 = ft.add_gate('OR', [nodes[0], nodes[1]])
    or2 = ft.add_gate('OR', [nodes[2], nodes[3]])
    ft.add_gate('AND', [or1, or2])
    trees['hier_OR_AND'] = ft

    # =========================================================================
    # 4. Voting
    # =========================================================================

    # 2oo3
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(3)]
    and_01 = ft.add_gate('AND', [nodes[0], nodes[1]])
    and_02 = ft.add_gate('AND', [nodes[0], nodes[2]])
    and_12 = ft.add_gate('AND', [nodes[1], nodes[2]])
    ft.add_gate('OR', [and_01, and_02, and_12])
    trees['2oo3'] = ft

    # =========================================================================
    # 5. Mixed
    # =========================================================================

    # (A ∧ B ∧ C) ∨ D - AND profondo con escape OR
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(4)]
    and_gate = ft.add_gate('AND', [nodes[0], nodes[1], nodes[2]])
    ft.add_gate('OR', [and_gate, nodes[3]])
    trees['mixed_deep_AND'] = ft

    # (A ∨ B ∨ C) ∧ D - OR profondo con gate AND
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f'C{i}', lam, mu) for i in range(4)]
    or_gate = ft.add_gate('OR', [nodes[0], nodes[1], nodes[2]])
    ft.add_gate('AND', [or_gate, nodes[3]])
    trees['mixed_deep_OR'] = ft

    return trees

def load_models(train_if_missing=True):
    """Carica i modelli addestrati, oppure li addestra se mancano."""
    import os
    from range_predictor import train_range_predictor
    from sample_predictor import train_sample_predictor

    os.makedirs('models', exist_ok=True)

    range_model = RangePredictor().to(device)
    sample_model = SamplePredictor().to(device)

    # Range predictor
    if os.path.exists('models/range_predictor.pth'):
        range_model.load_state_dict(torch.load('models/range_predictor.pth', map_location=device))
        print("Range predictor caricato.")
    elif train_if_missing:
        print("Range predictor non trovato. Addestramento...")
        range_model = train_range_predictor(n_iterations=200)
        torch.save(range_model.state_dict(), 'models/range_predictor.pth')
        print("Range predictor addestrato e salvato.")
    else:
        print("ERRORE: Range predictor non trovato.")
        return None, None

    # Sample predictor
    if os.path.exists('models/sample_predictor.pth'):
        sample_model.load_state_dict(torch.load('models/sample_predictor.pth', map_location=device))
        print("Sample predictor caricato.")
    elif train_if_missing:
        print("Sample predictor non trovato. Addestramento...")
        sample_model = train_sample_predictor(n_iterations=200)
        torch.save(sample_model.state_dict(), 'models/sample_predictor.pth')
        print("Sample predictor addestrato e salvato.")
    else:
        print("ERRORE: Sample predictor non trovato.")
        return None, None

    return range_model, sample_model

def test_models():
    """Esegue test su tutte le strutture e salva i risultati completi su file."""
    import os
    from datetime import datetime

    # 1. Inizializzazione buffer per il file e funzione di logging
    lines = []
    def log(msg):
        print(msg)
        lines.append(str(msg))

    log("=" * 80)
    log("TEST MODELLI SU STRUTTURE DIVERSE")
    log("=" * 80)

    # 2. Caricamento modelli
    range_model, sample_model = load_models()
    if range_model is None:
        log("Errore: Modelli non caricati correttamente. Interruzione.")
        return

    range_model.eval()
    sample_model.eval()

    # 3. Creazione alberi
    trees = create_test_trees()
    results = {}

    # Header Tabella
    log(f"\n{'Struttura':<20} | {'α_min':>6} {'α_max':>6} | {'β_min':>6} {'β_max':>6} | {'N_is':>7} {'N_mc':>7} | {'Ratio':>6} | {'P_is':>12} {'Var_is':>12} | {'P_mc':>12} {'Var_mc':>12}")
    log("-" * 80)

    # 4. Ciclo di predizione
    for name, ft in trees.items():
        pyg_data = ft.to_pyg_data().to(device)

        with torch.no_grad():
            # Predizione Range
            ranges, _ = range_model(pyg_data)
            r = ranges[0].cpu().numpy()
            ranges = {
                'alpha': (r[0], r[1]),
                'beta': (r[2], r[3]),
            }
            a_min, a_max = ranges['alpha']
            b_min, b_max = ranges['beta']

            # Predizione Samples
            if not hasattr(pyg_data, 'batch'):
                pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=device)
            n_is, n_mc = get_predicted_samples(sample_model, pyg_data)
            fault_tree_logic = ft.get_logic_function()
        print("\n")
        print("ALBERO -> ",name)
        p_is, var_is, p_mc, var_mc = run_overall_tester(ft, fault_tree_logic, ranges)

        ratio = n_mc / n_is if n_is > 0 else 0
        results[name] = {
            'alpha': (a_min, a_max),
            'beta': (b_min, b_max),
            'n_is': n_is,
            'n_mc': n_mc,
            'ratio': ratio,
            'p_is': p_is,
            'var_is': var_is,
            'p_mc': p_mc,
            'var_mc': var_mc
        }

        # Salvataggio riga riga nel log
        log(f"{name:<20} | {a_min:>6.2f} {a_max:>6.2f} | {b_min:>6.2f} {b_max:>6.2f} | {n_is:>7} {n_mc:>7} | {ratio:>6.2f} | {p_is:.6e} | {var_is:.6e} | {p_mc:.6e} | {var_mc:.6e}")

    # 5. Analisi Differenze (Sezioni specifiche)
    log("\n" + "=" * 80)
    log("ANALISI DIFFERENZE")
    log("=" * 80)

    sections = {
        "[AND crescente - α dovrebbe aumentare]": ['AND_2', 'AND_3', 'AND_4', 'AND_5'],
        "[OR crescente - α dovrebbe restare simile]": ['OR_2', 'OR_3', 'OR_5'],
        "[Gerarchici - profondità AND vs OR]": ['hier_AND_AND', 'hier_OR_OR', 'hier_AND_OR', 'hier_OR_AND']
    }

    for title, keys in sections.items():
        log(f"\n{title}")
        for k in keys:
            if k in results:
                log(f"  {k:<15}: α=[{results[k]['alpha'][0]:.2f}, {results[k]['alpha'][1]:.2f}]")

    # 6. Classifica Efficienza (Ratio)
    log("\n[N_mc / N_is ratio - ordinati per efficienza]")
    sorted_res = sorted(results.items(), key=lambda x: x[1]['ratio'], reverse=True)
    for name, data in sorted_res:
        log(f"  {name:<20}: ratio = {data['ratio']:.2f}")

    # 7. Salvataggio effettivo su disco
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/test_results_{timestamp}.txt'

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\n>>> Successo! Tutti i risultati sopra sono stati salvati in: '{filename}'")
    except Exception as e:
        print(f"\n>>> Errore durante il salvataggio del file: {e}")

if __name__ == "__main__":
    test_models()
