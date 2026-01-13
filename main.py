import os
import torch
from range_predictor import RangePredictor, train_range_predictor, FaultTreeGraph
from sample_predictor import SamplePredictor, train_sample_predictor, get_predicted_samples
from range_tester import run_overall_tester

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Percorsi file modelli
MODELS_DIR = 'models'
RANGE_MODEL_PATH = os.path.join(MODELS_DIR, 'range_predictor.pth')
SAMPLE_MODEL_PATH = os.path.join(MODELS_DIR, 'sample_predictor.pth')

# Modelli globali (addestrati una volta)
range_model = None
sample_model = None


def load_or_train_range_model(n_iterations=200, force_train=False):
    global range_model

    # Crea cartella models/ se non esiste
    os.makedirs(MODELS_DIR, exist_ok=True)

    range_model = RangePredictor().to(device)

    if os.path.exists(RANGE_MODEL_PATH) and not force_train:
        range_model.load_state_dict(torch.load(RANGE_MODEL_PATH, map_location=device))
        print(f"[RangePredictor] Caricato da {RANGE_MODEL_PATH}")
    else:
        print(f"[RangePredictor] Training ({n_iterations} iterazioni)...")
        range_model = train_range_predictor(n_iterations=n_iterations)
        torch.save(range_model.state_dict(), RANGE_MODEL_PATH)
        print(f"[RangePredictor] Salvato in {RANGE_MODEL_PATH}")

    return range_model

def load_or_train_sample_model(n_iterations=200, force_train=False):
    global sample_model, range_model

    # Assicurati che range_model sia caricato
    if range_model is None:
        load_or_train_range_model()

    sample_model = SamplePredictor().to(device)

    if os.path.exists(SAMPLE_MODEL_PATH) and not force_train:
        sample_model.load_state_dict(torch.load(SAMPLE_MODEL_PATH, map_location=device))
        print(f"[SamplePredictor] Caricato da {SAMPLE_MODEL_PATH}")
    else:
        print(f"[SamplePredictor] Training ({n_iterations} iterazioni)...")
        # Passa range_model al training
        sample_model = train_sample_predictor(range_model, n_iterations=n_iterations)
        if isinstance(sample_model, tuple):
            sample_model = sample_model[0]
        torch.save(sample_model.state_dict(), SAMPLE_MODEL_PATH)
        print(f"[SamplePredictor] Salvato in {SAMPLE_MODEL_PATH}")

    return sample_model

def initialize_models(n_iter_range=200, n_iter_sample=100, force_train=False):
    load_or_train_range_model(n_iter_range, force_train)
    load_or_train_sample_model(n_iter_sample, force_train)

def get_ranges(ft):
    global range_model

    if range_model is None:
        load_or_train_range_model()

    range_model.eval()
    data = ft.to_pyg_data().to(device)

    with torch.no_grad():
        ranges, _ = range_model(data)
        r = ranges[0].cpu().numpy()

    alpha_min, alpha_max = r[0], r[1]
    beta_min, beta_max = r[2], r[3]

    print(f"\n-> Range suggeriti:")
    print(f"   Alpha: [{alpha_min:.2f}, {alpha_max:.2f}]")
    print(f"   Beta:  [{beta_min:.2f}, {beta_max:.2f}]")

    return alpha_min, alpha_max, beta_min, beta_max

def get_samples(ft):
    global sample_model

    if sample_model is None:
        load_or_train_sample_model()

    sample_model.eval()
    pyg_data = ft.to_pyg_data().to(device)

    if not hasattr(pyg_data, 'batch'):
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=device)

    N_is, N_mc = get_predicted_samples(sample_model, pyg_data)

    print(f"\n-> Samples suggeriti:")
    print(f"   N_is: {N_is}")
    print(f"   N_mc: {N_mc}")

    return N_is, N_mc

def run_pipeline(ft, topology_name):
    print("\n" + "=" * 60)
    print("PIPELINE IMPORTANCE SAMPLING")
    print("=" * 60)

    # 1. Predici range
    print("\n[1/3] Predizione range α, β...")
    alpha_min, alpha_max, beta_min, beta_max = get_ranges(ft)

    ranges_dict = {
        'alpha': (alpha_min, alpha_max),
        'beta': (beta_min, beta_max),
    }

    # 2. Predici samples
    print("\n[2/3] Predizione numero samples...")
    N_is, N_mc = get_samples(ft)

    # 3. Esegui test (stampa direttamente i risultati)
    print("\n[3/3] Esecuzione IS vs MC...")
    fault_tree_logic = ft.get_logic_function()
    run_overall_tester(ft, fault_tree_logic, ranges_dict, N_is, N_mc, topology_name)

if __name__ == "__main__":

    # Inizializza modelli (carica se esistono, altrimenti addestra e salva)
    initialize_models(n_iter_range=200, n_iter_sample=200, force_train=False)

    # Esempio: AND di 2 componenti
    print("\n" + "=" * 60)
    print("ESEMPIO: AND di 2 componenti")
    print("=" * 60)

    ft = FaultTreeGraph()
    nodes = [ft.add_component(f"C{i}", 3e-3, 0.1) for i in range(2)]
    ft.add_gate('AND', nodes)
    run_pipeline(ft,"AND_2")

    # Esempio: OR di 3 componenti
    print("\n" + "=" * 60)
    print("ESEMPIO: OR di 3 componenti")
    print("=" * 60)

    ft = FaultTreeGraph()
    nodes2 = [ft.add_component(f"C{i}", 1e-4, 0.1) for i in range(3)]
    ft.add_gate('OR', nodes2)
    run_pipeline(ft, "OR_3")

    # Esempio: (A ∧ B) ∨ C
    print("\n" + "=" * 60)
    print("ESEMPIO: (A ∧ B) ∨ C")
    print("=" * 60)

    ft = FaultTreeGraph()
    idx_A = ft.add_component('A', 3e-3, 0.1)
    idx_B = ft.add_component('B', 3e-3, 0.1)
    idx_C = ft.add_component('C', 1e-4, 0.1)
    idx_AND = ft.add_gate('AND', [idx_A, idx_B])
    ft.add_gate('OR', [idx_AND, idx_C])
    run_pipeline(ft, "(A ∧ B) ∨ C")

    # 4. AND_3: tre componenti tutti guasti
    print("\n" + "=" * 60)
    print("ESEMPIO: AND_3")
    print("=" * 60)
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f"C{i}", 5e-3, 0.1) for i in range(3)]
    ft.add_gate('AND', nodes)
    run_pipeline(ft, "AND_3")

    # 5. OR_2: due componenti, basta uno guasto
    print("\n" + "=" * 60)
    print("ESEMPIO: OR_2")
    print("=" * 60)
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f"C{i}", 5e-5, 0.1) for i in range(2)]
    ft.add_gate('OR', nodes)
    run_pipeline(ft, "OR_2")

    # 6. 2oo3: almeno 2 su 3 guasti
    print("\n" + "=" * 60)
    print("ESEMPIO: 2oo3 (voting)")
    print("=" * 60)
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f"C{i}", 2e-3, 0.1) for i in range(3)]
    idx_AB = ft.add_gate('AND', [nodes[0], nodes[1]])
    idx_AC = ft.add_gate('AND', [nodes[0], nodes[2]])
    idx_BC = ft.add_gate('AND', [nodes[1], nodes[2]])
    ft.add_gate('OR', [idx_AB, idx_AC, idx_BC])
    run_pipeline(ft, "2oo3")

    # 7. (A ∨ B) ∧ C: OR sotto AND
    print("\n" + "=" * 60)
    print("ESEMPIO: (A ∨ B) ∧ C")
    print("=" * 60)
    ft = FaultTreeGraph()
    idx_A = ft.add_component('A', 5e-3, 0.1)  # 0.5
    idx_B = ft.add_component('B', 5e-3, 0.1)  # 0.5
    idx_C = ft.add_component('C', 2e-3, 0.1)  # 0.2
    idx_OR = ft.add_gate('OR', [idx_A, idx_B])
    ft.add_gate('AND', [idx_OR, idx_C])
    run_pipeline(ft, "(A ∨ B) ∧ C")

    # 8. (A ∧ B) ∨ (C ∧ D): due AND in parallelo
    print("\n" + "=" * 60)
    print("ESEMPIO: (A ∧ B) ∨ (C ∧ D)")
    print("=" * 60)
    ft = FaultTreeGraph()
    idx_A = ft.add_component('A', 3e-3, 0.1)
    idx_B = ft.add_component('B', 3e-3, 0.1)
    idx_C = ft.add_component('C', 3e-3, 0.1)
    idx_D = ft.add_component('D', 3e-3, 0.1)
    idx_AND1 = ft.add_gate('AND', [idx_A, idx_B])
    idx_AND2 = ft.add_gate('AND', [idx_C, idx_D])
    ft.add_gate('OR', [idx_AND1, idx_AND2])
    run_pipeline(ft, "(A ∧ B) ∨ (C ∧ D)")

    # 9. ((A ∧ B) ∧ C): AND profondo
    print("\n" + "=" * 60)
    print("ESEMPIO: AND_3 profondo ((A ∧ B) ∧ C)")
    print("=" * 60)
    ft = FaultTreeGraph()
    idx_A = ft.add_component('A', 8e-3, 0.1)
    idx_B = ft.add_component('B', 8e-3, 0.1)
    idx_C = ft.add_component('C', 8e-3, 0.1)
    idx_AND1 = ft.add_gate('AND', [idx_A, idx_B])
    ft.add_gate('AND', [idx_AND1, idx_C])
    run_pipeline(ft, "(A ∧ B) ∧ C")


    # 10. (A ∨ B) ∧ (C ∨ D): due OR in serie
    print("\n" + "=" * 60)
    print("ESEMPIO: (A ∨ B) ∧ (C ∨ D)")
    print("=" * 60)
    ft = FaultTreeGraph()
    idx_A = ft.add_component('A', 1e-3, 0.1)
    idx_B = ft.add_component('B', 1e-3, 0.1)
    idx_C = ft.add_component('C', 1e-3, 0.1)
    idx_D = ft.add_component('D', 1e-3, 0.1)
    idx_OR1 = ft.add_gate('OR', [idx_A, idx_B])
    idx_OR2 = ft.add_gate('OR', [idx_C, idx_D])
    ft.add_gate('AND', [idx_OR1, idx_OR2])
    run_pipeline(ft, "(A ∨ B) ∧ (C ∨ D)")