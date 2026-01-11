import os
import torch
from range_predictor import RangePredictor, train_range_predictor, FaultTreeGraph
from sample_predictor import SamplePredictor, train_sample_predictor, get_predicted_samples
from range_tester import run_range_tester

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Percorsi file modelli
MODELS_DIR = 'models'
RANGE_MODEL_PATH = os.path.join(MODELS_DIR, 'range_predictor.pth')
SAMPLE_MODEL_PATH = os.path.join(MODELS_DIR, 'sample_predictor.pth')

# Modelli globali (addestrati una volta)
range_model = None
sample_model = None


def load_or_train_range_model(n_iterations=200, force_train=False):
    """
    Carica il RangePredictor se esiste, altrimenti lo addestra e salva.

    Args:
        n_iterations: numero iterazioni training
        force_train: se True, riaddestra anche se esiste

    Returns:
        RangePredictor addestrato
    """
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
    """
    Carica il SamplePredictor se esiste, altrimenti lo addestra e salva.

    Args:
        n_iterations: numero iterazioni training
        force_train: se True, riaddestra anche se esiste

    Returns:
        SamplePredictor addestrato
    """
    global sample_model

    sample_model = SamplePredictor().to(device)

    if os.path.exists(SAMPLE_MODEL_PATH) and not force_train:
        sample_model.load_state_dict(torch.load(SAMPLE_MODEL_PATH, map_location=device))
        print(f"[SamplePredictor] Caricato da {SAMPLE_MODEL_PATH}")
    else:
        print(f"[SamplePredictor] Training ({n_iterations} iterazioni)...")
        sample_model = train_sample_predictor(n_iterations=n_iterations)
        # train_sample_predictor ritorna (model, history), prendiamo solo il model
        if isinstance(sample_model, tuple):
            sample_model = sample_model[0]
        torch.save(sample_model.state_dict(), SAMPLE_MODEL_PATH)
        print(f"[SamplePredictor] Salvato in {SAMPLE_MODEL_PATH}")

    return sample_model


def initialize_models(n_iter_range=200, n_iter_sample=200, force_train=False):
    """
    Inizializza entrambi i modelli (carica o addestra).

    Args:
        n_iter_range: iterazioni per RangePredictor
        n_iter_sample: iterazioni per SamplePredictor
        force_train: se True, riaddestra entrambi
    """
    load_or_train_range_model(n_iter_range, force_train)
    load_or_train_sample_model(n_iter_sample, force_train)


def get_ranges(ft):
    """
    Usa RangePredictor per ottenere i range ottimali di α e β.

    Args:
        ft: FaultTreeGraph

    Returns:
        (alpha_min, alpha_max, beta_min, beta_max)
    """
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
    """
    Usa SamplePredictor per ottenere N_is e N_mc ottimali.

    Args:
        ft: FaultTreeGraph

    Returns:
        (N_is, N_mc)
    """
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


def run_pipeline(ft, tree_structure, T=100):
    """
    Pipeline completa:
    1. Predici range α, β
    2. Predici N_is, N_mc
    3. Esegui IS e confronta con MC (stampa risultati)

    Args:
        ft: FaultTreeGraph
        T: orizzonte temporale
    """

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
    run_range_tester(ft, fault_tree_logic, ranges_dict, tree_structure, T, N_is, N_mc)


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
    run_pipeline(ft, "AND-2", T=100)

    # Esempio: OR di 3 componenti
    print("\n" + "=" * 60)
    print("ESEMPIO: OR di 3 componenti")
    print("=" * 60)

    ft2 = FaultTreeGraph()
    nodes2 = [ft2.add_component(f"C{i}", 1e-2, 0.1) for i in range(3)]
    ft2.add_gate('OR', nodes2)
    run_pipeline(ft, "OR-3", T=100)

    # Esempio: (A ∧ B) ∨ C
    print("\n" + "=" * 60)
    print("ESEMPIO: (A ∧ B) ∨ C")
    print("=" * 60)

    ft3 = FaultTreeGraph()
    idx_A = ft3.add_component('A', 5e-3, 0.1)
    idx_B = ft3.add_component('B', 5e-3, 0.1)
    idx_C = ft3.add_component('C', 1e-2, 0.1)
    idx_AND = ft3.add_gate('AND', [idx_A, idx_B])
    ft3.add_gate('OR', [idx_AND, idx_C])
    run_pipeline(ft, "GENERICO", T=100)

