import os
import torch
from alfa_beta_range_predictor import RangePredictor, train_range_predictor, FaultTreeGraph, generate_simple_fault_tree
from N_samples_predictor import SamplePredictor, train_sample_predictor, get_predicted_samples
from cdf_analysis import run_cdf_analysis
from is_optimizer_evaluator import run_overall_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Percorsi file modelli
MODELS_DIR = 'models'
RANGE_MODEL_PATH = os.path.join(MODELS_DIR, 'range_predictor.pth')
SAMPLE_MODEL_PATH = os.path.join(MODELS_DIR, 'sample_predictor.pth')

# Modelli globali (addestrati una volta)
range_model = None
sample_model = None


def load_or_train_range_model(n_iterations=300, T_range=(10, 500), force_train=False):
    """
    Carica o addestra il RangePredictor.

    NOVITÀ: Il modello ora è addestrato con T variabile!
    """
    global range_model

    os.makedirs(MODELS_DIR, exist_ok=True)

    range_model = RangePredictor().to(device)

    if os.path.exists(RANGE_MODEL_PATH) and not force_train:
        range_model.load_state_dict(torch.load(RANGE_MODEL_PATH, map_location=device))
        print(f"[RangePredictor] Caricato da {RANGE_MODEL_PATH}")
    else:
        print(f"[RangePredictor] Training ({n_iterations} iterazioni, T in {T_range})...")
        range_model = train_range_predictor(n_iterations=n_iterations, T_range=T_range)
        torch.save(range_model.state_dict(), RANGE_MODEL_PATH)
        print(f"[RangePredictor] Salvato in {RANGE_MODEL_PATH}")

    return range_model


def load_or_train_sample_model(n_iterations=200, force_train=False):
    global sample_model, range_model

    if range_model is None:
        load_or_train_range_model()

    sample_model = SamplePredictor().to(device)

    if os.path.exists(SAMPLE_MODEL_PATH) and not force_train:
        sample_model.load_state_dict(torch.load(SAMPLE_MODEL_PATH, map_location=device))
        print(f"[SamplePredictor] Caricato da {SAMPLE_MODEL_PATH}")
    else:
        print(f"[SamplePredictor] Training ({n_iterations} iterazioni)...")
        sample_model = train_sample_predictor(range_model, n_iterations=n_iterations)
        if isinstance(sample_model, tuple):
            sample_model = sample_model[0]
        torch.save(sample_model.state_dict(), SAMPLE_MODEL_PATH)
        print(f"[SamplePredictor] Salvato in {SAMPLE_MODEL_PATH}")

    return sample_model


def initialize_models(n_iter_range=300, n_iter_sample=200, T_range=(10, 500), force_train=False):
    """
    Inizializza entrambi i modelli.

    Args:
        n_iter_range: iterazioni per RangePredictor
        n_iter_sample: iterazioni per SamplePredictor
        T_range: (T_min, T_max) per training RangePredictor
        force_train: se True, riaddestra anche se esistono file salvati
    """
    load_or_train_range_model(n_iter_range, T_range, force_train)
    load_or_train_sample_model(n_iter_sample, force_train)


def get_ranges(ft, T=100, T_max=500):
    """
    Ottiene i range α/β per un fault tree a un dato T.

    NOVITÀ: Ora richiede T come parametro!
    """
    global range_model

    if range_model is None:
        load_or_train_range_model()

    range_model.eval()
    data = ft.to_pyg_data().to(device)

    with torch.no_grad():
        ranges, _ = range_model(data, T=T, T_max=T_max)
        r = ranges[0].cpu().numpy()

    alpha_min, alpha_max = r[0], r[1]
    beta_min, beta_max = r[2], r[3]

    print(f"\n-> Range suggeriti (T={T}):")
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


def run_pipeline(ft, topology_name, T=100, T_max=500):
    """
    Esegue la pipeline completa per un singolo T.

    NOVITÀ: T è ora un parametro!
    """
    print("\n" + "=" * 60)
    print(f"PIPELINE IMPORTANCE SAMPLING (T={T})")
    print("=" * 60)

    # 1. Predici range (ora con T!)
    print(f"\n[1/3] Predizione range α, β per T={T}...")
    alpha_min, alpha_max, beta_min, beta_max = get_ranges(ft, T=T, T_max=T_max)

    ranges_dict = {
        'alpha': (alpha_min, alpha_max),
        'beta': (beta_min, beta_max),
    }

    # 2. Predici samples
    print("\n[2/3] Predizione numero samples...")
    N_is, N_mc = get_samples(ft)

    # 3. Esegui test
    print("\n[3/3] Esecuzione IS vs MC...")
    fault_tree_logic = ft.get_logic_function()
    run_overall_test(ft, fault_tree_logic, ranges_dict, N_is, N_mc, topology_name, T)


def run_cdf_pipeline(ft_data, t_max=500, t_step=10):
    """
    Esegue l'analisi CDF completa.

    Calcola P(T_fail ≤ t) per t da t_step a t_max.
    Si ferma quando P > 10%.
    """
    global range_model, sample_model

    if range_model is None:
        load_or_train_range_model()
    if sample_model is None:
        load_or_train_sample_model()

    results = run_cdf_analysis(
        ft_data['graph'],
        ft_data['fault_tree'],
        range_model,
        topology_name=ft_data['structure'],
        t_max=t_max,
        t_step=t_step,
        sample_model=sample_model
    )

    return results


if __name__ == "__main__":
    initialize_models(
        n_iter_range=1500,
        n_iter_sample=1000,
        T_range=(10, 500),
        force_train=False
    )

    iterations = 5

    for iteration in range(iterations):
        ft_data = generate_simple_fault_tree()
        results = run_cdf_pipeline(ft_data, t_max=500, t_step=5)
