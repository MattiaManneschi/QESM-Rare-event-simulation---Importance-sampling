import os
import torch

from alfa_beta_range_predictor import RangePredictor, train_range_predictor, FaultTreeGraph, generate_simple_fault_tree
from N_samples_predictor import SamplePredictor, train_sample_predictor, get_predicted_samples
from cdf_analysis import run_cdf_analysis
from is_optimizer_evaluator import run_overall_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Percorsi file modelli
MODELS_DIR = '../models'
SAMPLE_MODEL_PATH = os.path.join(MODELS_DIR, 'sample_predictor_2_15.pth')

# Modelli globali (addestrati una volta)
range_model = None
sample_model = None


def load_or_train_range_model(n_iterations, T_range, comp_range, force_train=False):
    """
    Carica o addestra il RangePredictor.

    Il modello viene salvato con nome che riflette il comp_range.
    Per fine-tuning, carica automaticamente il modello del range precedente.
    """
    global range_model

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Nome file basato su comp_range
    min_comp = comp_range[0]
    max_comp = comp_range[1]
    model_path = os.path.join(MODELS_DIR, f'range_predictor_{min_comp}_{max_comp}.pth')

    range_model = RangePredictor().to(device)

    if os.path.exists(model_path) and not force_train:
        range_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[RangePredictor] Caricato da {model_path}")
    else:
        # Cerca modello precedente per fine-tuning
        pretrained_model = None

        if max_comp > 15:
            prev_ranges = [(2, 15), (15, 30), (30, 45)]
            for prev_min, prev_max in prev_ranges:
                if prev_max < max_comp:
                    candidate = os.path.join(MODELS_DIR, f'range_predictor_{prev_min}_{prev_max}.pth')
                    if os.path.exists(candidate):
                        pretrained_model = RangePredictor().to(device)
                        pretrained_model.load_state_dict(torch.load(candidate, map_location=device))
                        print(f"[RangePredictor] Fine-tuning da {candidate}")

        print(f"[RangePredictor] Training ({n_iterations} iter, T={T_range}, comp={comp_range})...")
        range_model = train_range_predictor(
            n_iterations=n_iterations,
            T_range=T_range,
            comp_range=comp_range,
            pretrained_model=pretrained_model
        )
        torch.save(range_model.state_dict(), model_path)
        print(f"[RangePredictor] Salvato in {model_path}")

    return range_model


def load_or_train_sample_model(n_iterations, comp_range, T_range=(10, 500),
                               target_cv=0.3, force_train=False):
    """
    Carica o addestra il SamplePredictor.

    NOVITÀ:
    - T_range invece di T fisso
    - target_cv invece di target_top_events
    """
    global sample_model

    os.makedirs(MODELS_DIR, exist_ok=True)

    min_comp, max_comp = comp_range
    model_path = os.path.join(MODELS_DIR, f'sample_predictor_{min_comp}_{max_comp}.pth')

    sample_model = SamplePredictor().to(device)

    if os.path.exists(model_path) and not force_train:
        sample_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[SamplePredictor] Caricato da {model_path}")
    else:
        pretrained_model = None

        if max_comp > 15:
            prev_ranges = [(2, 15), (15, 30), (30, 45)]
            for prev_min, prev_max in prev_ranges:
                if prev_max < max_comp:
                    candidate = os.path.join(MODELS_DIR, f'sample_predictor_{prev_min}_{prev_max}.pth')
                    if os.path.exists(candidate):
                        pretrained_model = SamplePredictor().to(device)
                        pretrained_model.load_state_dict(torch.load(candidate, map_location=device))
                        print(f"[SamplePredictor] Fine-tuning da {candidate}")

        print(f"[SamplePredictor] Training ({n_iterations} iter, T={T_range}, comp={comp_range})...")
        sample_model = train_sample_predictor(
            range_model=range_model,
            n_iterations=n_iterations,
            T_range=T_range,
            target_cv=target_cv,
            comp_range=comp_range,
            pretrained_model=pretrained_model
        )
        torch.save(sample_model.state_dict(), model_path)
        print(f"[SamplePredictor] Salvato in {model_path}")

    return sample_model

def initialize_models(n_iter_range, n_iter_sample, T_range, comp_range, force_train=False):
    """
    Inizializza entrambi i modelli.

    Args:
        n_iter_range: iterazioni per RangePredictor
        n_iter_sample: iterazioni per SamplePredictor
        T_range: (T_min, T_max) per training RangePredictor
        force_train: se True, riaddestra anche se esistono file salvati
    """
    load_or_train_range_model(n_iter_range, T_range, comp_range, force_train)
    load_or_train_sample_model(n_iter_sample, comp_range, T_range, force_train)

def get_ranges(ft, T, T_max):
    """
    Ottiene i range α/β per un fault tree a un dato T.

    NOVITÀ: Ora richiede T come parametro!
    """

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


if __name__ == "__main__":

    n_1 = 30
    n_2 = 45

    comp_range = (n_1, n_2)

    initialize_models(
        n_iter_range=(n_2-n_1) * (10 ** 2),
        n_iter_sample=(n_2-n_1) * (10 ** 2),
        T_range=(10, 500),
        comp_range=comp_range,
        force_train=False
    )

    """

    iterations = 1

    for iteration in range(iterations):
        ft_data = generate_simple_fault_tree((30, 45))
        print("TOPOLOGIA ALBERO: ", ft_data['structure'])
        results = run_cdf_analysis(
            ft_data['graph'],
            ft_data['fault_tree'],
            range_model,
            topology_name=ft_data['structure'],
            t_max=500,
            t_step=10,
            sample_model=sample_model,
            use_component_criticality=True
        )
    """

