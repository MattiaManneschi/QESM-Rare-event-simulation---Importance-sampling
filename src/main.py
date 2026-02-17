from src.fault_tree_generator import generate_rare_event_fault_tree, _estimate_tree_log_prob

if __name__ == "__main__":
    from direct_predictor import (
        DirectPredictor,
        train_direct_predictor_incremental
    )
    from n_samples_predictor import (
        SamplePredictor,
        train_sample_predictor_incremental
    )
    from cdf_analysis import run_cdf_analysis
    import torch
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODELS_DIR = '../models'
    DIRECT_MODEL_PATH = os.path.join(MODELS_DIR, 'direct_predictor.pth')
    SAMPLE_MODEL_PATH = os.path.join(MODELS_DIR, 'sample_predictor.pth')

    if os.path.exists(DIRECT_MODEL_PATH):
        direct_model = DirectPredictor().to(device)
        direct_model.load_state_dict(torch.load(DIRECT_MODEL_PATH, map_location=device))
        print(f"[DirectPredictor] Caricato da {DIRECT_MODEL_PATH}")
    else:
        print("[DirectPredictor] Training incrementale...")
        direct_model = train_direct_predictor_incremental(
            stages=[(15, 30), (30, 45)],
            n_iterations_per_stage=2000,
            T_range=(10, 500)
        )
        torch.save(direct_model.state_dict(), DIRECT_MODEL_PATH)
        print(f"[DirectPredictor] Salvato in {DIRECT_MODEL_PATH}")

    if os.path.exists(SAMPLE_MODEL_PATH):
        sample_model = SamplePredictor().to(device)
        sample_model.load_state_dict(torch.load(SAMPLE_MODEL_PATH, map_location=device))
        print(f"[SamplePredictor] Caricato da {SAMPLE_MODEL_PATH}")
    else:
        print("[SamplePredictor] Training incrementale...")
        sample_model = train_sample_predictor_incremental(
            stages=[(15, 30), (30, 45)],
            n_iterations_per_stage=2000,
            T_range=(10, 500)
        )
        torch.save(sample_model.state_dict(), SAMPLE_MODEL_PATH)
        print(f"[SamplePredictor] Salvato in {SAMPLE_MODEL_PATH}")

    iterations = 1

    target_order = -7

    for iteration in range(iterations):
        ft_data = generate_rare_event_fault_tree((30, 45), target_p_order=-target_order)
        log_p = _estimate_tree_log_prob(ft_data['graph'])
        print(f"DEBUG: target={target_order}, actual log_p={log_p:.1f}")
        print(f"TOPOLOGIA: {ft_data['structure']}")
        results = run_cdf_analysis(
            ft_data['graph'],
            ft_data['fault_tree'],
            direct_model,
            topology_name=ft_data['structure'],
            t_max=1000,
            t_step=1,
            sample_model=sample_model
        )

