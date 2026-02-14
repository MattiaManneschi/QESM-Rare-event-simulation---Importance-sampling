"""
CDF Analysis v4 - Usa DirectPredictor + Adaptive IS (Cross-Entropy)

Stessa interfaccia di prima, ma internamente usa Cross-Entropy
per ottimizzare α, β ad ogni punto T.

Flusso per ogni T:
1. DirectPredictor → α₀, β₀ (punto di partenza)
2. Cross-Entropy (3-4 iter) → α_opt, β_opt (raffinamento)
3. IS con parametri ottimizzati → P_is
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import math
import torch

from direct_predictor import simulate_CTMC_simple
from n_samples_predictor import get_predicted_samples
from adaptive import adaptive_is_cross_entropy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_cdf_point(lambda_, mu_, alpha, beta, t, fault_tree_logic,
                      n_is=10000, n_mc=50000):
    """
    Calcola un punto della CDF con IS e MC.
    USA STANDARD IS (non self-normalized).
    """
    comps = list(lambda_.keys())

    # === IS ===
    results_is = [simulate_CTMC_simple(lambda_, mu_, alpha, beta, t, fault_tree_logic)
                  for _ in range(n_is)]

    all_log_w = [r['log_w'] for r in results_is]
    top_indicators = [1.0 if r['top'] else 0.0 for r in results_is]

    # Standard IS: p = (1/n) * Σ(w_i * ind_i)
    valid_log_w = [lw for lw in all_log_w if lw > -700]

    if valid_log_w:
        max_log_w = max(valid_log_w)

        numerator = 0.0
        for lw, ind in zip(all_log_w, top_indicators):
            if lw > -700:
                w = math.exp(lw - max_log_w)
                numerator += w * ind

        p_is = (numerator / n_is) * math.exp(max_log_w)
    else:
        p_is = 0.0

    n_top_is = sum(top_indicators)

    # Calcola std IS
    if n_top_is > 1:
        weights_top = []
        for lw, ind in zip(all_log_w, top_indicators):
            if ind > 0 and lw > -700:
                weights_top.append(math.exp(lw))
        if len(weights_top) > 1 and np.mean(weights_top) > 0:
            cv_is = np.std(weights_top) / np.mean(weights_top)
            std_is = p_is * cv_is / math.sqrt(len(weights_top))
        else:
            std_is = p_is
    else:
        std_is = p_is if p_is > 0 else 0.0

    # === MC ===
    alpha_mc = {c: 1.0 for c in comps}
    beta_mc = {c: 1.0 for c in comps}
    results_mc = [simulate_CTMC_simple(lambda_, mu_, alpha_mc, beta_mc, t, fault_tree_logic)
                  for _ in range(n_mc)]
    hits = [1.0 if r['top'] else 0.0 for r in results_mc]
    p_mc = np.mean(hits)
    std_mc = np.std(hits) / np.sqrt(n_mc)
    n_top_mc = sum(hits)

    return {
        'p_is': p_is, 'p_mc': p_mc,
        'std_is': std_is, 'std_mc': std_mc,
        'n_top_is': int(n_top_is), 'n_top_mc': int(n_top_mc)
    }


def compute_cdf_curve(ft, fault_tree_logic, direct_model, sample_model=None,
                      t_max=500, t_step=10,
                      ce_iterations=4, ce_samples=2000,
                      verbose=True):
    """
    Calcola la curva CDF usando DirectPredictor + Adaptive IS.
    """
    lambda_, mu_ = ft.get_lambda_mu()
    comps = list(lambda_.keys())
    n_comps = len(comps)

    t_values = np.arange(t_step, t_max + t_step, t_step)

    results = {
        't': [],
        'p_is': [], 'p_mc': [],
        'std_is': [], 'std_mc': [],
        'n_samples_is': [], 'n_samples_mc': [],
        'n_top_is': [], 'n_top_mc': [],
        'alphas': {c: [] for c in comps},
        'betas': {c: [] for c in comps},
    }

    n_and = sum(1 for n in ft.nodes if n.get('type') == 'AND')
    n_or = sum(1 for n in ft.nodes if n.get('type') == 'OR')

    if verbose:
        print("=" * 70)
        print("CALCOLO CURVA CDF (DirectPredictor + Adaptive IS)")
        print(f"T: [{t_step}, {t_max}], step={t_step}, punti={len(t_values)}")
        print(f"Componenti: {n_comps}, AND: {n_and}, OR: {n_or}")
        print(f"CE: {ce_iterations} iter, {ce_samples} samples/iter")
        print("=" * 70)

    pyg_data = ft.to_pyg_data().to(device)

    for t in t_values:
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Step 1: DirectPredictor → α₀, β₀
        direct_model.eval()
        alpha_init, beta_init = direct_model.predict(ft, T=t, T_max=float(t_max))

        # Step 2: Determina numero samples
        if sample_model is not None:
            sample_model.to(device)
            n_is, n_mc = get_predicted_samples(sample_model, pyg_data, T=t, T_max=float(t_max))
        else:
            # Fallback heuristic
            t_factor = max(1.0, 2.5 * (1.0 - t / t_max))
            and_factor = 1.0 + n_and * 0.1

            base_is = 30000
            base_mc = 80000

            n_is = int(base_is * t_factor * and_factor)
            n_mc = int(base_mc * t_factor * and_factor)

            n_is = min(200000, max(20000, n_is))
            n_mc = min(500000, max(50000, n_mc))

        # Step 3: Cross-Entropy per ottimizzare α, β
        alpha_opt, beta_opt, ce_stats = adaptive_is_cross_entropy(
            lambda_, mu_, t, fault_tree_logic,
            alpha_init, beta_init,
            n_iterations=ce_iterations,
            n_samples_per_iter=ce_samples,
            elite_fraction=0.15,
            smoothing=0.6,
            verbose=False
        )

        # Step 4: Calcola punto CDF con α, β ottimizzati
        cdf_point = compute_cdf_point(
            lambda_, mu_, alpha_opt, beta_opt, t,
            fault_tree_logic, n_is, n_mc
        )

        # Salva risultati
        results['t'].append(t)
        results['p_is'].append(cdf_point['p_is'])
        results['p_mc'].append(cdf_point['p_mc'])
        results['std_is'].append(cdf_point['std_is'])
        results['std_mc'].append(cdf_point['std_mc'])
        results['n_samples_is'].append(n_is)
        results['n_samples_mc'].append(n_mc)
        results['n_top_is'].append(cdf_point['n_top_is'])
        results['n_top_mc'].append(cdf_point['n_top_mc'])

        for c in comps:
            results['alphas'][c].append(alpha_opt[c])
            results['betas'][c].append(beta_opt[c])

        if verbose:
            avg_alpha_init = np.mean(list(alpha_init.values()))
            avg_alpha_opt = np.mean(list(alpha_opt.values()))
            avg_beta_opt = np.mean(list(beta_opt.values()))

            cv_is = cdf_point['std_is'] / cdf_point['p_is'] if cdf_point['p_is'] > 0 else float('inf')
            cv_mc = cdf_point['std_mc'] / cdf_point['p_mc'] if cdf_point['p_mc'] > 0 else float('inf')

            cv_is_str = f"{cv_is:.0%}" if cv_is != float('inf') else "inf"
            cv_mc_str = f"{cv_mc:.0%}" if cv_mc != float('inf') else "inf"

            print(f"[{timestamp}] T={t:3.0f} | "
                  f"P_is={cdf_point['p_is']:.2e} (CV={cv_is_str}, n={cdf_point['n_top_is']}/{n_is}) | "
                  f"P_mc={cdf_point['p_mc']:.2e} (CV={cv_mc_str}, n={cdf_point['n_top_mc']}/{n_mc}) | "
                  f"α: {avg_alpha_init:.1f}→{avg_alpha_opt:.1f} | | "
                  f"β: {np.mean(list(beta_init.values())):.2f}→{avg_beta_opt:.2f}")

    return results


def plot_cdf(results, topology_name="FaultTree", save_path=None):
    """Plotta la curva CDF."""
    t = np.array(results['t'])
    p_is = np.array(results['p_is'])
    p_mc = np.array(results['p_mc'])
    std_is = np.array(results['std_is'])
    std_mc = np.array(results['std_mc'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot lineare
    ax1 = axes[0]
    ax1.plot(t, p_is, 'b-', linewidth=2, label='IS (Adaptive)')
    ax1.fill_between(t, np.maximum(0, p_is - 1.96*std_is), p_is + 1.96*std_is,
                     alpha=0.3, color='blue')
    ax1.plot(t, p_mc, 'r--', linewidth=2, label='Monte Carlo')
    ax1.fill_between(t, np.maximum(0, p_mc - 1.96*std_mc), p_mc + 1.96*std_mc,
                     alpha=0.3, color='red')
    ax1.set_xlabel('Tempo t')
    ax1.set_ylabel('P(T_fail ≤ t)')
    ax1.set_title(f'CDF - {topology_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot log
    ax2 = axes[1]
    p_is_plot = np.where(p_is > 0, p_is, np.nan)
    p_mc_plot = np.where(p_mc > 0, p_mc, np.nan)
    ax2.semilogy(t, p_is_plot, 'b-', linewidth=2, label='IS (Adaptive)')
    ax2.semilogy(t, p_mc_plot, 'r--', linewidth=2, label='Monte Carlo')
    ax2.set_xlabel('Tempo t')
    ax2.set_ylabel('P(T_fail ≤ t) [log]')
    ax2.set_title(f'CDF (log scale) - {topology_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Salvato: {save_path}")

    plt.close(fig)
    return fig


def plot_alpha_beta_evolution(results, topology_name="FaultTree", save_path=None):
    """Plotta l'evoluzione di α e β nel tempo."""
    t = np.array(results['t'])
    alphas = results['alphas']
    betas = results['betas']
    comps = list(alphas.keys())
    n_comps = len(comps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, n_comps))

    # Plot alpha
    ax1 = axes[0]
    for i, c in enumerate(comps):
        ax1.plot(t, alphas[c], 'o-', markersize=3, color=colors[i], linewidth=1, alpha=0.7)
    ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='α = 1')
    ax1.set_xlabel('Tempo t', fontsize=12)
    ax1.set_ylabel('α (optimized)', fontsize=12)
    ax1.set_title(f'Evoluzione di α (dopo CE) - {topology_name}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # Plot beta
    ax2 = axes[1]
    for i, c in enumerate(comps):
        ax2.plot(t, betas[c], 's-', markersize=3, color=colors[i], linewidth=1, alpha=0.7)
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='β = 1')
    ax2.set_xlabel('Tempo t', fontsize=12)
    ax2.set_ylabel('β (optimized)', fontsize=12)
    ax2.set_title(f'Evoluzione di β (dopo CE) - {topology_name}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Salvato: {save_path}")

    plt.close(fig)
    return fig


def run_cdf_analysis(ft, fault_tree_logic, direct_model, topology_name="FaultTree",
                     t_max=500, t_step=10, sample_model=None,
                     ce_iterations=5, ce_samples=2000):
    """
    Esegue l'analisi CDF completa con Adaptive IS.

    Args:
        ft: FaultTreeGraph
        fault_tree_logic: funzione logica
        direct_model: DirectPredictor già addestrato
        topology_name: nome per i file
        t_max, t_step: parametri temporali
        sample_model: SamplePredictor (opzionale)
        ce_iterations: iterazioni Cross-Entropy (default 4)
        ce_samples: samples per iterazione CE (default 2500)

    Returns:
        dict con risultati
    """
    # Crea directory
    os.makedirs('../results/CDF', exist_ok=True)
    os.makedirs('../results/ALFA_BETA', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Calcola CDF
    results = compute_cdf_curve(
        ft, fault_tree_logic, direct_model,
        sample_model=sample_model,
        t_max=t_max, t_step=t_step,
        ce_iterations=ce_iterations,
        ce_samples=ce_samples
    )

    # Plot CDF
    cdf_path = f'../results/CDF/cdf_{topology_name}_{timestamp}.png'
    plot_cdf(results, topology_name, save_path=cdf_path)

    # Plot α, β
    ab_path = f'../results/ALFA_BETA/alpha_beta_{topology_name}_{timestamp}.png'
    plot_alpha_beta_evolution(results, topology_name, save_path=ab_path)

    # Salva dati
    data_path = f'../results/CDF/cdf_data_{topology_name}_{timestamp}.txt'
    with open(data_path, 'w') as f:
        f.write(f"Topologia: {topology_name}\n")
        f.write(f"T_max: {t_max}, Step: {t_step}\n")
        f.write(f"Metodo: DirectPredictor + Adaptive IS (CE iter={ce_iterations})\n")
        f.write("=" * 70 + "\n\n")

        f.write("t\tP_is\tP_mc\tstd_is\tstd_mc\tn_top_is\tn_top_mc\n")
        for i, t in enumerate(results['t']):
            f.write(f"{t:.1f}\t{results['p_is'][i]:.6e}\t{results['p_mc'][i]:.6e}\t"
                    f"{results['std_is'][i]:.6e}\t{results['std_mc'][i]:.6e}\t"
                    f"{results['n_top_is'][i]}\t{results['n_top_mc'][i]}\n")

    print(f"\nDati salvati: {data_path}")

    return results