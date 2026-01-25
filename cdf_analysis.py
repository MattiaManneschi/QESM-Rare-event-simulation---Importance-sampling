"""
Modulo per calcolare e plottare la CDF (Cumulative Distribution Function)
della probabilità di fallimento del sistema.

P(t) = P(T_fail ≤ t) = probabilità che il sistema fallisca entro il tempo t

INTEGRATO con la pipeline esistente:
- Usa RangePredictor già addestrato per ottenere i range α/β
- Usa SamplePredictor per determinare il numero di samples ottimale
- Per ogni t, addestra MLP e calcola P(t)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import math
import torch


from is_optimizer_evaluator import simulate_CTMC, ExternalConfig, train_mlp_cross_entropy
from N_samples_predictor import SamplePredictor, get_predicted_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_cdf_point(lambda_, mu_, alpha, beta, t, fault_tree_logic, n_is=500, n_mc=2000, verbose=False):
    """
    Calcola un singolo punto della CDF usando IS e MC.

    Usa Self-Normalized Importance Sampling per robustezza ai pesi grandi.

    Args:
        n_is: numero samples per IS
        n_mc: numero samples per MC
        verbose: stampa diagnostica sui pesi
    """
    comps = list(lambda_.keys())

    # IS
    results_is = [simulate_CTMC(lambda_, mu_, alpha, beta, t, fault_tree_logic)
                  for _ in range(n_is)]

    # Estrai log_w per tutti (non solo top events)
    # Self-normalized IS: P = sum(w_i * I_i) / sum(w_i) dove I_i = 1 se top event
    all_log_w = [r['log_w'] for r in results_is]
    top_indicators = [1.0 if r['top'] else 0.0 for r in results_is]

    # Stabilizzazione numerica: sottrai max(log_w)
    max_log_w = max(all_log_w)
    stable_weights = [math.exp(lw - max_log_w) for lw in all_log_w]

    # Self-normalized IS estimate
    numerator = sum(w * ind for w, ind in zip(stable_weights, top_indicators))
    denominator = sum(stable_weights)

    if denominator > 0:
        p_is = numerator / denominator
    else:
        p_is = 0.0

    # Varianza approssimata per self-normalized IS
    n_top_is = sum(top_indicators)
    if n_top_is > 0 and denominator > 0:
        # Effective sample size
        ess = (sum(stable_weights) ** 2) / sum(w**2 for w in stable_weights)
        std_is = math.sqrt(p_is * (1 - p_is) / max(ess, 1))
    else:
        std_is = 0.0

    if verbose:
        pass  # Debug rimosso

    # MC (unchanged)
    alpha_mc = {c: 1.0 for c in comps}
    beta_mc = {c: 1.0 for c in comps}
    results_mc = [simulate_CTMC(lambda_, mu_, alpha_mc, beta_mc, t, fault_tree_logic)
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


def compute_cdf_curve(ft, fault_tree_logic, range_model, sample_model=None,
                      t_max=100, t_step=5, n_samples_fallback=1000,
                      training_epochs=30, verbose=True):
    """
    Calcola l'intera curva CDF usando la pipeline esistente.

    Args:
        ft: FaultTreeGraph
        fault_tree_logic: funzione booleana del fault tree
        range_model: RangePredictor addestrato (predice range α/β per ogni t)
        sample_model: SamplePredictor addestrato (se None, usa n_samples_fallback)
        t_max: tempo massimo
        t_step: passo temporale
        n_samples_fallback: samples se sample_model è None
        training_epochs: epoche per addestrare MLP a ogni t
        verbose: stampa progresso

    Returns:
        dict con t, p_is, p_mc, alphas, betas, etc.
    """
    lambda_, mu_ = ft.get_lambda_mu()
    comps = list(lambda_.keys())
    n_comps = len(comps)

    # Prepara dati PyG base
    pyg_data = ft.to_pyg_data()
    pyg_data = pyg_data.to(device)

    t_values = np.arange(t_step, t_max + t_step, t_step)

    results = {
        't': [],
        'p_is': [], 'p_mc': [],
        'std_is': [], 'std_mc': [],
        'n_samples_is': [], 'n_samples_mc': [],
        'alphas': {c: [] for c in comps},
        'betas': {c: [] for c in comps},
        'ranges_alpha': [], 'ranges_beta': []
    }

    if verbose:
        print("=" * 60)
        print("CALCOLO CURVA CDF (range adattivi per ogni t)")
        print(f"T: [{t_step}, {t_max}], step={t_step}, punti={len(t_values)}")
        print(f"SamplePredictor: {'Sì' if sample_model else 'No (fallback)'}")
        print("=" * 60)

    for t in t_values:
        if verbose:
            print(f"\n[T = {t:.0f}] ", end="")

        # 1. Predici range α/β dalla topologia CON T
        range_model.to(device)

        with torch.no_grad():
            # Passa T al RangePredictor (nuova interfaccia)
            ranges_pred, _ = range_model(pyg_data, T=t, T_max=500.0)

        # Usa i range predetti dal modello (che ora considera T)
        ranges_dict = {
            'alpha': (ranges_pred[0, 0].item(), ranges_pred[0, 1].item()),
            'beta': (ranges_pred[0, 2].item(), ranges_pred[0, 3].item())
        }

        if verbose:
            print(f"α:[{ranges_dict['alpha'][0]:.2f},{ranges_dict['alpha'][1]:.2f}] "
                  f"β:[{ranges_dict['beta'][0]:.2f},{ranges_dict['beta'][1]:.2f}] | \n", end="")

        # 2. Determina numero samples
        if sample_model is not None:
            sample_model.to(device)
            n_is, n_mc = get_predicted_samples(sample_model, pyg_data)
        else:
            n_is, n_mc = n_samples_fallback, n_samples_fallback * 5

        # 3. Configura ExternalConfig per questo t con range specifici
        config = ExternalConfig(lambda_, mu_, fault_tree_logic, ranges_dict, T=t)
        config.epochs = training_epochs
        config.n_samples = 500
        config.n_trajectories = 500

        # 4. Addestra MLP da zero per questo t
        model = train_mlp_cross_entropy(config)

        # 5. Estrai α, β
        input_tensor = torch.eye(len(comps)).mean(dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            alpha_tensor, beta_tensor = model(input_tensor)

        alpha = {c: alpha_tensor[0, i].item() for i, c in enumerate(comps)}
        beta = {c: beta_tensor[0, i].item() for i, c in enumerate(comps)}

        # 6. Calcola P(t) con diagnostica
        cdf_point = compute_cdf_point(lambda_, mu_, alpha, beta, t,
                                       fault_tree_logic, n_is, n_mc, verbose=True)

        # 7. Salva
        results['t'].append(t)
        results['p_is'].append(cdf_point['p_is'])
        results['p_mc'].append(cdf_point['p_mc'])
        results['std_is'].append(cdf_point['std_is'])
        results['std_mc'].append(cdf_point['std_mc'])
        results['n_samples_is'].append(n_is)
        results['n_samples_mc'].append(n_mc)
        results['ranges_alpha'].append(ranges_dict['alpha'])
        results['ranges_beta'].append(ranges_dict['beta'])

        for c in comps:
            results['alphas'][c].append(alpha[c])
            results['betas'][c].append(beta[c])

        if verbose:
            avg_alpha = np.mean([alpha[c] for c in comps])
            avg_beta = np.mean([beta[c] for c in comps])
            print(f"P_is={cdf_point['p_is']:.2e}, P_mc={cdf_point['p_mc']:.2e} | "
                  f"ᾱ={avg_alpha:.2f}, β̄={avg_beta:.2f}")

        # Stop se P > 10% (oltre questa soglia l'unreliability non è interessante)
        if cdf_point['p_mc'] > 0.1:
            if verbose:
                print(f"\n[STOP] P_is > 10%, interrompo a T={t}")
            break

    return results


def plot_cdf(results, topology_name="FaultTree", save_path=None):
    """
    Plotta la curva CDF con bande di confidenza.
    """
    t = np.array(results['t'])
    p_is = np.array(results['p_is'])
    p_mc = np.array(results['p_mc'])
    std_is = np.array(results['std_is'])
    std_mc = np.array(results['std_mc'])

    fig, ax = plt.subplots(figsize=(10, 6))

    # IS
    ax.plot(t, p_is, 'b-', linewidth=2, label='Importance Sampling')
    ax.fill_between(t, p_is - 1.96*std_is, p_is + 1.96*std_is, alpha=0.3, color='blue')

    # MC
    ax.plot(t, p_mc, 'r--', linewidth=2, label='Monte Carlo')
    ax.fill_between(t, p_mc - 1.96*std_mc, p_mc + 1.96*std_mc, alpha=0.3, color='red')

    # Linea 1%
    ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, label='P = 1%')

    ax.set_xlabel('Tempo t', fontsize=12)
    ax.set_ylabel('P(T_fail ≤ t)', fontsize=12)
    ax.set_title(f'CDF - Probabilità di Fallimento\n{topology_name}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(t) * 1.05])
    ax.set_ylim([0, max(max(p_is), max(p_mc)) * 1.15])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Salvato: {save_path}")

    plt.close(fig)
    return fig


def plot_alpha_beta_evolution(results, topology_name="FaultTree", save_path=None):
    """
    Plotta l'evoluzione di α e β nel tempo per ogni componente.

    Comportamento atteso (dal ricercatore):
    - α, β molto alti per t piccoli
    - Convergono verso 1 per t grandi
    """
    t = np.array(results['t'])
    alphas = results['alphas']
    betas = results['betas']
    comps = list(alphas.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # α(t)
    ax1 = axes[0]
    for c in comps:
        ax1.plot(t, alphas[c], 'o-', markersize=4, label=c)
    ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='α = 1')
    ax1.set_xlabel('Tempo t', fontsize=12)
    ax1.set_ylabel('α (failure rate multiplier)', fontsize=12)
    ax1.set_title(f'Evoluzione di α nel tempo\n{topology_name}', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # β(t)
    ax2 = axes[1]
    for c in comps:
        ax2.plot(t, betas[c], 's-', markersize=4, label=c)
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='β = 1')
    ax2.set_xlabel('Tempo t', fontsize=12)
    ax2.set_ylabel('β (repair rate multiplier)', fontsize=12)
    ax2.set_title(f'Evoluzione di β nel tempo\n{topology_name}', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Salvato: {save_path}")

    plt.close(fig)
    return fig


def run_cdf_analysis(ft, fault_tree_logic, range_model, topology_name="FaultTree",
                     t_max=100, t_step=5, sample_model=None):
    """
    Funzione principale: calcola CDF, plotta, salva.

    Args:
        ft: FaultTreeGraph
        fault_tree_logic: funzione booleana
        range_model: RangePredictor addestrato
        topology_name: nome della topologia
        t_max: tempo massimo
        t_step: passo temporale
        sample_model: SamplePredictor addestrato (opzionale)

    Returns:
        results dict
    """
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Calcola CDF con range adattivi per ogni t
    results = compute_cdf_curve(ft, fault_tree_logic, range_model,
                                sample_model=sample_model,
                                t_max=t_max, t_step=t_step)

    # 2. Plot CDF
    cdf_path = f'results/CDF/cdf_{topology_name}_{timestamp}.png'
    plot_cdf(results, topology_name, save_path=cdf_path)

    # 3. Plot α/β
    ab_path = f'results/ALFA_BETA/alpha_beta_{topology_name}_{timestamp}.png'
    plot_alpha_beta_evolution(results, topology_name, save_path=ab_path)

    # 4. Salva dati
    data_path = f'results/CDF/cdf_data_{topology_name}_{timestamp}.txt'
    with open(data_path, 'w') as f:
        f.write(f"Topologia: {topology_name}\n")
        f.write(f"T_max: {t_max}, Step: {t_step}\n")
        f.write("=" * 60 + "\n\n")

        f.write("t\tRange_α\tRange_β\tP_is\tP_mc\tstd_is\tstd_mc\n")
        for i, t in enumerate(results['t']):
            r_a = results['ranges_alpha'][i]
            r_b = results['ranges_beta'][i]
            f.write(f"{t:.1f}\t[{r_a[0]:.2f},{r_a[1]:.2f}]\t[{r_b[0]:.2f},{r_b[1]:.2f}]\t"
                    f"{results['p_is'][i]:.6e}\t{results['p_mc'][i]:.6e}\t"
                    f"{results['std_is'][i]:.6e}\t{results['std_mc'][i]:.6e}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("ALPHA per componente:\n")
        for c in results['alphas']:
            vals = [f"{v:.3f}" for v in results['alphas'][c]]
            f.write(f"  {c}: {vals}\n")

        f.write("\nBETA per componente:\n")
        for c in results['betas']:
            vals = [f"{v:.3f}" for v in results['betas'][c]]
            f.write(f"  {c}: {vals}\n")

    print(f"Dati salvati: {data_path}")

    return results