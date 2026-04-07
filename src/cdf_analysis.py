"""
CDF Analysis v6 - Usa DirectPredictor + Adaptive IS (Cross-Entropy) + SMC + Weight Clipping

Features:
- SMC (Sequential Monte Carlo) per IS con resampling
- Weight Clipping per ridurre varianza
- ESS (Effective Sample Size) nei log

Plot:
- CDF: IS (log) + MC (log)
- Alpha/Beta: 2 subplot
- CV: andamento nel tempo
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import math
import torch
from typing import Dict, List, Tuple, Callable
from statistics import NormalDist

from direct_predictor import simulate_CTMC_simple
from n_samples_predictor import get_predicted_samples
from adaptive import adaptive_is_cross_entropy

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _pointwise_confidence_interval(p_hat: np.ndarray, std_hat: np.ndarray, 
                                   n_samples: np.ndarray = None,
                                   confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Wilson score confidence interval for binomial proportions (better for rare events)."""
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    
    # Wilson score interval
    z_sq = z * z
    p = p_hat
    n = np.asarray(n_samples, dtype=float)
    
    denominator = 1.0 + z_sq / n
    center = (p + z_sq / (2.0 * n)) / denominator
    margin = z * np.sqrt(p * (1.0 - p) / n + z_sq / (4.0 * n * n)) / denominator
    
    lower = np.clip(center - margin, 0.0, 1.0)
    upper = np.clip(center + margin, 0.0, 1.0)
    return lower, upper


def _dkw_confidence_band(n_samples: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    """DKW simultaneous band width for each time point."""
    alpha = 1.0 - confidence
    n_samples = np.asarray(n_samples, dtype=float)
    epsilon = np.sqrt(np.log(2.0 / alpha) / (2.0 * np.maximum(n_samples, 1.0)))
    return epsilon


# =============================================================================
# SMC Helper Functions
# =============================================================================

# Parametro globale per Defensive Mixture
DEFENSIVE_MIX_RATIO = 0.8


def _simulate_ctmc_step(
    state: Dict[str, int],
    lambda_: Dict[str, float],
    mu_: Dict[str, float],
    alpha: Dict[str, float],
    beta: Dict[str, float],
    t_start: float,
    t_end: float,
    fault_tree: Callable,
    max_transitions: int = 500,
    defensive_mix: float = DEFENSIVE_MIX_RATIO
) -> Tuple[Dict[str, int], float, bool, float]:
    """
    Simula un singolo step temporale del CTMC con IS + Defensive Mixture.

    Defensive Mixture: mescola la proposta IS con la distribuzione originale.
    Questo evita pesi estremi mantenendo la stima unbiased.

    rate_proposal = mix * rate_is + (1-mix) * rate_orig
    """
    comps = list(lambda_.keys())
    state = dict(state)

    t = t_start
    log_w = 0.0
    top_event_hit = False
    n_transitions = 0

    mix = defensive_mix  # 0.9 = 90% IS, 10% originale

    while t < t_end and n_transitions < max_transitions:
        rates_orig = {}
        rates_is = {}
        rates_proposal = {}  # Defensive mixture

        for c in comps:
            if state[c] == 0:
                r_orig = lambda_[c]
                r_is = lambda_[c] * alpha[c]
                rates_orig[('fail', c)] = r_orig
                rates_is[('fail', c)] = r_is
                # Defensive mixture: blend IS con originale
                rates_proposal[('fail', c)] = mix * r_is + (1 - mix) * r_orig
            else:
                r_orig = mu_[c]
                r_is = mu_[c] / beta[c]
                rates_orig[('repair', c)] = r_orig
                rates_is[('repair', c)] = r_is
                # Defensive mixture: blend IS con originale
                rates_proposal[('repair', c)] = mix * r_is + (1 - mix) * r_orig

        R_orig = sum(rates_orig.values())
        R_proposal = sum(rates_proposal.values())

        if R_proposal <= 0:
            break

        # Campiona usando la proposta (defensive mixture)
        dt = np.random.exponential(1.0 / R_proposal)

        if t + dt > t_end:
            remaining = t_end - t
            # Peso per il tempo rimanente: exp((R_proposal - R_orig) * remaining)
            log_w += (R_proposal - R_orig) * remaining
            t = t_end
            break

        t += dt
        # Peso per il tempo: contributo dalla differenza di rate totali
        log_w += (R_proposal - R_orig) * dt + np.log(R_orig / R_proposal)

        # Scegli evento dalla proposta
        r = np.random.random() * R_proposal
        cumsum = 0.0
        chosen_event = None
        for event, rate in rates_proposal.items():
            cumsum += rate
            if r <= cumsum:
                chosen_event = event
                break

        if chosen_event is None:
            break

        # Peso per la scelta dell'evento
        log_w += np.log(rates_orig[chosen_event] / rates_proposal[chosen_event]) + np.log(R_proposal / R_orig)

        event_type, comp = chosen_event
        state[comp] = 1 if event_type == 'fail' else 0

        if fault_tree(state):
            top_event_hit = True
            break

        n_transitions += 1

    return state, log_w, top_event_hit, t


def _systematic_resample(
    particles: List[Dict[str, int]],
    log_weights: List[float],
    top_hits: List[bool],
    hit_times: List[float]
) -> Tuple[List[Dict[str, int]], List[float], List[bool], List[float]]:
    """
    Systematic resampling: duplica particelle con pesi alti, elimina quelle con pesi bassi.
    """
    n = len(particles)

    max_log_w = max(log_weights)
    weights = np.array([math.exp(lw - max_log_w) for lw in log_weights])
    weights /= weights.sum()

    positions = (np.arange(n) + np.random.random()) / n
    cumsum = np.cumsum(weights)

    indices = []
    i, j = 0, 0
    while i < n:
        if positions[i] < cumsum[j]:
            indices.append(j)
            i += 1
        else:
            j += 1
            if j >= n:
                j = n - 1

    new_particles = [dict(particles[i]) for i in indices]
    new_top_hits = [top_hits[i] for i in indices]
    new_hit_times = [hit_times[i] for i in indices]
    new_log_weights = [0.0] * n

    return new_particles, new_log_weights, new_top_hits, new_hit_times


def _effective_sample_size(log_weights: List[float]) -> float:
    """Calcola ESS = (Σw)² / Σw²"""
    if not log_weights:
        return 0.0
    max_log_w = max(log_weights)
    weights = np.array([math.exp(lw - max_log_w) for lw in log_weights])

    if weights.sum() == 0:
        return 0.0

    weights_normalized = weights / weights.sum()
    ess = 1.0 / np.sum(weights_normalized ** 2)
    return ess


def _clip_weights(weights: np.ndarray, percentile: float = 95) -> Tuple[np.ndarray, float]:
    """Weight clipping: taglia i pesi sopra un percentile."""
    if len(weights) == 0:
        return weights, 0.0

    threshold = np.percentile(weights, percentile)
    n_clipped = np.sum(weights > threshold)
    clip_ratio = n_clipped / len(weights)

    clipped = np.minimum(weights, threshold)
    return clipped, clip_ratio


# =============================================================================
# Main CDF Point Computation (SMC + Weight Clipping)
# =============================================================================

def compute_cdf_point(
    lambda_: Dict[str, float],
    mu_: Dict[str, float],
    alpha: Dict[str, float],
    beta: Dict[str, float],
    T: float,
    fault_tree_logic: Callable,
    n_is: int = 10000,
    n_mc: int = 50000,
    n_steps: int = 10,
    ess_threshold: float = 0.5,
    clip_percentile: float = 95
) -> Dict:
    """
    Calcola un punto della CDF usando SMC IS con weight clipping + MC standard.

    Args:
        lambda_, mu_: Rates originali
        alpha, beta: Parametri IS (da CE)
        T: Tempo finale
        fault_tree_logic: Funzione logica
        n_is: Numero particelle IS (SMC)
        n_mc: Numero samples MC
        n_steps: Step temporali SMC
        ess_threshold: Soglia ESS per resampling (0.5 = 50%)
        clip_percentile: Percentile per weight clipping (95 = taglia top 5%)
    """
    comps = list(lambda_.keys())
    dt = T / n_steps

    # === SMC IS ===
    particles = [{c: 0 for c in comps} for _ in range(n_is)]
    log_weights = [0.0] * n_is
    top_hits = [False] * n_is
    hit_times = [T] * n_is

    log_normalization = 0.0
    n_resamples = 0
    ess_final = 0.0

    for step in range(n_steps):
        t_start = step * dt
        t_end = (step + 1) * dt

        n_particles = len(particles)
        for i in range(n_particles):
            if top_hits[i]:
                continue

            new_state, log_w, hit, t_final = _simulate_ctmc_step(
                particles[i], lambda_, mu_, alpha, beta,
                t_start, t_end, fault_tree_logic
            )

            particles[i] = new_state
            log_weights[i] += log_w

            if hit:
                top_hits[i] = True
                hit_times[i] = t_final

        # Calcola ESS sulle particelle attive
        n_particles = len(particles)
        active_indices = [i for i in range(n_particles) if not top_hits[i]]

        if len(active_indices) > 0:
            active_log_weights = [log_weights[i] for i in active_indices]
            ess = _effective_sample_size(active_log_weights)
            ess_ratio = ess / len(active_indices)
        else:
            ess_ratio = 1.0

        # Resample se ESS basso - SOLO sulle particelle attive (non quelle con hit)
        if ess_ratio < ess_threshold and len(active_indices) > 10:
            # Salva particelle con hit (non partecipano al resampling)
            hit_particles = [(i, particles[i], log_weights[i], hit_times[i])
                             for i in range(n_particles) if top_hits[i]]

            # Resampling solo sulle particelle attive
            active_particles = [particles[i] for i in active_indices]
            active_log_weights = [log_weights[i] for i in active_indices]

            # Calcola normalizzazione solo sulle attive
            max_log_w = max(active_log_weights) if active_log_weights else 0.0
            weights_active = np.array([math.exp(lw - max_log_w) for lw in active_log_weights])
            log_normalization += max_log_w + np.log(weights_active.mean())

            # Resample attive
            n_active = len(active_indices)
            weights_active_norm = weights_active / weights_active.sum()

            positions = (np.arange(n_active) + np.random.random()) / n_active
            cumsum = np.cumsum(weights_active_norm)

            new_active_indices = []
            i, j = 0, 0
            while i < n_active:
                if positions[i] < cumsum[j]:
                    new_active_indices.append(j)
                    i += 1
                else:
                    j += 1
                    if j >= n_active:
                        j = n_active - 1

            # Ricostruisci array: prima le hit (preservate), poi le resample attive
            new_particles = []
            new_log_weights = []
            new_top_hits = []
            new_hit_times = []

            # Aggiungi particelle con hit (preservate intatte)
            for _, p, lw, ht in hit_particles:
                new_particles.append(p)
                new_log_weights.append(lw)  # Mantieni peso originale
                new_top_hits.append(True)
                new_hit_times.append(ht)

            # Aggiungi particelle attive resample
            for idx in new_active_indices:
                new_particles.append(dict(active_particles[idx]))
                new_log_weights.append(0.0)  # Reset peso dopo resampling
                new_top_hits.append(False)
                new_hit_times.append(T)

            particles = new_particles
            log_weights = new_log_weights
            top_hits = new_top_hits
            hit_times = new_hit_times
            n_resamples += 1

    # Calcola stima finale
    n_particles_final = len(particles)
    valid_weights = []
    for i in range(n_particles_final):
        if log_weights[i] > -700:
            w = math.exp(log_weights[i])
            if top_hits[i]:
                valid_weights.append(w)
            else:
                valid_weights.append(0.0)
        else:
            valid_weights.append(0.0)

    valid_weights = np.array(valid_weights)
    n_top_is = sum(top_hits)

    # Inizializza risultati
    p_is = 0.0
    cv_is = float('inf')
    ess_final = 0.0
    ess_ratio_final = 0.0

    if n_top_is > 0:
        top_weights = valid_weights[valid_weights > 0]

        # Calcola ESS sui top weights
        if len(top_weights) > 1:
            weights_norm = top_weights / top_weights.sum()
            ess_final = 1.0 / np.sum(weights_norm ** 2)
            ess_ratio_final = ess_final / len(top_weights)

        # Weight Clipping
        if len(top_weights) > 1:
            top_weights_clipped, clip_ratio = _clip_weights(top_weights, clip_percentile)

            # Ricostruisci array con pesi clippati
            clipped_weights = valid_weights.copy()
            top_idx = 0
            for i in range(len(clipped_weights)):
                if clipped_weights[i] > 0:
                    clipped_weights[i] = top_weights_clipped[top_idx]
                    top_idx += 1

            p_is = clipped_weights.mean() * math.exp(log_normalization)

            if np.mean(top_weights_clipped) > 0:
                cv_is = np.std(top_weights_clipped) / np.mean(top_weights_clipped)
            else:
                cv_is = float('inf')
        else:
            p_is = valid_weights.mean() * math.exp(log_normalization)
            cv_is = 1.0

    # Calcola std IS
    if n_top_is > 1 and p_is > 0 and cv_is != float('inf'):
        std_is = p_is * cv_is / math.sqrt(n_top_is)
    else:
        std_is = p_is if p_is > 0 else 0.0

    # === MC standard ===
    alpha_mc = {c: 1.0 for c in comps}
    beta_mc = {c: 1.0 for c in comps}
    results_mc = [simulate_CTMC_simple(lambda_, mu_, alpha_mc, beta_mc, T, fault_tree_logic)
                  for _ in range(n_mc)]
    hits = [1.0 if r['top'] else 0.0 for r in results_mc]
    p_mc = np.mean(hits)
    std_mc = np.std(hits) / np.sqrt(n_mc)
    n_top_mc = sum(hits)
    cv_mc = std_mc / p_mc if p_mc > 0 else float('inf')

    return {
        'p_is': p_is, 'p_mc': p_mc,
        'std_is': std_is, 'std_mc': std_mc,
        'cv_is': cv_is, 'cv_mc': cv_mc,
        'n_top_is': int(n_top_is), 'n_top_mc': int(n_top_mc),
        'ess': ess_final, 'ess_ratio': ess_ratio_final,
        'n_resamples': n_resamples
    }


# =============================================================================
# CDF Curve Computation
# =============================================================================

def compute_cdf_curve(ft, fault_tree_logic, direct_model, sample_model=None,
                      t_max=500, t_step=10,
                      ce_iterations=5, ce_samples=2000,
                      smc_steps=10, ess_threshold=0.5, clip_percentile=95,
                      dkw_confidence=0.95,
                      verbose=True):
    """
    Calcola la curva CDF usando DirectPredictor + Adaptive IS + SMC.
    """
    lambda_, mu_ = ft.get_lambda_mu()
    comps = list(lambda_.keys())
    n_comps = len(comps)

    t_values = np.arange(t_step, t_max + t_step, t_step)

    results = {
        't': [],
        'p_is': [], 'p_mc': [],
        'std_is': [], 'std_mc': [],
        'cv_is': [], 'cv_mc': [],
        'n_samples_is': [], 'n_samples_mc': [],
        'n_top_is': [], 'n_top_mc': [],
        'ess': [], 'ess_ratio': [],
        'alphas': {c: [] for c in comps},
        'betas': {c: [] for c in comps},
        'pointwise_lower': [], 'pointwise_upper': [],
        'pointwise_is_lower': [], 'pointwise_is_upper': [],
        'dkw_lower': [], 'dkw_upper': [],
    }

    n_and = sum(1 for n in ft.nodes if n.get('type') == 'AND')
    n_or = sum(1 for n in ft.nodes if n.get('type') == 'OR')

    if verbose:
        print("=" * 100)
        print("CALCOLO CURVA CDF (DirectPredictor + Adaptive IS + SMC + Weight Clipping + Defensive Mixture)")
        print(f"T: [{t_step}, {t_max}], step={t_step}, punti={len(t_values)}")
        print(f"Componenti: {n_comps}, AND: {n_and}, OR: {n_or}")
        print(f"CE: {ce_iterations} iter, {ce_samples} samples/iter")
        print(f"SMC: {smc_steps} steps, ESS threshold={ess_threshold:.0%}, clip={clip_percentile}%")
        print(f"Defensive Mixture: {DEFENSIVE_MIX_RATIO:.0%} IS + {1-DEFENSIVE_MIX_RATIO:.0%} originale")
        print("=" * 100)

    pyg_data = ft.to_pyg_data().to(device)

    P_THRESHOLD = 1e-2

    for t in t_values:
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Step 1: DirectPredictor → α₀, β₀
        direct_model.eval()
        alpha_init, beta_init = direct_model.predict(ft, T=t, T_max=float(t_max))

        # Step 2: Determina numero samples
        if sample_model is not None:
            sample_model.to(device)
            n_is, n_mc = get_predicted_samples(sample_model, pyg_data, T=float(t), T_max=float(t_max))
        else:
            n_is = 30000
            n_mc = 10000000

        # Step 3: Cross-Entropy per ottimizzare α, β
        alpha_opt, beta_opt, ce_stats = adaptive_is_cross_entropy(
            lambda_, mu_, t, fault_tree_logic,
            alpha_init, beta_init,
            n_iterations=ce_iterations,
            n_samples_per_iter=ce_samples,
            elite_fraction=0.05,
            smoothing=0.6,
            verbose=False
        )

        # Step 4: Calcola punto CDF con SMC + weight clipping
        cdf_point = compute_cdf_point(
            lambda_, mu_, alpha_opt, beta_opt, float(t), fault_tree_logic,
            n_is=n_is, n_mc=n_mc,
            n_steps=smc_steps,
            ess_threshold=ess_threshold,
            clip_percentile=clip_percentile
        )

        # Salva risultati
        results['t'].append(t)
        results['p_is'].append(cdf_point['p_is'])
        results['p_mc'].append(cdf_point['p_mc'])
        results['std_is'].append(cdf_point['std_is'])
        results['std_mc'].append(cdf_point['std_mc'])
        results['cv_is'].append(cdf_point['cv_is'])
        results['cv_mc'].append(cdf_point['cv_mc'])
        results['n_samples_is'].append(n_is)
        results['n_samples_mc'].append(n_mc)
        results['n_top_is'].append(cdf_point['n_top_is'])
        results['n_top_mc'].append(cdf_point['n_top_mc'])
        results['ess'].append(cdf_point['ess'])
        results['ess_ratio'].append(cdf_point['ess_ratio'])

        p_mc = cdf_point['p_mc']
        std_mc = cdf_point['std_mc']
        p_is = cdf_point['p_is']
        std_is = cdf_point['std_is']
        
        pointwise_lower, pointwise_upper = _pointwise_confidence_interval(
            np.array([p_mc]), np.array([std_mc]), np.array([n_mc])
        )
        pointwise_is_lower, pointwise_is_upper = _pointwise_confidence_interval(
            np.array([p_is]), np.array([std_is]), np.array([n_is])
        )
        dkw_eps = _dkw_confidence_band(np.array([n_mc]), confidence=dkw_confidence)
        results['pointwise_lower'].append(float(pointwise_lower[0]))
        results['pointwise_upper'].append(float(pointwise_upper[0]))
        results['pointwise_is_lower'].append(float(pointwise_is_lower[0]))
        results['pointwise_is_upper'].append(float(pointwise_is_upper[0]))
        results['dkw_lower'].append(float(np.clip(p_mc - dkw_eps[0], 0.0, 1.0)))
        results['dkw_upper'].append(float(np.clip(p_mc + dkw_eps[0], 0.0, 1.0)))

        for c in comps:
            results['alphas'][c].append(alpha_opt[c])
            results['betas'][c].append(beta_opt[c])

        if verbose:
            avg_alpha_init = np.mean(list(alpha_init.values()))
            avg_alpha_opt = np.mean(list(alpha_opt.values()))
            avg_beta_init = np.mean(list(beta_init.values()))
            avg_beta_opt = np.mean(list(beta_opt.values()))

            cv_is_str = f"{cdf_point['cv_is']:.0%}" if cdf_point['cv_is'] != float('inf') else "inf"
            cv_mc_str = f"{cdf_point['cv_mc']:.0%}" if cdf_point['cv_mc'] != float('inf') else "inf"
            ess_str = f"{cdf_point['ess']:.1f}" if cdf_point['ess'] > 0 else "N/A"
            ess_ratio_str = f"{cdf_point['ess_ratio']:.0%}" if cdf_point['ess_ratio'] > 0 else "N/A"

            print(f"[{timestamp}] T={t:3.0f} | "
                  f"P_is={cdf_point['p_is']:.2e} (CV={cv_is_str}, n={cdf_point['n_top_is']}/{n_is}) | "
                  f"P_mc={cdf_point['p_mc']:.2e} (CV={cv_mc_str}, n={cdf_point['n_top_mc']}/{n_mc}) | "
                  f"α: {avg_alpha_init:.1f}→{avg_alpha_opt:.1f} | "
                  f"β: {avg_beta_init:.1f}→{avg_beta_opt:.1f} | "
                  f"ESS: {ess_str} ({ess_ratio_str})")

        if cdf_point['p_is'] >= P_THRESHOLD:
            if verbose:
                print(f"\n*** EARLY STOP: P_is = {cdf_point['p_is']:.2e} >= {P_THRESHOLD:.0e} ***")
                print(f"*** Evento non più raro, calcolo CDF terminato a T={t} ***\n")
            break

    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_cdf(results, topology_name="FaultTree", save_path=None):
    """
    Plot unico: CDF IS + MC, bande Pointwise Wilson per entrambi, DKW per MC (scala log).
    
    - Blu: IS con banda Pointwise Wilson (area leggera)
    - Rosso: MC con banda Pointwise Wilson (area leggera)
    - Arancio: Banda DKW su MC (linee sottili)
    """
    t = np.array(results['t'])
    p_is = np.array(results['p_is'])
    p_mc = np.array(results['p_mc'])
    pointwise_lower = np.array(results.get('pointwise_lower', []))
    pointwise_upper = np.array(results.get('pointwise_upper', []))
    pointwise_is_lower = np.array(results.get('pointwise_is_lower', []))
    pointwise_is_upper = np.array(results.get('pointwise_is_upper', []))
    dkw_lower = np.array(results.get('dkw_lower', []))
    dkw_upper = np.array(results.get('dkw_upper', []))

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Transforma in log-safe
    p_is_safe = np.clip(p_is, 1e-6, 1.0)
    p_mc_safe = np.clip(p_mc, 1e-6, 1.0)

    # === Banda DKW su MC (linee visibili) ===
    if len(dkw_lower) == len(t) and len(dkw_upper) == len(t):
        band_lower = np.clip(dkw_lower, 1e-6, 1.0)
        band_upper = np.clip(dkw_upper, 1e-6, 1.0)
        ax.plot(t, band_lower, color='orange', linewidth=1.5, linestyle='--', alpha=0.8,
                label='DKW 95% band (MC)', zorder=1)
        ax.plot(t, band_upper, color='orange', linewidth=1.5, linestyle='--', alpha=0.8, zorder=1)

    # === Banda Pointwise Wilson su MC (fill) ===
    if len(pointwise_lower) == len(t) and len(pointwise_upper) == len(t):
        pw_lower = np.clip(pointwise_lower, 1e-6, 1.0)
        pw_upper = np.clip(pointwise_upper, 1e-6, 1.0)
        ax.fill_between(t, pw_lower, pw_upper, color='red', alpha=0.12, 
                        label='Pointwise Wilson 95% (MC)', zorder=2)

    # === CDF MC (linea rossa spessa) ===
    ax.plot(t, p_mc_safe, 'r-', linewidth=2.5, label='MC', marker='s', 
           markersize=5, alpha=0.85, zorder=4)

    # === Banda Pointwise Wilson su IS (fill) ===
    if len(pointwise_is_lower) == len(t) and len(pointwise_is_upper) == len(t):
        pw_lower_is = np.clip(pointwise_is_lower, 1e-6, 1.0)
        pw_upper_is = np.clip(pointwise_is_upper, 1e-6, 1.0)
        ax.fill_between(t, pw_lower_is, pw_upper_is, color='blue', alpha=0.12, 
                        label='Pointwise Wilson 95% (IS)', zorder=2)

    # === CDF IS (linea blu spessa) ===
    ax.plot(t, p_is_safe, 'b-', linewidth=2.5, label='IS', marker='o', 
           markersize=5, alpha=0.85, zorder=3)

    ax.set_xlabel('Tempo t', fontsize=12, fontweight='bold')
    ax.set_ylabel('P(T_fail ≤ t)', fontsize=12, fontweight='bold')
    ax.set_title(f'CDF: IS vs MC con Bande di Confidenza - {topology_name}', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Salvato: {save_path}")

    plt.close(fig)
    return fig


def plot_alpha_beta(results, topology_name="FaultTree", save_path=None):
    """Plotta l'evoluzione di α e β nel tempo."""
    t = np.array(results['t'])
    alphas = results['alphas']
    betas = results['betas']
    comps = list(alphas.keys())
    n_comps = len(comps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.colormaps['viridis'](np.linspace(0, 1, n_comps))

    ax1 = axes[0]
    for i, c in enumerate(comps):
        ax1.plot(t, alphas[c], 'o-', markersize=3, color=colors[i], linewidth=1, alpha=0.7)
    ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    ax1.set_xlabel('Tempo t', fontsize=12)
    ax1.set_ylabel('α (optimized)', fontsize=12)
    ax1.set_title(f'Evoluzione di α - {topology_name}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend([f'{n_comps} componenti'], loc='upper right', fontsize=9)

    ax2 = axes[1]
    for i, c in enumerate(comps):
        ax2.plot(t, betas[c], 's-', markersize=3, color=colors[i], linewidth=1, alpha=0.7)
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    ax2.set_xlabel('Tempo t', fontsize=12)
    ax2.set_ylabel('β (optimized)', fontsize=12)
    ax2.set_title(f'Evoluzione di β - {topology_name}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend([f'{n_comps} componenti'], loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Salvato: {save_path}")

    plt.close(fig)
    return fig


def plot_cv(results, topology_name="FaultTree", save_path=None):
    """Plotta l'andamento del CV nel tempo."""
    t = np.array(results['t'])
    cv_is = np.array(results['cv_is'])
    cv_mc = np.array(results['cv_mc'])

    cv_is_plot = np.where(np.isinf(cv_is), np.nan, cv_is)
    cv_mc_plot = np.where(np.isinf(cv_mc), np.nan, cv_mc)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(t, cv_is_plot * 100, 'b-o', linewidth=2, markersize=4, label='CV IS (%)')
    ax.plot(t, cv_mc_plot * 100, 'r--s', linewidth=2, markersize=4, label='CV MC (%)')
    ax.axhline(y=50, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='CV = 50%')
    ax.axhline(y=100, color='red', linestyle=':', linewidth=1, alpha=0.7, label='CV = 100%')

    ax.set_xlabel('Tempo t', fontsize=12)
    ax.set_ylabel('Coefficiente di Variazione (%)', fontsize=12)
    ax.set_title(f'Andamento CV nel tempo - {topology_name}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    max_cv_is = np.nanmax(cv_is_plot) * 100 if not np.all(np.isnan(cv_is_plot)) else 100
    ax.set_ylim(0, min(500, max(200, max_cv_is * 1.2)))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Salvato: {save_path}")

    plt.close(fig)
    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def run_cdf_analysis(ft, fault_tree_logic, direct_model, topology_name="FaultTree",
                     t_max=500, t_step=10, sample_model=None,
                     ce_iterations=5, ce_samples=3000,
                     smc_steps=20, ess_threshold=0.5, clip_percentile=90, dkw_confidence=0.95):
    """
    Esegue l'analisi CDF completa con Adaptive IS + SMC + Weight Clipping.

    Args:
        ft: FaultTreeGraph
        fault_tree_logic: funzione logica
        direct_model: DirectPredictor già addestrato
        topology_name: nome per i file
        t_max, t_step: parametri temporali
        sample_model: SamplePredictor (opzionale)
        ce_iterations: iterazioni Cross-Entropy
        ce_samples: samples per iterazione CE
        smc_steps: step temporali per SMC
        ess_threshold: soglia ESS per resampling
        clip_percentile: percentile per weight clipping

    Returns:
        dict con risultati
    """
    os.makedirs(os.path.join(RESULTS_DIR, 'CDF'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'ALFA_BETA'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'CV'), exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results = compute_cdf_curve(
        ft, fault_tree_logic, direct_model,
        sample_model=sample_model,
        t_max=t_max, t_step=t_step,
        ce_iterations=ce_iterations,
        ce_samples=ce_samples,
        smc_steps=smc_steps,
        ess_threshold=ess_threshold,
        clip_percentile=clip_percentile,
        dkw_confidence=dkw_confidence
    )

    cdf_path = os.path.join(RESULTS_DIR, 'CDF', f'cdf_{topology_name}_{timestamp}.png')
    plot_cdf(results, topology_name, save_path=cdf_path)

    ab_path = os.path.join(RESULTS_DIR, 'ALFA_BETA', f'alpha_beta_{topology_name}_{timestamp}.png')
    plot_alpha_beta(results, topology_name, save_path=ab_path)

    cv_path = os.path.join(RESULTS_DIR, 'CV', f'cv_{topology_name}_{timestamp}.png')
    plot_cv(results, topology_name, save_path=cv_path)

    data_path = os.path.join(RESULTS_DIR, 'CDF', f'cdf_data_{topology_name}_{timestamp}.txt')
    with open(data_path, 'w') as f:
        f.write(f"Topologia: {topology_name}\n")
        f.write(f"T_max: {t_max}, Step: {t_step}\n")
        f.write(f"Metodo: DirectPredictor + Adaptive IS + SMC + Weight Clipping\n")
        f.write(f"SMC steps: {smc_steps}, ESS threshold: {ess_threshold}, Clip: {clip_percentile}%\n")
        f.write("Pointwise CI: normal approximation on MC estimates (95%)\n")
        f.write("DKW band: simultaneous 95% band on MC estimates\n")
        f.write("=" * 100 + "\n\n")

        f.write("t\tP_is\tP_mc\tstd_is\tstd_mc\tcp_lo\tcp_hi\tdkw_lo\tdkw_hi\tcv_is\tcv_mc\tn_top_is\tn_top_mc\tess\tess_ratio\n")
        for i, t in enumerate(results['t']):
            f.write(f"{t:.1f}\t{results['p_is'][i]:.6e}\t{results['p_mc'][i]:.6e}\t"
                    f"{results['std_is'][i]:.6e}\t{results['std_mc'][i]:.6e}\t"
                    f"{results['pointwise_lower'][i]:.6e}\t{results['pointwise_upper'][i]:.6e}\t"
                    f"{results['dkw_lower'][i]:.6e}\t{results['dkw_upper'][i]:.6e}\t"
                    f"{results['cv_is'][i]:.4f}\t{results['cv_mc'][i]:.4f}\t"
                    f"{results['n_top_is'][i]}\t{results['n_top_mc'][i]}\t"
                    f"{results['ess'][i]:.2f}\t{results['ess_ratio'][i]:.4f}\n")

    print(f"\nDati salvati: {data_path}")

    return results