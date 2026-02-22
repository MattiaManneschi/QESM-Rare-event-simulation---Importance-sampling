"""
Adaptive IS via Cross-Entropy

MODIFICHE rispetto alla versione originale:
  - Escalation: modifica sia α che β (prima solo α)
  - Cooldown: modifica sia α che β (prima solo α)
  - Cap: α ≤ 10, β ≤ 10 (prima α ≤ 200)
  - β trattato simmetricamente ad α
  - MLE per-componente da elites (come prima)
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Detailed CTMC simulation (serve per avere stats per-componente)
# ---------------------------------------------------------------------------

def _simulate_CTMC_detailed(lambda_, mu_, alpha, beta, T, fault_tree):
    """
    CTMC simulation con IS che traccia anche statistiche per-componente:
      n_fail, n_repair, time_up, time_down per ogni componente.
    Calcolo log_w identico a simulate_CTMC_simple.
    """
    comps = list(lambda_.keys())
    state = {c: 0 for c in comps}

    n_fail   = {c: 0   for c in comps}
    n_repair = {c: 0   for c in comps}
    time_up  = {c: 0.0 for c in comps}
    time_down = {c: 0.0 for c in comps}

    t = 0.0
    log_w = 0.0
    top_event_hit = False
    n_transitions = 0
    max_transitions = 5000

    while t < T and n_transitions < max_transitions:
        rates_orig = {}
        rates_is   = {}

        for c in comps:
            if state[c] == 0:
                rates_orig[('fail', c)]   = lambda_[c]
                rates_is[('fail', c)]     = lambda_[c] * alpha[c]
            else:
                rates_orig[('repair', c)] = mu_[c]
                rates_is[('repair', c)]   = mu_[c] / beta[c]

        R_orig = sum(rates_orig.values())
        R_is   = sum(rates_is.values())

        if R_is <= 0:
            break

        dt = np.random.exponential(1.0 / R_is)

        if t + dt > T:
            remaining = T - t
            log_w += (R_is - R_orig) * remaining
            for c in comps:
                if state[c] == 0:
                    time_up[c] += remaining
                else:
                    time_down[c] += remaining
            break

        for c in comps:
            if state[c] == 0:
                time_up[c] += dt
            else:
                time_down[c] += dt

        t += dt
        log_w += (R_is - R_orig) * dt + np.log(R_orig / R_is)

        r = np.random.random() * R_is
        cumsum = 0.0
        chosen_event = None
        for event, rate in rates_is.items():
            cumsum += rate
            if r <= cumsum:
                chosen_event = event
                break

        if chosen_event is None:
            break

        log_w += (np.log(rates_orig[chosen_event] / rates_is[chosen_event])
                  + np.log(R_is / R_orig))

        event_type, comp = chosen_event
        if event_type == 'fail':
            state[comp] = 1
            n_fail[comp] += 1
        else:
            state[comp] = 0
            n_repair[comp] += 1

        if fault_tree(state):
            top_event_hit = True
            break

        n_transitions += 1

    return {
        'top': top_event_hit,
        'log_w': log_w,
        't_final': t,
        'n_fail': n_fail,
        'n_repair': n_repair,
        'time_up': time_up,
        'time_down': time_down,
        'state_final': dict(state),
    }


# ---------------------------------------------------------------------------
# Cross-Entropy con α e β simmetrici
# ---------------------------------------------------------------------------

# === PARAMETRI GLOBALI (facili da modificare) ===
MAX_ALPHA = 100.0  # Cap massimo per α (era 200)
MAX_BETA = 100.0   # Cap massimo per β (era 10)
ESCALATION_FACTOR = 1.3  # Fattore moltiplicativo per escalation
COOLDOWN_FACTOR = 0.8    # Fattore moltiplicativo per cooldown


def adaptive_is_cross_entropy(
    lambda_, mu_, T, fault_tree,
    alpha_init, beta_init,
    n_iterations=5,
    n_samples_per_iter=5000,
    elite_fraction=0.2,
    smoothing=0.7,
    verbose=False
):
    """
    Cross-Entropy method con α e β trattati simmetricamente.
    
    MODIFICHE:
    - Escalation modifica sia α che β
    - Cooldown modifica sia α che β
    - Cap ridotto: α ≤ 10, β ≤ 10
    """
    comps = list(lambda_.keys())

    alpha = {c: float(alpha_init[c]) for c in comps}
    beta  = {c: float(beta_init[c])  for c in comps}

    n_elite = max(10, int(n_samples_per_iter * elite_fraction))

    stats = {
        'iterations': [],
        'n_top': [],
        'alpha_mean': [],
        'beta_mean': [],
        'cv': [],
    }

    if verbose:
        print(f"{'=' * 60}")
        print("ADAPTIVE IS – CROSS ENTROPY (α e β simmetrici)")
        print(f"T={T}, samples/iter={n_samples_per_iter}, elite={n_elite}")
        print(f"α₀ medio: {np.mean(list(alpha.values())):.2f}, "
              f"β₀ medio: {np.mean(list(beta.values())):.2f}")
        print(f"Cap: α ≤ {MAX_ALPHA}, β ≤ {MAX_BETA}")
        print(f"{'=' * 60}")

    for iteration in range(n_iterations):
        # --- Simulate (detailed per avere stats per-componente) ---
        results = [
            _simulate_CTMC_detailed(lambda_, mu_, alpha, beta, T, fault_tree)
            for _ in range(n_samples_per_iter)
        ]

        top_results = [r for r in results if r['top']]
        n_top = len(top_results)

        if verbose:
            print(f"Iter {iteration + 1}: n_top = {n_top}/{n_samples_per_iter} "
                  f"({100 * n_top / n_samples_per_iter:.1f}%)")

        # === MODIFICA: Escalation per ENTRAMBI α e β ===
        if n_top < n_elite:
            alpha = {c: min(alpha[c] * ESCALATION_FACTOR, MAX_ALPHA) for c in comps}
            beta  = {c: min(beta[c]  * ESCALATION_FACTOR, MAX_BETA)  for c in comps}

            if verbose:
                print(f"  → Escalation ({n_top}<{n_elite}) → "
                      f"α medio: {np.mean(list(alpha.values())):.2f}, "
                      f"β medio: {np.mean(list(beta.values())):.2f}")

            stats['iterations'].append(iteration)
            stats['n_top'].append(n_top)
            stats['alpha_mean'].append(np.mean(list(alpha.values())))
            stats['beta_mean'].append(np.mean(list(beta.values())))
            stats['cv'].append(float('inf'))
            continue

        # === MODIFICA: Cooldown per ENTRAMBI α e β ===
        if n_top > n_samples_per_iter * 0.5:
            alpha = {c: max(alpha[c] * COOLDOWN_FACTOR, 1.0) for c in comps}
            beta  = {c: max(beta[c]  * COOLDOWN_FACTOR, 1.0) for c in comps}

            if verbose:
                print(f"  → Troppi top events, riduco → "
                      f"α medio: {np.mean(list(alpha.values())):.2f}, "
                      f"β medio: {np.mean(list(beta.values())):.2f}")

            stats['iterations'].append(iteration)
            stats['n_top'].append(n_top)
            stats['alpha_mean'].append(np.mean(list(alpha.values())))
            stats['beta_mean'].append(np.mean(list(beta.values())))
            stats['cv'].append(float('inf'))
            continue

        # === Select elite samples (highest log_w tra i top events) ===
        top_sorted = sorted(top_results, key=lambda r: r['log_w'], reverse=True)
        elite = top_sorted[:n_elite]

        # === MLE per-componente da elites ===
        for c in comps:
            total_fail   = sum(r['n_fail'][c]   for r in elite)
            total_repair = sum(r['n_repair'][c] for r in elite)
            total_t_up   = sum(r['time_up'][c]  for r in elite)
            total_t_down = sum(r['time_down'][c] for r in elite)

            # α_mle = (n_fail / t_up) / λ
            if total_t_up > 1e-12 and total_fail > 0:
                alpha_mle = (total_fail / total_t_up) / lambda_[c]
                alpha_new = max(1.0, min(MAX_ALPHA, alpha_mle))
            else:
                alpha_new = alpha[c]

            # === MODIFICA: β_mle simmetrico ad α ===
            # β_mle = μ / (n_repair / t_down) = μ * t_down / n_repair
            # Interpretazione: se nelle elite ci sono poche riparazioni,
            # β deve essere alto per rallentare ulteriormente le riparazioni
            if total_t_down > 1e-12 and total_repair > 0:
                # Rate di riparazione osservata nelle elite
                repair_rate_observed = total_repair / total_t_down
                # Rate originale = μ, rate IS = μ/β
                # Se repair_rate_observed è bassa, β deve essere alto
                beta_mle = mu_[c] / repair_rate_observed if repair_rate_observed > 1e-12 else beta[c]
                beta_new = max(1.0, min(MAX_BETA, beta_mle))
            elif total_t_down > 1e-12 and total_repair == 0:
                # Nessuna riparazione nelle elite → β deve essere molto alto
                beta_new = MAX_BETA
            else:
                beta_new = beta[c]

            # Smoothing
            alpha[c] = smoothing * alpha_new + (1 - smoothing) * alpha[c]
            beta[c]  = smoothing * beta_new  + (1 - smoothing) * beta[c]
            
            # Applica cap finale
            alpha[c] = max(1.0, min(MAX_ALPHA, alpha[c]))
            beta[c]  = max(1.0, min(MAX_BETA,  beta[c]))

        # --- CV diagnostic ---
        weights = [math.exp(r['log_w'])
                   for r in top_results if r['log_w'] > -700]
        if len(weights) > 1:
            cv = (np.std(weights) / np.mean(weights)
                  if np.mean(weights) > 0 else float('inf'))
        else:
            cv = float('inf')

        stats['iterations'].append(iteration)
        stats['n_top'].append(n_top)
        stats['alpha_mean'].append(np.mean(list(alpha.values())))
        stats['beta_mean'].append(np.mean(list(beta.values())))
        stats['cv'].append(cv)

        if verbose:
            a_std = np.std(list(alpha.values()))
            b_std = np.std(list(beta.values()))
            print(f"  → α medio: {np.mean(list(alpha.values())):.2f} (σ={a_std:.2f}), "
                  f"β medio: {np.mean(list(beta.values())):.2f} (σ={b_std:.2f}), "
                  f"CV: {cv:.1%}")

    return alpha, beta, stats


# ---------------------------------------------------------------------------
# Compute one CDF point with CE + IS  (interfaccia invariata)
# ---------------------------------------------------------------------------

def compute_cdf_point_adaptive(
    lambda_, mu_, T, fault_tree,
    alpha_init, beta_init,
    n_is=50000,
    n_mc=100000,
    ce_iterations=2,
    ce_samples=3000,
    verbose=False
):
    from direct_predictor import simulate_CTMC_simple

    comps = list(lambda_.keys())

    alpha_opt, beta_opt, ce_stats = adaptive_is_cross_entropy(
        lambda_, mu_, T, fault_tree,
        alpha_init, beta_init,
        n_iterations=ce_iterations,
        n_samples_per_iter=ce_samples,
        elite_fraction=0.15,
        smoothing=0.6,
        verbose=verbose
    )

    # --- Standard IS estimate ---
    results_is = [
        simulate_CTMC_simple(lambda_, mu_, alpha_opt, beta_opt, T, fault_tree)
        for _ in range(n_is)
    ]

    all_log_w = [r['log_w'] for r in results_is]
    top_indicators = [1.0 if r['top'] else 0.0 for r in results_is]

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

    if n_top_is > 1:
        weights_top = []
        for lw, ind in zip(all_log_w, top_indicators):
            if ind > 0 and lw > -700:
                weights_top.append(math.exp(lw))
        if len(weights_top) > 1:
            cv_is = np.std(weights_top) / np.mean(weights_top)
            std_is = p_is * cv_is / math.sqrt(n_top_is)
        else:
            std_is = p_is
    else:
        std_is = p_is if p_is > 0 else 0.0

    # --- MC reference ---
    alpha_mc = {c: 1.0 for c in comps}
    beta_mc  = {c: 1.0 for c in comps}

    results_mc = [
        simulate_CTMC_simple(lambda_, mu_, alpha_mc, beta_mc, T, fault_tree)
        for _ in range(n_mc)
    ]

    hits = [1.0 if r['top'] else 0.0 for r in results_mc]
    p_mc = np.mean(hits)
    std_mc = np.std(hits) / np.sqrt(n_mc)
    n_top_mc = sum(hits)

    return {
        'p_is': p_is,
        'p_mc': p_mc,
        'std_is': std_is,
        'std_mc': std_mc,
        'n_top_is': int(n_top_is),
        'n_top_mc': int(n_top_mc),
        'alpha_opt': alpha_opt,
        'beta_opt': beta_opt,
        'ce_stats': ce_stats,
    }
