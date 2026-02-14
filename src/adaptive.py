import math

import numpy as np


def adaptive_is_cross_entropy(
    lambda_, mu_, T, fault_tree,
    alpha_init, beta_init,
    n_iterations=5,
    n_samples_per_iter=5000,
    elite_fraction=0.1,
    smoothing=0.7,
    verbose=False
):

    from direct_predictor import simulate_CTMC_simple

    comps = list(lambda_.keys())

    alpha = {c: alpha_init[c] for c in comps}
    beta = {c: beta_init[c] for c in comps}

    n_elite = max(10, int(n_samples_per_iter * elite_fraction))

    stats = {
        'iterations': [],
        'n_top': [],
        'alpha_mean': [],
        'beta_mean': [],
        'cv': []
    }

    if verbose:
        print(f"{'='*60}")
        print(f"ADAPTIVE IS - CROSS ENTROPY")
        print(f"T={T}, samples/iter={n_samples_per_iter}, elite={n_elite}")
        print(f"α₀ medio: {np.mean(list(alpha.values())):.2f}")
        print(f"{'='*60}")

    for iteration in range(n_iterations):

        results = [
            simulate_CTMC_simple(lambda_, mu_, alpha, beta, T, fault_tree)
            for _ in range(n_samples_per_iter)
        ]

        top_results = [r for r in results if r['top']]
        n_top = len(top_results)

        if verbose:
            print(f"Iter {iteration+1}: n_top = {n_top}/{n_samples_per_iter} ({100*n_top/n_samples_per_iter:.1f}%)")

        if n_top < n_elite:
            if verbose:
                print(f"  → Troppo pochi top events, aumento α di 1.3x")
            alpha = {c: min(alpha[c] * 1.3, 200.0) for c in comps}

            stats['iterations'].append(iteration)
            stats['n_top'].append(n_top)
            stats['alpha_mean'].append(np.mean(list(alpha.values())))
            stats['beta_mean'].append(np.mean(list(beta.values())))
            stats['cv'].append(float('inf'))
            continue

        if n_top > n_samples_per_iter * 0.5:
            if verbose:
                print(f"  → Troppi top events, riduco α di 0.8x")
            alpha = {c: max(alpha[c] * 0.8, 1.0) for c in comps}

            stats['iterations'].append(iteration)
            stats['n_top'].append(n_top)
            stats['alpha_mean'].append(np.mean(list(alpha.values())))
            stats['beta_mean'].append(np.mean(list(beta.values())))
            stats['cv'].append(float('inf'))
            continue

        top_with_weights = [(r, r['log_w']) for r in top_results]
        top_with_weights.sort(key=lambda x: x[1], reverse=True)
        elite = [r for r, w in top_with_weights[:n_elite]]

        alpha_new = {}
        beta_new = {}

        for c in comps:

            top_rate = n_top / n_samples_per_iter

            if top_rate < 0.1:

                scale = 1.0 + (0.1 - top_rate) * 2
            elif top_rate > 0.3:

                scale = 1.0 - (top_rate - 0.3) * 0.5
            else:

                scale = 1.0

            alpha_new[c] = alpha[c] * scale
            alpha_new[c] = max(1.0, min(200.0, alpha_new[c]))

            beta_new[c] = beta[c] * (1.0 + (scale - 1.0) * 0.3)
            beta_new[c] = max(1.0, min(10.0, beta_new[c]))

        for c in comps:
            alpha[c] = smoothing * alpha_new[c] + (1 - smoothing) * alpha[c]
            beta[c] = smoothing * beta_new[c] + (1 - smoothing) * beta[c]

        weights = [math.exp(r['log_w']) for r in top_results if r['log_w'] > -700]
        if len(weights) > 1:
            cv = np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else float('inf')
        else:
            cv = float('inf')

        stats['iterations'].append(iteration)
        stats['n_top'].append(n_top)
        stats['alpha_mean'].append(np.mean(list(alpha.values())))
        stats['beta_mean'].append(np.mean(list(beta.values())))
        stats['cv'].append(cv)

        if verbose:
            print(f"  → α medio: {np.mean(list(alpha.values())):.2f}, "
                  f"β medio: {np.mean(list(beta.values())):.2f}, "
                  f"CV: {cv:.1%}")

    return alpha, beta, stats


def compute_cdf_point_adaptive(
    lambda_, mu_, T, fault_tree,
    alpha_init, beta_init,
    n_is=50000,
    n_mc=100000,
    ce_iterations=3,
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

    alpha_mc = {c: 1.0 for c in comps}
    beta_mc = {c: 1.0 for c in comps}

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
        'ce_stats': ce_stats
    }
