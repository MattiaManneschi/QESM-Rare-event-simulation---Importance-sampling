import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Hardware rilevato: {device}")


class AlphaBetaMLP(nn.Module):

    def __init__(self, n_components, config=None):
        super(AlphaBetaMLP, self).__init__()
        self.config = config or OptimizedConfig()

        self.net = nn.Sequential(
            nn.Linear(n_components, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_components * 2) # Output: alpha e beta per ogni componente
        )

        # Inizializzazione conservativa dei pesi per partire da valori centrali
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.net(x)
        alpha_raw, beta_raw = torch.chunk(out, 2, dim=-1)

        # Sigmoid mappa in [0,1], poi scaliamo nel range desiderato
        # Con bias=0 iniziale: sigmoid(0)=0.5 → valore centrale del range
        alpha = torch.sigmoid(alpha_raw) * (self.config.alpha_max - self.config.alpha_min) + self.config.alpha_min
        beta = torch.sigmoid(beta_raw) * (self.config.beta_max - self.config.beta_min) + self.config.beta_min

        return alpha, beta

class DiagnosticLogger:

    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch_data = defaultdict(list)
        self.trajectory_stats = []
        self.weight_stats = []

    def log_trajectory_batch(self, results, epoch):
        top_events = sum(1 for r in results if r['top'])
        log_weights = [r['log_w'] for r in results]

        stats = {
            'epoch': epoch,
            'n_trajectories': len(results),
            'top_events': top_events,
            'top_rate': top_events / len(results) if results else 0,
            'log_w_mean': np.mean(log_weights),
            'log_w_std': np.std(log_weights),
            'log_w_min': np.min(log_weights),
            'log_w_max': np.max(log_weights),
        }
        self.trajectory_stats.append(stats)
        return stats

    def log_weights(self, weights, epoch):
        if len(weights) == 0:
            return None

        weights_array = np.array(weights)
        positive_weights = weights_array[weights_array > 0]

        stats = {
            'epoch': epoch,
            'n_weights': len(weights),
            'n_positive': len(positive_weights),
            'mean': np.mean(weights_array),
            'std': np.std(weights_array),
            'max': np.max(weights_array),
            'min': np.min(weights_array),
        }

        if len(positive_weights) > 0:
            stats['positive_mean'] = np.mean(positive_weights)
            stats['positive_std'] = np.std(positive_weights)

        self.weight_stats.append(stats)
        return stats

    def print_epoch_summary(self, epoch, loss, alpha_dict, beta_dict,
                            n_elite, n_samples, extra_info=None):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}")
        print(f"{'=' * 60}")
        print(f"Loss: {loss:.6f}")
        print(f"Elite samples: {n_elite}/{n_samples}")

        print("\nParametri Alpha:")
        for c, v in alpha_dict.items():
            print(f"  {c}: {v:.4f}")

        print("\nParametri Beta:")
        for c, v in beta_dict.items():
            print(f"  {c}: {v:.4f}")

        if self.trajectory_stats:
            last_traj = self.trajectory_stats[-1]
            print(f"\nTraiettorie:")
            print(f"  Top events rate: {last_traj['top_rate'] * 100:.2f}%")
            print(f"  Log weights: mean={last_traj['log_w_mean']:.4f}, "
                  f"std={last_traj['log_w_std']:.4f}")

        if self.weight_stats:
            last_w = self.weight_stats[-1]
            print(f"\nPesi IS:")
            print(f"  Mean: {last_w['mean']:.6e}")
            print(f"  Std: {last_w['std']:.6e}")
            print(f"  Range: [{last_w['min']:.6e}, {last_w['max']:.6e}]")

class OptimizedConfig:

    PRESETS = {
        "easy": "~10^-2",
        "medium": "~10^-4",
        "hard": "~10^-6",
        "very_hard": "~10^-9",
        "custom": "definito da utente",
    }

    def __init__(self, preset="easy", T=100):
        self.rho = 0.15
        self.alpha_std = 0.3
        self.beta_std = 0.05
        self.max_grad_norm = 0.5
        self.entropy_coef = 0.01
        self.learning_rate = 0.005

        self.T = T
        self.preset = preset

        self._set_difficulty_params(preset)

        self._generate_fault_tree(preset)

    def _set_difficulty_params(self, preset):

        # FACILE: P ~ 10^-2
        if preset == "easy":
            self.alpha_min = 1.0
            self.alpha_max = 3.0
            self.beta_min = 0.3
            self.beta_max = 1.0
            self.n_trajectories = 500
            self.n_samples = 500
            self.epochs = 30

        # MEDIO: P ~ 10^-4
        elif preset == "medium":
            self.alpha_min = 10.0
            self.alpha_max = 30.0
            self.beta_min = 0.3
            self.beta_max = 1.0
            self.n_trajectories = 500
            self.n_samples = 500
            self.epochs = 30

        # DIFFICILE: P ~ 10^-5
        elif preset in ["hard"]:
            self.alpha_min = 2.0
            self.alpha_max = 10.0
            self.beta_min = 0.3
            self.beta_max = 1.0
            self.n_trajectories = 500
            self.n_samples = 500
            self.epochs = 30

        # MOLTO DIFFICILE: P ~ 10^-9
        elif preset in ["very_hard"]:
            self.alpha_min = 2.0
            self.alpha_max = 7.0
            self.beta_min = 0.3
            self.beta_max = 0.7
            self.n_trajectories = 500
            self.n_samples = 500
            self.epochs = 30

    def _generate_fault_tree(self, preset):

        if preset == "easy":
            self._setup_k_out_of_n(5, 2, base_lambda=1e-3)
        elif preset == "medium":
            self._setup_hierarchical(2, 2, "OR", "OR", base_lambda=1e-6)
        elif preset == "hard":
            self._setup_deep_and(4, base_lambda=3e-3)
        elif preset == "very_hard":
            self._setup_deep_and(6, base_lambda=3e-3)
        else:
            raise ValueError(f"Preset '{preset}' non trovato. Disponibili: {list(self.PRESETS.keys())}")


    def _setup_deep_and(self, n_components, base_lambda, base_mu=0.1):
        self.tree_type = f"deep_and_{n_components}"
        components = [f"C{i}" for i in range(n_components)]

        self.lambda_ = {c: base_lambda * (1 + 0.1 * i) for i, c in enumerate(components)}
        self.mu_ = {c: base_mu for c in components}

        def fault_tree(state):
            return all(state[c] == 1 for c in components)

        self.fault_tree = fault_tree

    def _setup_k_out_of_n(self, n, k, base_lambda, base_mu=0.1):
        self.tree_type = f"k{k}_of_{n}"
        components = [f"C{i}" for i in range(n)]

        self.lambda_ = {c: base_lambda for c in components}
        self.mu_ = {c: base_mu for c in components}

        def fault_tree(state):
            failed = sum(1 for c in components if state[c] == 1)
            return failed >= k

        self.fault_tree = fault_tree

    def _setup_hierarchical(self, n_subs, comp_per_sub, sub_logic, top_logic, base_lambda, base_mu=0.1):
        self.tree_type = f"hier_{n_subs}x{comp_per_sub}"

        subsystems = {}
        all_components = []
        for i in range(n_subs):
            prefix = chr(65 + i)
            subsystems[prefix] = [f"{prefix}{j}" for j in range(comp_per_sub)]
            all_components.extend(subsystems[prefix])

        self.lambda_ = {c: base_lambda for c in all_components}
        self.mu_ = {c: base_mu for c in all_components}

        def fault_tree(state):
            sub_states = []
            for prefix, comps in subsystems.items():
                if sub_logic == "AND":
                    sub_failed = all(state[c] == 1 for c in comps)
                else:
                    sub_failed = any(state[c] == 1 for c in comps)
                sub_states.append(sub_failed)

            if top_logic == "AND":
                return all(sub_states)
            else:
                return any(sub_states)

        self.fault_tree = fault_tree

def simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree):
    """
        Simula una traiettoria CTMC con Importance Sampling.

        Il likelihood ratio (log_w) tiene traccia della differenza tra:
        - Distribuzione originale (lambda_, mu_)
        - Distribuzione biased (lambda_*alpha, mu_*beta)

        Formula del likelihood ratio per CTMC:
        - Holding time: log_w += (R_is - R_orig) * dt
        - Scelta transizione: log_w += log(P_orig / P_is)

        Alla fine: weight = exp(log_w) corregge il bias introdotto.
        """

    t = 0.0
    state = {i: 0 for i in lambda_} # Tutti i componenti partono funzionanti (0)
    log_w = 0.0
    top_event_hit = False
    n_transitions = 0

    while t < T:
        # Calcola i tassi originali e biased per ogni componente
        rates_orig = {}
        rates_is = {}

        for i in lambda_:
            if state[i] == 0: # Componente funzionante → può guastarsi
                rates_orig[i] = lambda_[i]
                rates_is[i] = lambda_[i] * alpha[i] # Accelera guasti
            else: # Componente guasto → può essere riparato
                rates_orig[i] = mu_[i]
                rates_is[i] = mu_[i] * beta[i]  # Rallenta riparazioni

        R_orig = sum(rates_orig.values()) # Tasso totale originale
        R_is = sum(rates_is.values()) # Tasso totale IS

        if R_is <= 0:
            break

        # Tempo di holding: esponenziale con rate R_is
        dt = random.expovariate(R_is)

        if t + dt > T:
            # Correzione per il tempo residuo fino a T
            log_w += (R_is - R_orig) * (T - t)
            break

        # Aggiorna likelihood ratio per il tempo di holding
        # P(dt|orig) / P(dt|IS) = (R_orig * exp(-R_orig*dt)) / (R_is * exp(-R_is*dt))
        # log di questo = log(R_orig/R_is) + (R_is - R_orig)*dt
        # Ma la parte log(R_orig/R_is) si compensa con la scelta della transizione
        log_w += (R_is - R_orig) * dt
        t += dt
        n_transitions += 1

        # Scelta del componente che transisce (proporzionale ai tassi IS)
        comps = list(lambda_.keys())
        p_is = [rates_is[c] / R_is for c in comps]
        chosen_comp = random.choices(comps, weights=p_is)[0]

        rate_orig_chosen = rates_orig[chosen_comp]
        rate_is_chosen = rates_is[chosen_comp]

        # Correzione per la scelta della transizione
        # P(scegliere c | orig) / P(scegliere c | IS)
        log_w += math.log(rate_orig_chosen / rate_is_chosen)

        # Esegui la transizione (toggle dello stato)
        state[chosen_comp] = 1 - state[chosen_comp]

        if fault_tree(state):
            top_event_hit = True

    result = {
        "top": top_event_hit,
        "log_w": log_w,
        "n_transitions": n_transitions,
        "final_state": dict(state)
    }

    return result

def train_mlp_cross_entropy(config):
    """
        Simula una traiettoria CTMC con Importance Sampling.

        Il likelihood ratio (log_w) tiene traccia della differenza tra:
        - Distribuzione originale (lambda_, mu_)
        - Distribuzione biased (lambda_*alpha, mu_*beta)

        Formula del likelihood ratio per CTMC:
        - Holding time: log_w += (R_is - R_orig) * dt
        - Scelta transizione: log_w += log(P_orig / P_is)

        Alla fine: weight = exp(log_w) corregge il bias introdotto.
        """

    lambda_ = config.lambda_
    mu_ = config.mu_
    T = config.T

    n = len(lambda_)
    comps = list(lambda_.keys())

    model = AlphaBetaMLP(n, config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    logger = DiagnosticLogger()

    alpha_hist = {c: [] for c in comps}
    beta_hist = {c: [] for c in comps}
    loss_hist = []
    prob_estimates = []

    # Input fittizio per la rete (media degli one-hot vectors)
    input_tensor = torch.eye(n).mean(dim=0).unsqueeze(0).to(device)

    print(f"Configurazione:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Trajectories per sample: {config.n_trajectories}")
    print(f"  Samples per epoch: {config.n_samples}")
    print(f"  Elite fraction (rho): {config.rho}")
    print("="*60 + "\n")

    for epoch in range(config.epochs):
        # La rete predice le medie delle distribuzioni
        alpha_mu, beta_mu = model(input_tensor)

        samples_data = []
        log_performances = []
        all_weights_epoch = []

        # Campiona n_samples configurazioni di parametri
        for s in range(config.n_samples):
            # Crea distribuzioni normali centrate sui valori predetti
            dist_a = torch.distributions.Normal(alpha_mu, config.alpha_std)
            dist_b = torch.distributions.Normal(beta_mu, config.beta_std)

            # Campiona e clampa nei range validi
            a_sampled = dist_a.sample()
            b_sampled = dist_b.sample()

            a_dict = {c: torch.clamp(a_sampled[0, i],
                                      config.alpha_min,
                                      config.alpha_max).item()
                      for i, c in enumerate(comps)}
            b_dict = {c: torch.clamp(b_sampled[0, i],
                                      config.beta_min,
                                      config.beta_max).item()
                      for i, c in enumerate(comps)}

            # Simula traiettorie con questi parametri
            trajs = [simulate_CTMC(lambda_, mu_, a_dict, b_dict, T, config.fault_tree)
                     for _ in range(config.n_trajectories)]

            # Raccogli i log-weights delle traiettorie che hanno raggiunto il top event
            traj_logs = [tr["log_w"] for tr in trajs if tr["top"]]

            weights = [math.exp(tr["log_w"]) if tr["top"] else 0.0 for tr in trajs]
            all_weights_epoch.extend(weights)

            # Calcola la performance del campione (log-sum-exp per stabilità numerica)
            # Performance = media dei pesi IS = stima della probabilità
            if traj_logs:
                m_log = max(traj_logs)
                lse = m_log + math.log(sum(math.exp(lw - m_log) for lw in traj_logs))
                log_perf = lse - math.log(config.n_trajectories)
                log_performances.append(log_perf)
            else:
                log_performances.append(-1e10) # Penalità se nessun top event

            samples_data.append({
                'a_sampled': a_sampled,
                'b_sampled': b_sampled,
                'dist_a': dist_a,
                'dist_b': dist_b,
                'a_dict': a_dict,
                'b_dict': b_dict,
                'n_top': len(traj_logs)
            })

        logger.log_weights(all_weights_epoch, epoch)

        # Cross-Entropy Method: seleziona il top rho% come "elite"
        sorted_idx = sorted(range(len(log_performances)),
                            key=lambda i: log_performances[i], reverse=True)
        n_elite = max(1, int(config.rho * config.n_samples))
        elite_indices = [i for i in sorted_idx[:n_elite]
                         if log_performances[i] > -1e9]

        if elite_indices:
            elite_logs = torch.tensor([log_performances[i] for i in elite_indices],
                                      device=device)

            # Pesi softmax per dare più importanza agli elite migliori
            with torch.no_grad():
                shifted_logs = elite_logs - torch.max(elite_logs)
                weights = torch.softmax(shifted_logs, dim=0)

            # Policy Gradient Loss: aumenta la probabilità degli elite
            # loss = -Σ weight_i * log π(params_i | θ)
            policy_loss = 0.0
            entropy_bonus = 0.0

            for i, idx in enumerate(elite_indices):
                sample = samples_data[idx]
                # log π(alpha, beta | θ) = log_prob della distribuzione normale
                log_p_a = sample['dist_a'].log_prob(sample['a_sampled']).sum()
                log_p_b = sample['dist_b'].log_prob(sample['b_sampled']).sum()
                policy_loss -= weights[i] * (log_p_a + log_p_b)

                # Entropy bonus per incoraggiare l'esplorazione
                entropy_bonus += sample['dist_a'].entropy().sum()
                entropy_bonus += sample['dist_b'].entropy().sum()

            entropy_bonus = entropy_bonus / len(elite_indices)
            total_loss = policy_loss - config.entropy_coef * entropy_bonus

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            max_norm=config.max_grad_norm)
            optimizer.step()

            loss_val = total_loss.item()
            loss_hist.append(loss_val)
            scheduler.step(loss_val)

            current_prob = np.mean([w for w in all_weights_epoch if w > 0]) if any(w > 0 for w in all_weights_epoch) else 0
            prob_estimates.append(current_prob)

            if epoch % 5 == 0:
                a_dict = {c: alpha_mu[0, i].item() for i, c in enumerate(comps)}
                b_dict = {c: beta_mu[0, i].item() for i, c in enumerate(comps)}

                logger.print_epoch_summary(
                    epoch, loss_val, a_dict, b_dict,
                    len(elite_indices), config.n_samples,
                    extra_info={
                        'Prob estimate': f"{current_prob:.6e}",
                        'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
                    }
                )
        else:
            loss_hist.append(0.0)
            prob_estimates.append(0.0)
            print(f"Epoch {epoch:3d} | Nessun elite trovato - aumentare alpha?")

        for i, c in enumerate(comps):
            alpha_hist[c].append(alpha_mu[0, i].item())
            beta_hist[c].append(beta_mu[0, i].item())

    return model, loss_hist, alpha_hist, beta_hist, prob_estimates, logger

def evaluate_model(model, config, N_eval=10000):
    print("\n" + "=" * 60)
    print("VALUTAZIONE FINALE")
    print("=" * 60)

    lambda_ = config.lambda_
    mu_ = config.mu_
    T = config.T

    comps = list(lambda_.keys())
    input_tensor = torch.eye(len(lambda_)).mean(dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        alpha_final, beta_final = model(input_tensor)

    a_f = {c: alpha_final[0, i].item() for i, c in enumerate(comps)}
    b_f = {c: beta_final[0, i].item() for i, c in enumerate(comps)}

    print("\nParametri ottimizzati:")
    for c in comps:
        print(f"  {c}: alpha={a_f[c]:.4f}, beta={b_f[c]:.4f}")

    print(f"\nSimulando {N_eval} traiettorie...")
    results = [simulate_CTMC(lambda_, mu_, a_f, b_f, T, config.fault_tree)
               for _ in range(N_eval)]

    n_top = sum(1 for r in results if r['top'])
    weights = np.array([math.exp(r['log_w']) if r['top'] else 0.0 for r in results])

    p_is = np.mean(weights)
    var_is = np.var(weights) / N_eval
    std_is = np.sqrt(var_is)

    cv = std_is / p_is if p_is > 0 else float('inf')

    ci_low = max(0, p_is - 1.96 * std_is)
    ci_high = p_is + 1.96 * std_is

    print(f"\nRisultati:")
    print(f"  Top events: {n_top}/{N_eval} ({100 * n_top / N_eval:.2f}%)")
    print(f"  Stima probabilità (IS): {p_is:.6e}")
    print(f"  Errore standard: {std_is:.6e}")
    print(f"  Coefficiente di variazione: {cv:.4f}")
    print(f"  IC 95%: [{ci_low:.6e}, {ci_high:.6e}]")

    print("\n=== CONFRONTO IS VS MC ===")
    naive_results = [simulate_CTMC(lambda_, mu_,
                                             {c: 1.0 for c in comps},
                                             {c: 1.0 for c in comps}, T, config.fault_tree)
                     for _ in range(N_eval)]
    n_top_naive = sum(1 for r in naive_results if r['top'])
    p_naive = n_top_naive / N_eval
    var_naive = p_naive * (1 - p_naive) / N_eval
    std_naive = np.sqrt(var_naive)
    cv_naive = std_naive / p_naive if p_naive > 0 else float('inf')

    print(f"MC Naive:")
    print(f"  Stima: {p_naive:.6e}")
    print(f"  Std: {std_naive:.6e}")
    print(f"  CV: {cv_naive:.4f}")

    print(f"\nImportance Sampling:")
    print(f"  Stima: {p_is:.6e}")
    print(f"  Std: {std_is:.6e}")
    print(f"  CV: {cv:.4f}")

    var_reduction = var_naive / var_is if var_is > 0 else 0
    print(f"\nVariance Reduction Factor: {var_reduction:.2f}x")

    equivalent_mc_samples = N_eval * var_reduction
    print(f"Simulazioni MC equivalenti: {equivalent_mc_samples:.0f}")

    eval_results = {
        'p_is': p_is,
        'std_is': std_is,
        'cv': cv,
        'ci': (ci_low, ci_high),
        'alpha': a_f,
        'beta': b_f,
        'n_top': n_top,
        'top_rate': n_top / N_eval
    }

    save_results_to_file(
        tree_type=tree_type,
        eval_results=eval_results,
        p_naive=p_naive,
        n_top_naive=n_top_naive,
        N_eval=N_eval,
        std_naive=std_naive,
        cv_naive=cv_naive,
        var_reduction=var_reduction,
        equivalent_mc_samples=equivalent_mc_samples
    )

def plot_results_extended(loss_hist, alpha_hist, beta_hist, prob_estimates, tree_type):
    output_dir = f"results/{tree_type}"
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(len(loss_hist))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_hist, color='tab:red', linewidth=1.5, alpha=0.7)
    plt.title(f"Andamento Loss - {tree_type}", fontsize=12)
    plt.xlabel("Epoca")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    valid_probs = [p for p in prob_estimates if p > 0]
    valid_epochs = [i for i, p in enumerate(prob_estimates) if p > 0]
    if valid_probs:
        plt.semilogy(valid_epochs, valid_probs, color='tab:blue', linewidth=1.5)
        plt.title(f"Stima Probabilità durante Training", fontsize=12)
        plt.xlabel("Epoca")
        plt.ylabel("Probabilità (scala log)")
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/andamento_loss.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for comp, values in alpha_hist.items():
        axes[0, 0].plot(epochs, values, label=f"α_{comp}", linewidth=1.5)
    axes[0, 0].set_title("Evoluzione Alpha (accelerazione guasti)")
    axes[0, 0].set_ylabel("Valore Alpha")
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    for comp, values in beta_hist.items():
        axes[0, 1].plot(epochs, values, label=f"β_{comp}", linestyle='--', linewidth=1.5)
    axes[0, 1].set_title("Evoluzione Beta (rallentamento riparazioni)")
    axes[0, 1].set_ylabel("Valore Beta")
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    final_alphas = {c: vals[-1] for c, vals in alpha_hist.items()}
    axes[1, 0].bar(final_alphas.keys(), final_alphas.values(), color='steelblue')
    axes[1, 0].set_title("Alpha finali")
    axes[1, 0].set_ylabel("Valore")

    final_betas = {c: vals[-1] for c, vals in beta_hist.items()}
    axes[1, 1].bar(final_betas.keys(), final_betas.values(), color='coral')
    axes[1, 1].set_title("Beta finali")
    axes[1, 1].set_ylabel("Valore")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/andamento_parametri.png", dpi=300)
    plt.close()

def save_results_to_file(tree_type, eval_results, p_naive, n_top_naive, N_eval, std_naive, cv_naive, var_reduction, equivalent_mc_samples):
    output_dir = f"results/{tree_type}"
    os.makedirs(output_dir, exist_ok=True)

    filepath = f"{output_dir}/valutazione_{tree_type}.txt"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"VALUTAZIONE FINALE - {tree_type.upper()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Data: {timestamp}\n")
        f.write(f"Simulazioni: {N_eval}\n")
        f.write("\n")

        f.write("PARAMETRI OTTIMIZZATI:\n")
        f.write("-" * 40 + "\n")
        for c in eval_results['alpha']:
            f.write(f"  {c}: alpha={eval_results['alpha'][c]:.4f}, beta={eval_results['beta'][c]:.4f}\n")
        f.write("\n")

        f.write("RISULTATI IMPORTANCE SAMPLING:\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"  Top events: {eval_results.get('n_top', 'N/A')}/{N_eval} ({eval_results.get('top_rate', 0) * 100:.2f}%)\n")
        f.write(f"  Stima probabilità: {eval_results['p_is']:.6e}\n")
        f.write(f"  Errore standard: {eval_results['std_is']:.6e}\n")
        f.write(f"  Coefficiente di variazione: {eval_results['cv']:.4f}\n")
        f.write(f"  IC 95%: [{eval_results['ci'][0]:.6e}, {eval_results['ci'][1]:.6e}]\n")
        f.write("\n")

        f.write("RISULTATI MC NAIVE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Top events: {n_top_naive}/{N_eval} ({100 * n_top_naive / N_eval:.2f}%)\n")
        f.write(f"  Stima probabilità: {p_naive:.6e}\n")
        f.write(f"  Errore standard: {std_naive:.6e}\n")
        f.write(f"  Coefficiente di variazione: {cv_naive:.4f}\n")
        f.write("\n")

        f.write("CONFRONTO IS VS MC:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Variance Reduction Factor: {var_reduction:.2f}x\n")
        f.write(f"  Simulazioni MC equivalenti: {equivalent_mc_samples:.0f}\n")
        f.write(f"  Differenza stime: {abs(eval_results['p_is'] - p_naive):.6e}\n")
        f.write(f"  Rapporto stime (IS/MC): {eval_results['p_is'] / p_naive:.4f}\n")
        f.write("\n")

        f.write("VALUTAZIONE:\n")
        f.write("-" * 40 + "\n")
        if var_reduction > 1:
            f.write(f"  ✓ IS è {var_reduction:.1f}x più efficiente di MC naive\n")
        else:
            f.write(f"  ✗ IS non migliora rispetto a MC naive\n")

        ratio = eval_results['p_is'] / p_naive
        if 0.9 <= ratio <= 1.1:
            f.write("  ✓ Stime concordanti (differenza < 10%)\n")
        else:
            f.write(f"  ⚠ Stime discordanti (rapporto: {ratio:.2f})\n")

        f.write("=" * 60 + "\n")


if __name__ == "__main__":

    config = OptimizedConfig(preset = "very_hard")

    lambda_ = config.lambda_
    mu_ = config.mu_
    tree_type = config.tree_type
    faul_tree = config.fault_tree
    T = config.T


    print("=" * 60)
    print("IMPORTANCE SAMPLING PER FAULT TREE CON CTMC")
    print("=" * 60)
    print(f"\nTipo fault tree: {tree_type}")
    print(f"Tempo simulazione T: {T}")
    print("\nTassi di guasto (lambda):")
    for c, v in lambda_.items():
        print(f"  {c}: {v:.2e}")
    print("\nTassi di riparazione (mu):")
    for c, v in mu_.items():
        print(f"  {c}: {v:.2e}")

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    trained_model, loss_hist, alpha_hist, beta_hist, prob_estimates, logger = train_mlp_cross_entropy(config)

    evaluate_model(trained_model, config)

    plot_results_extended(loss_hist, alpha_hist, beta_hist, prob_estimates, tree_type)

