from datetime import datetime
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlphaBetaMLP(nn.Module):
    """
    Rete neurale che predice i parametri di biasing (alpha, beta) per ogni componente.
    """

    def __init__(self, n_components, config):
        super(AlphaBetaMLP, self).__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(n_components, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_components * 2)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.net(x)
        alpha_raw, beta_raw = torch.chunk(out, 2, dim=-1)

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

    def log_weights(self, weights, epoch):
        if len(weights) == 0:
            return None
        weights_array = np.array(weights)
        stats = {
            'epoch': epoch,
            'mean': np.mean(weights_array),
            'std': np.std(weights_array),
            'max': np.max(weights_array),
        }
        self.weight_stats.append(stats)
        return stats

    def print_epoch_summary(self, epoch, loss, alpha_dict, beta_dict, n_elite, n_samples):
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Elite: {n_elite}/{n_samples}")

class ExternalConfig:
    """Configurazione per il training IS con range esterni."""

    def __init__(self, lambda_, mu_, fault_tree_logic, ranges, T=100):
        self.lambda_ = lambda_
        self.mu_ = mu_
        self.fault_tree = fault_tree_logic
        self.T = T

        # Parametri ottimizzazione
        self.rho = 0.15
        self.alpha_std = 0.3
        self.beta_std = 0.05
        self.max_grad_norm = 0.5
        self.entropy_coef = 0.01
        self.learning_rate = 0.005

        # Range predetti dal GNN
        self.alpha_min, self.alpha_max = ranges['alpha']
        self.beta_min, self.beta_max = ranges['beta']

        # Parametri simulazione
        self.n_trajectories = 500
        self.n_samples = 500
        self.epochs = 30

def simulate_CTMC(lambda_, mu_, alpha, beta, T, fault_tree):
    t = 0.0
    state = {i: 0 for i in lambda_} # Tutti i componenti partono funzionanti (0)
    log_w = 0.0
    top_event_hit = False
    n_transitions = 0

    while t < T:
        # Calcola i tassi di transizione originali e biased per ogni componente
        rates_orig = {}
        rates_is = {}

        for i in lambda_:
            if state[i] == 0: # Componente funzionante â†’ puÃ² guastarsi
                rates_orig[i] = lambda_[i]
                rates_is[i] = lambda_[i] * alpha[i] # Accelera guasti
            else: # Componente guasto â†’ puÃ² essere riparato
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

    print("=" * 60)
    print("TRAINING MLP - IS VS MC")
    print("=" * 60)

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

            # Calcola la performance del campione (log-sum-exp per stabilita'  numerica)
            # Performance = media dei pesi IS = stima della probabilitÃ
            if traj_logs:
                m_log = max(traj_logs)
                lse = m_log + math.log(sum(math.exp(lw - m_log) for lw in traj_logs))
                log_perf = lse - math.log(config.n_trajectories)
                log_performances.append(log_perf)
            else:
                log_performances.append(-1e10) # Penalita'  se nessun top event

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

            # Pesi softmax per dare piÃ¹ importanza agli elite migliori
            with torch.no_grad():
                shifted_logs = elite_logs - torch.max(elite_logs)
                weights = torch.softmax(shifted_logs, dim=0)

            # Policy Gradient Loss: aumenta la probabilitÃ  degli elite
            # loss = -Î£ weight_i * log Ï€(params_i | Î¸)
            policy_loss = 0.0
            entropy_bonus = 0.0

            for i, idx in enumerate(elite_indices):
                sample = samples_data[idx]
                # log Ï€(alpha, beta | Î¸) = log_prob della distribuzione normale
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
                    len(elite_indices), config.n_samples
                )
        else:
            loss_hist.append(0.0)
            prob_estimates.append(0.0)
            print(f"Epoch {epoch:3d} | Nessun elite trovato - aumentare alpha?")

        for i, c in enumerate(comps):
            alpha_hist[c].append(alpha_mu[0, i].item())
            beta_hist[c].append(beta_mu[0, i].item())

    return model

def evaluate_model(model, config, N_is, N_mc, topology_name):
    """Valuta IS vs MC con metriche dettagliate."""

    lines = []  # Buffer per salvataggio TXT

    def log(msg):
        print(msg)
        lines.append(str(msg))

    log("\n" + "=" * 60)
    log("RISULTATI")
    log("=" * 60)

    # Topologia
    lambda_ = config.lambda_
    mu_ = config.mu_
    comps = list(lambda_.keys())

    log(f"\n-> Topologia: {topology_name}")
    log(f"-> Lambda: {[lambda_[c] for c in comps]}")
    log(f"-> Mu:     {[mu_[c] for c in comps]}")
    log(f"-> T:      {config.T}")

    log(f"\n-> Range suggeriti:")
    log(f"   Alpha: [{config.alpha_min:.2f}, {config.alpha_max:.2f}]")
    log(f"   Beta:  [{config.beta_min:.2f}, {config.beta_max:.2f}]")

    input_tensor = torch.eye(len(lambda_)).mean(dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        alpha_final, beta_final = model(input_tensor)

    a_f = {c: alpha_final[0, i].item() for i, c in enumerate(comps)}
    b_f = {c: beta_final[0, i].item() for i, c in enumerate(comps)}

    log(f"\nParametri ottimizzati:")
    for c in comps:
        log(f"  {c}: α={a_f[c]:.3f}, β={b_f[c]:.3f}")

    # IS
    log(f"\nEsecuzione IS ({N_is} simulazioni)...")
    results_is = [simulate_CTMC(lambda_, mu_, a_f, b_f, config.T, config.fault_tree)
                  for _ in range(N_is)]

    weights_is = [math.exp(r['log_w']) if r['top'] else 0.0 for r in results_is]
    n_top_is = sum(1 for r in results_is if r['top'])
    p_is = np.mean(weights_is)
    var_is = np.var(weights_is)
    std_is = np.std(weights_is) / np.sqrt(N_is)
    cv_is = std_is / p_is if p_is > 0 else float('inf')

    # MC
    log(f"Esecuzione MC ({N_mc} simulazioni)...")
    a_mc = {c: 1.0 for c in comps}
    b_mc = {c: 1.0 for c in comps}
    results_mc = [simulate_CTMC(lambda_, mu_, a_mc, b_mc, config.T, config.fault_tree)
                  for _ in range(N_mc)]

    hits_mc = [1.0 if r['top'] else 0.0 for r in results_mc]
    n_top_mc = int(sum(hits_mc))
    p_mc = np.mean(hits_mc)
    var_mc = np.var(hits_mc)
    std_mc = np.sqrt(p_mc * (1 - p_mc) / N_mc) if p_mc > 0 else 0
    cv_mc = std_mc / p_mc if p_mc > 0 else float('inf')

    # Calcola efficienza
    if var_is > 0 and var_mc > 0:
        efficiency_gain = var_mc / var_is
    elif var_is == 0 and n_top_is > 0:
        efficiency_gain = float('inf')
    else:
        efficiency_gain = 1.0

    rel_error = abs(p_is - p_mc) / p_mc * 100 if p_mc > 0 else float('inf')

    # Risultati
    log("\n" + "=" * 60)
    log("RISULTATI")
    log("=" * 60)

    log(f"\n{'IMPORTANCE SAMPLING':=^40}")
    log(f"  Campioni totali:    {N_is}")
    log(f"  Top events:         {n_top_is} ({100 * n_top_is / N_is:.2f}%)")
    log(f"  Probabilità:        {p_is:.6e}")
    log(f"  Varianza:           {var_is:.6e}")
    log(f"  Errore standard:    {std_is:.6e}")
    log(f"  CV:                 {cv_is:.4f}")

    log(f"\n{'MONTE CARLO':=^40}")
    log(f"  Campioni totali:    {N_mc}")
    log(f"  Top events:         {n_top_mc} ({100 * n_top_mc / N_mc:.2f}%)")
    log(f"  Probabilità:        {p_mc:.6e}")
    log(f"  Varianza:           {var_mc:.6e}")
    log(f"  Errore standard:    {std_mc:.6e}")
    log(f"  CV:                 {cv_mc:.4f}")

    log(f"\n{'CONFRONTO':=^40}")
    if efficiency_gain != float('inf'):
        log(f"  Guadagno efficienza: {efficiency_gain:.2f}x")
    else:
        log(f"  Guadagno efficienza: ∞ (MC non ha trovato eventi)")

    if p_mc > 0:
        log(f"  Errore relativo:     {rel_error:.2f}%")

    log("=" * 60)

    # Salva risultati TXT
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_filename = f'results/evaluation_{timestamp}.txt'
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Risultati salvati in '{txt_filename}'")

    return p_is, var_is, p_mc, var_mc

def run_overall_tester(ft, fault_tree_logic, ranges_dict, N_is, N_mc,  topology_name, T=100):
    lambda_dict, mu_dict = ft.get_lambda_mu()
    config = ExternalConfig(lambda_dict, mu_dict, fault_tree_logic, ranges_dict, T)

    # Training
    model = train_mlp_cross_entropy(config)

    # Usa valori default se non forniti
    if N_is is None:
        N_is = 10000
    if N_mc is None:
        N_mc = 10000

    # Valutazione (stampa direttamente i risultati)
    p_is, var_is, p_mc, var_mc = evaluate_model(model, config, N_is, N_mc, topology_name)

    return p_is, var_is, p_mc, var_mc