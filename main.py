import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Hardware rilevato: {device}")

#TODO FARE ALTRI TEST
#TODO DIVIDERE LA GENERAZIONE DEI RISULTATI IN BASE ALLA TIPOLOGIA DI ALBERO

def fault_tree(state):
    #return (state["A"] == 1 and state["B"] == 1) or (state["C"] == 1)

    #a, b, c = state["A"], state["B"], state["C"]
    #return (a == 1 and b == 1) or (a == 1 and c == 1) or (b == 1 and c == 1)

    #sub_system = (state["A"] == 1 or state["B"] == 1) and (state["C"] == 1 or state["D"] == 1)
    #critical_node = (state["E"] == 1)
    #return sub_system or critical_node

    power_fail = (state["A"] == 1)
    cooling_fail = (state["B"] == 1)
    server_cluster_fail = (state["C"] == 1 and state["D"] == 1 and state["E"] == 1)
    return power_fail or cooling_fail or server_cluster_fail

def simulate_CTMC(lambda_, mu_, alpha, beta, T):
    t = 0.0
    state = {i: 0 for i in lambda_}

    NF = {i: 0 for i in lambda_}
    NR = {i: 0 for i in lambda_}
    Tup = {i: 0.0 for i in lambda_}
    Tdn = {i: 0.0 for i in lambda_}

    log_w = 0.0

    while t < T:
        rates = {}
        for i in state:
            rates[i] = (
                alpha[i] * lambda_[i]
                if state[i] == 0
                else beta[i] * mu_[i]
            )

        R = sum(rates.values())
        if R == 0:
            break

        dt = random.expovariate(R)
        dt = min(dt, T - t)

        for i in state:
            if state[i] == 0:
                Tup[i] += dt
            else:
                Tdn[i] += dt

        t += dt
        if t >= T:
            break

        r = random.uniform(0, R)
        acc = 0.0
        comp = None
        for i, rate in rates.items():
            acc += rate
            if r <= acc:
                comp = i
                break

        if comp is None:
            comp = list(rates.keys())[-1]

        if state[comp] == 0:
            state[comp] = 1
            NF[comp] += 1
            if alpha[comp] > 1e-10:
                log_w += math.log(1.0 / alpha[comp])
            else:
                log_w += math.log(1.0 / 1e-10)
        else:
            state[comp] = 0
            NR[comp] += 1
            if beta[comp] > 1e-10:
                log_w += math.log(1.0 / beta[comp])
            else:
                log_w += math.log(1.0 / 1e-10)

    for i in state:
        log_w += (lambda_[i] - alpha[i] * lambda_[i]) * Tup[i]
        log_w += (mu_[i] - beta[i] * mu_[i]) * Tdn[i]

    log_w = max(min(log_w, 700), -700)

    return {
        "NF": NF,
        "NR": NR,
        "Tup": Tup,
        "Tdn": Tdn,
        "log_w": log_w,
        "top": fault_tree(state)
    }

def extract_features(trajs, lambda_, mu_):
    feats = []
    N = len(trajs)
    for i in lambda_:
        feats.extend([
            math.log10(lambda_[i]),
            math.log10(mu_[i]),
            sum(tr["NF"][i] for tr in trajs) / N,
            sum(tr["NR"][i] for tr in trajs) / N,
            sum(tr["Tup"][i] for tr in trajs) / (N * 1000.0)
        ])
    return torch.tensor(feats, dtype=torch.float32)

class AlphaBetaMLP(nn.Module):
    def __init__(self, input_dim, n_comp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_comp)
        )

    def forward(self, x):
        out = self.net(x)
        a, b = torch.chunk(out, 2, dim=-1)
        return a, b

def sample_parameters(alpha_raw, beta_raw, noise_std):
    n = len(alpha_raw)

    c_n_alpha = torch.randn(n, device=alpha_raw.device)
    c_n_beta = torch.randn(n, device=beta_raw.device)

    alpha_noisy = alpha_raw + noise_std * c_n_alpha
    beta_noisy = beta_raw + noise_std * c_n_beta

    alpha = torch.clamp(torch.exp(alpha_noisy), min=0.5, max=3.0)
    beta = torch.clamp(torch.sigmoid(beta_noisy), min=0.1, max=0.9)

    return alpha, beta, c_n_alpha, c_n_beta

def cross_entropy_loss_ML(samples_data, rho):
    log_performances = []
    for sample in samples_data:
        traj_logs = [tr["log_w"] for tr in sample['trajs'] if tr["top"]]

        if traj_logs:
            max_log = max(traj_logs)
            lse = max_log + math.log(sum(math.exp(lw - max_log) for lw in traj_logs))
            log_perf = lse - math.log(len(sample['trajs']))
            log_performances.append(log_perf)
        else:
            log_performances.append(-float('inf'))

    sorted_indices = sorted(range(len(log_performances)),
                            key=lambda i: log_performances[i], reverse=True)

    elite_idx = [i for i in sorted_indices[:max(1, int(rho * len(log_performances)))]
                 if log_performances[i] > -float('inf')]

    elite_mean = np.mean([math.exp(min(log_performances[i], 20)) for i in elite_idx]) if elite_idx else 0
    return elite_idx, elite_mean, log_performances

def train_mlp_cross_entropy(lambda_, mu_, T, epochs, N_trajs, N_samples, rho, noise_std):
    comps = list(lambda_.keys())
    n = len(comps)

    dummy_x = extract_features(
        [simulate_CTMC(lambda_, mu_, {i: 1.0 for i in comps}, {i: 1.0 for i in comps}, T)],
        lambda_, mu_
    )
    input_dim = dummy_x.shape[0]

    model = AlphaBetaMLP(input_dim, n).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)

    alpha_init = {i: 1.0 for i in comps}
    beta_init = {i: 1.0 for i in comps}

    loss_history = []
    elite_history = []
    alpha_hist = {i: [] for i in comps}
    beta_hist = {i: [] for i in comps}

    best_elite_mean = -float('inf')
    print(f"--- AVVIO TRAINING STABILE (Log-Domain) ---")

    for ep in range(epochs):
        with torch.no_grad():
            trajs_init = [simulate_CTMC(lambda_, mu_, alpha_init, beta_init, T) for _ in range(N_trajs)]
            x = extract_features(trajs_init, lambda_, mu_).to(device)

        current_noise = noise_std * (0.995 ** ep)

        alpha_raw, beta_raw = model(x)
        samples_data = []
        log_performances = []

        for s in range(N_samples):
            a_s, b_s, c_n_a, c_n_b = sample_parameters(alpha_raw, beta_raw, current_noise)
            a_dict = {c_id: a_s[k].item() for k, c_id in enumerate(comps)}
            b_dict = {c_id: b_s[k].item() for k, c_id in enumerate(comps)}

            trajs_s = [simulate_CTMC(lambda_, mu_, a_dict, b_dict, T) for _ in range(N_trajs)]
            traj_logs = [tr["log_w"] for tr in trajs_s if tr["top"]]

            if traj_logs:
                m_log = max(traj_logs)
                lse = m_log + math.log(sum(math.exp(lw - m_log) for lw in traj_logs))
                log_perf = lse - math.log(N_trajs)
                log_performances.append(log_perf)
            else:
                log_performances.append(-1e10)

            samples_data.append({'alpha': a_s, 'beta': b_s, 'c_n_a': c_n_a, 'c_n_b': c_n_b})

        sorted_idx = sorted(range(len(log_performances)), key=lambda i: log_performances[i], reverse=True)
        n_elite = max(1, int(rho * N_samples))
        elite_indices = [i for i in sorted_idx[:n_elite] if log_performances[i] > -1e9]

        if elite_indices:
            elite_logs = torch.tensor([log_performances[i] for i in elite_indices], device=device)
            max_log = torch.max(elite_logs)
            lme = max_log + torch.log(torch.mean(torch.exp(elite_logs - max_log)))
            current_elite_log_mean = lme.item()

            with torch.no_grad():
                shifted_logs = elite_logs - torch.max(elite_logs)
                weights = torch.softmax(shifted_logs, dim=0)

            policy_loss = 0.0
            for i, idx in enumerate(elite_indices):
                sample = samples_data[idx]
                log_p = -0.5 * torch.sum(((sample['alpha'] - alpha_raw) / current_noise) ** 2) + \
                        -0.5 * torch.sum(((sample['beta'] - beta_raw) / current_noise) ** 2)
                policy_loss -= weights[i] * log_p / (len(comps) * 2)

            opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()

            if current_elite_log_mean > best_elite_mean:
                best_elite_mean = current_elite_log_mean
                torch.save(model.state_dict(), "best_model.pth")
                print(f"--> Nuovo Record Epoca {ep}: Elite Mean {best_elite_mean:.2e}. Modello salvato.")

            loss_history.append(policy_loss.item())
            elite_history.append(current_elite_log_mean)

            for k, i in enumerate(comps):
                a_val = torch.exp(alpha_raw[k]).clamp(0.5, 4.0).item()
                b_val = torch.sigmoid(beta_raw[k]).clamp(0.2, 0.8).item()
                alpha_init[i], beta_init[i] = a_val, b_val
                alpha_hist[i].append(a_val)
                beta_hist[i].append(b_val)

            print(
                f"Ep {ep:02d} | Loss: {policy_loss.item():.2e} | ElitePerf (LogSum): {max(log_performances):.2f} | Elite: {len(elite_indices)}/{N_samples}")

        else:
            print(f"Ep {ep:02d} | Nessun guasto trovato. Esplorazione attiva.")
            loss_history.append(0.0)
            elite_history.append(0.0)
            for i in comps:
                alpha_hist[i].append(alpha_init[i])
                beta_hist[i].append(beta_init[i])

    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("--- Training concluso. Caricato il miglior modello salvato. ---")

    return model, loss_history, elite_history, alpha_hist, beta_hist

def plot_loss(loss_history, elite_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(loss_history)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True)

    ax2.plot(elite_history)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Elite Set Mean Performance")
    ax2.set_title("Elite Set Performance")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("Andamento_loss.png", dpi=300, bbox_inches='tight')

def plot_alpha_beta(alpha_hist, beta_hist):
    for comp in alpha_hist:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        epochs = range(len(alpha_hist[comp]))

        a_vals = alpha_hist[comp]
        ax1.plot(epochs, a_vals, color='tab:blue', marker='o', markersize=3, linewidth=1.5)
        ax1.set_title(f"Alpha Evolution (Biasing) - {comp}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Value")
        ax1.grid(True, linestyle='--', alpha=0.6)

        a_min, a_max = min(a_vals), max(a_vals)
        a_delta = max(0.01, a_max - a_min)
        ax1.set_ylim(a_min - a_delta * 0.1, a_max + a_delta * 0.1)

        b_vals = beta_hist[comp]
        ax2.plot(epochs, b_vals, color='tab:orange', marker='s', markersize=3, linewidth=1.5)
        ax2.set_title(f"Beta Evolution (Correction) - {comp}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Value")
        ax2.grid(True, linestyle='--', alpha=0.6)

        b_min, b_max = min(b_vals), max(b_vals)
        b_delta = max(0.01, b_max - b_min)
        ax2.set_ylim(b_min - b_delta * 0.1, b_max + b_delta * 0.1)

        plt.tight_layout()

        filename = f"Andamento_alfa_beta_{comp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_distribution(active_weights):
    if active_weights:
        plt.figure(figsize=(10, 6))
        plt.hist(np.log10(active_weights), bins=50, color='skyblue', edgecolor='black')
        plt.title("Analisi della Varianza: Distribuzione Logaritmica dei Pesi IS")
        plt.xlabel("Log10(Peso W)")
        plt.ylabel("Frequenza (Numero di Traiettorie)")
        plt.grid(True, alpha=0.3)
        plt.savefig("distribuzione_pesi_IS.png", dpi=300, bbox_inches='tight')
        print("--> Grafico salvato come 'distribuzione_pesi_IS.png'")
    else:
        print("Nessun peso maggiore di zero trovato per il grafico.")

def estimate_probability(trajs):
    return sum(tr["top"] for tr in trajs) / len(trajs)

def compare_MC_IS(lambda_, mu_, alpha, beta, T, N):
    trajs_mc = [simulate_CTMC(lambda_, mu_, {i: 1.0 for i in lambda_}, {i: 1.0 for i in lambda_}, T) for _ in range(N)]
    p_mc = sum(1 for tr in trajs_mc if tr["top"]) / N

    trajs_is = [simulate_CTMC(lambda_, mu_, alpha, beta, T) for _ in range(N)]

    weights = []
    for tr in trajs_is:
        if tr["top"]:
            log_w_bounded = min(tr["log_w"], 0.0)
            try:
                w = math.exp(log_w_bounded)
            except OverflowError:
                w = 0.0
            weights.append(w)
        else:
            weights.append(0.0)

    p_is = sum(weights) / N
    n_top_is = sum(1 for w in weights if w > 0)
    w_max = max(weights) if weights else 0.0
    w_min = min(w for w in weights if w > 0) if n_top_is > 0 else 0.0

    print(f"Top events IS osservati: {n_top_is}/{N}")
    print(f"Peso IS max: {w_max:.3e}")
    print(f"Peso IS min (non-zero): {w_min:.3e}")

    var_is = (sum(w ** 2 for w in weights) / N) - (p_is ** 2)

    if p_is > 1e-300:
        print(f"log10(IS estimate): {math.log10(p_is):.2f}")
    else:
        print("IS estimate sotto precisione numerica (p ≈ 0)")

    return p_mc, p_is, max(0, var_is), weights


if __name__ == "__main__":

    #lambda_ = {"A": 1e-5, "B": 2e-5, "C": 1e-6}
    #mu_ = {"A": 1e-1, "B": 1e-1, "C": 1e-1}

    lambda_ = {"A": 1e-5, "B": 2e-5, "C": 1e-6, "D": 1e-7, "E": 1e-8}
    mu_ = {"A": 1e-1, "B": 1e-1, "C": 1e-1, "D": 1e-1, "E": 1e-2}

    T = 1e3

    lambda_train = {k: v * 10 for k, v in lambda_.items()}

    print("Inizio training con Cross-Entropy Method + ML...")

    comps = list(lambda_.keys())
    n = len(comps)

    
    alpha_init = {i: 1.0 for i in comps}
    beta_init = {i: 1.0 for i in comps}
    traj_dummy = [simulate_CTMC(lambda_train, mu_, alpha_init, beta_init, T)]
    test_feats = extract_features(traj_dummy, lambda_train, mu_)
    input_dim = test_feats.shape[0]  

    print(f"Dimensione Input MLP rilevata: {input_dim}")

    model, loss_hist, elite_hist, alpha_hist, beta_hist = train_mlp_cross_entropy(
        lambda_train, mu_, T,
        epochs=100,
        N_trajs=500,  
        N_samples=500,
        rho=0.3,
        noise_std=1.0  
    )

    plot_loss(loss_hist, elite_hist)
    plot_alpha_beta(alpha_hist, beta_hist)

    
    alpha = {i: alpha_hist[i][-1] for i in alpha_hist}
    beta = {i: beta_hist[i][-1] for i in beta_hist}

    alpha_valutazione = {i: max(alpha[i], 10.0) for i in alpha}
    beta_valutazione = {i: min(beta[i], 0.5) for i in beta}

    print("\n" + "=" * 60)
    print("VALUTAZIONE FINALE")
    print("=" * 60)

    p_mc, p_is, var_is, weights = compare_MC_IS(lambda_, mu_, alpha_valutazione, beta_valutazione, T, N=500000)

    active_weights = [w for w in weights if w > 0]

    plot_distribution(active_weights)

    print(f"Crude MC estimate:     {p_mc:.6e}")
    print(f"IS estimate:           {p_is:.6e}")
    print(f"IS variance:           {var_is:.6e}")
    print(f"IS std dev:            {math.sqrt(var_is):.6e}")

    print(f"\nParametri ottimali:")
    for i in alpha:
        print(f"  {i}: alpha={alpha[i]:.4f}, beta={beta[i]:.4f}")