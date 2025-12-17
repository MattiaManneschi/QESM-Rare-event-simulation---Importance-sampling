import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 1. GENERAZIONE ALBERO ---
def generate_random_tree(n_comp, n_gates):
    total_nodes = n_comp + n_gates
    adj = torch.zeros((total_nodes, total_nodes), device=device)
    for gate in range(n_comp, total_nodes):
        inputs = np.random.choice(gate, np.random.randint(2, 4), replace=False)
        for i in inputs: adj[i, gate] = 1
    return adj


# --- 2. MLP ARCHITECTURE ---
class ImportanceMLP(nn.Module):
    def __init__(self, n_nodes, n_comp):
        super().__init__()
        input_dim = (n_nodes * n_nodes) + (2 * n_comp)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_comp * 2),
            nn.Softplus()
        )

    def forward(self, adj, l, m):
        x = torch.cat([adj.view(-1), l, m]).unsqueeze(0)
        out = self.net(x)
        return out[:, :l.size(0)].squeeze(0), out[:, l.size(0):].squeeze(0)


# --- 3. LOGICA EVALUATION (GPU) ---
def evaluate_fault_tree(samples, adj, n_comp):
    n_samples, n_nodes = samples.shape[0], adj.shape[0]
    states = torch.zeros((n_samples, n_nodes), device=device)
    states[:, :n_comp] = samples
    for g in range(n_comp, n_nodes):
        child_indices = torch.where(adj[:, g] == 1)[0]
        states[:, g] = (torch.sum(states[:, child_indices], dim=1) >= 1).float()
    return states[:, -1]


# --- 4. MAIN TRAINING LOOP CON EARLY STOPPING ---
def run_optimized_project():
    N_COMP, N_GATES = 6, 4
    N_NODES = N_COMP + N_GATES
    N_SAMPLES = 10000

    adj = generate_random_tree(N_COMP, N_GATES)
    lmbdas = torch.full((N_COMP,), 0.005, device=device)
    mus = torch.full((N_COMP,), 0.15, device=device)

    model = ImportanceMLP(N_NODES, N_COMP).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Parametri Early Stopping
    patience = 15
    best_loss = float('inf')
    counter = 0
    history = []

    print(f"Hardware: {device} | Inizio Training con Cross-Entropy...")

    for epoch in range(300):
        optimizer.zero_grad()

        # Predizione Alpha/Beta
        alphas, betas = model(adj, lmbdas, mus)
        p_p = (lmbdas * alphas) / (lmbdas * alphas + mus * betas)
        p_r = lmbdas / (lmbdas + mus)

        # Campionamento IS
        samples = torch.bernoulli(p_p.expand(N_SAMPLES, -1))

        # Likelihood Ratio (f/g)
        prob_f = torch.prod(torch.where(samples == 1, p_r, 1 - p_r), dim=1)
        prob_g = torch.prod(torch.where(samples == 1, p_p, 1 - p_p), dim=1)
        L = (prob_f / (prob_g + 1e-8)).detach()  # Detach per CE

        # Valutazione
        is_failed = evaluate_fault_tree(samples, adj, N_COMP)

        # LOSS: CROSS-ENTROPY (Punto 2 appunti)
        log_q = torch.sum(torch.where(samples == 1, torch.log(p_p + 1e-8), torch.log(1 - p_p + 1e-8)), dim=1)
        loss = -torch.mean(is_failed * L * log_q)

        loss.backward()
        optimizer.step()

        # Logica Early Stopping
        current_loss = loss.item()
        history.append({'epoch': epoch, 'loss': current_loss, 'var': torch.var(is_failed * L).item()})

        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1

        if counter >= patience:
            print(f"Early Stopping all'epoca {epoch}! Nessun miglioramento per {patience} epoche.")
            break

        if epoch % 20 == 0:
            print(f"Epoca {epoch:3d} | CE Loss: {current_loss:.4f} | Varianza: {history[-1]['var']:.2e}")

    # --- RISULTATI ---
    df = pd.DataFrame(history)
    df.to_csv('log_ricerca.csv', index=False)
    plt.plot(df['epoch'], df['loss'])
    plt.title('Convergenza Cross-Entropy (MLP)');
    plt.xlabel('Epoca');
    plt.ylabel('Loss')
    plt.savefig('convergenza.png');
    plt.show()


if __name__ == "__main__":
    run_optimized_project()
