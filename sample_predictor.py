import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

from range_predictor import generate_simple_fault_tree
from range_tester import simulate_CTMC


class SampleGNN(nn.Module):
    def __init__(self, n_features=5):
        super(SampleGNN, self).__init__()
        self.conv1 = GCNConv(n_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.header = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [log10_N_is, log10_N_mc]
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.header(x)


def find_required_samples(lambda_, mu_, a_dict, b_dict, T, ft_logic, target_cv=0.05, max_n=100000):
    samples = []
    batch_size = 500
    n = 0
    while n < max_n:
        res = [simulate_CTMC(lambda_, mu_, a_dict, b_dict, T, ft_logic) for _ in range(batch_size)]
        w = [math.exp(r['log_w']) if r['top'] else 0.0 for r in res]
        samples.extend(w)
        n += batch_size
        if n >= 1000:
            avg = np.mean(samples)
            if avg > 0:
                cv = (np.std(samples) / np.sqrt(n)) / avg
                if cv <= target_cv: return n
    return max_n


def train_sample_predictor(n_train=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = []

    print("=" * 60)
    print("TRAINING GNN - STIMA NUMERO DI CAMPIONI")
    print("=" * 60)

    for i in range(n_train):
        data_ft = generate_simple_fault_tree()

        # Misuriamo N per Monte Carlo (Target)
        params_mc = {
            'lambda_': data_ft['lambda_'], 'mu_': data_ft['mu_'],
            'a_dict': {c: 1.0 for c in data_ft['lambda_']},
            'b_dict': {c: 1.0 for c in data_ft['lambda_']},
            'T': 100, 'ft_logic': data_ft['fault_tree']
        }
        n_mc = find_required_samples()

        # Per IS, qui potresti simulare con i range predetti dal Predictor 1
        # Per ora usiamo una stima di guadagno basata sulla struttura
        n_is = max(500, n_mc / 3.0)

        pyg_data = data_ft['graph'].to_pyg_data()
        pyg_data.y = torch.tensor([[math.log10(n_is), math.log10(n_mc)]], dtype=torch.float)
        dataset.append(pyg_data)

    # Training
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = SampleGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(100):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(batch), batch.y)
            loss.backward()
            optimizer.step()

    return model


def get_predicted_samples(model, pyg_data):
    """
    Riceve il modello addestrato e il grafo (PyG Data).
    Restituisce i numeri interi di campioni per IS e MC.
    """
    model.eval()
    with torch.no_grad():
        # Il modello restituisce [log10(N_is), log10(N_mc)]
        prediction = model(pyg_data)

        # Invertiamo il logaritmo: 10^valore
        n_is = torch.pow(10, prediction[0, 0]).item()
        n_mc = torch.pow(10, prediction[0, 1]).item()

        # Arrotondiamo a intero e mettiamo un limite minimo di sicurezza (es. 100)
        return max(100, int(n_is)), max(100, int(n_mc))