import os
import torch
from range_predictor import RangePredictor, train_range_predictor, FaultTreeGraph
from range_tester import run_range_tester

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_ranges(ft):
    model = train_range_predictor()

    model.eval()
    data = ft.to_pyg_data().to(device)
    with torch.no_grad():
        ranges, _ = model(data)
        r = ranges[0].cpu().numpy()
        a_opt = (r[0], r[1])
        b_opt = (r[2], r[3])

    print(f"-> Range suggeriti: Alpha = [{a_opt[0]:.2f}, {a_opt[1]:.2f}], Beta = [{b_opt[0]:.2f}, {b_opt[1]:.2f}]")

    return a_opt[0], a_opt[1], b_opt[0], b_opt[1]

def get_prob_est(ft, alfa_min, alfa_max, beta_min, beta_max):
    ranges_dict = {
        'alpha': (alfa_min, alfa_max),
        'beta': (beta_min, beta_max),
    }
    fault_tree_logic = ft.get_logic_function()
    run_range_tester(ft, fault_tree_logic, ranges_dict, T=100)

if __name__ == "__main__":
    ft = FaultTreeGraph()
    nodes = [ft.add_component(f"C{i}", 3e-3, 0.1) for i in range(2)]
    ft.add_gate('AND', nodes)
    alfa_min, alfa_max, beta_min, beta_max = get_ranges(ft)
    get_prob_est(ft, alfa_min, alfa_max, beta_min, beta_max)


