# Adaptive Importance Sampling for Rare Event Simulation in Fault Trees

Rare event simulation in Fault Tree Analysis using adaptive Importance Sampling, optimized with neural networks (MLP) and Cross-Entropy method.

## Main Functions

| Function | Description |
|----------|-------------|
| `fault_tree(state)` | Boolean logic for Top Event: **SPOF** `A ∨ B ∨ (C ∧ D ∧ E)`, **Bridge** `[(A ∨ B) ∧ (C ∨ D)] ∨ E`, **2oo3** `(A ∧ B) ∨ (A ∧ C) ∨ (B ∧ C)`, **Generic** `(A ∧ B) ∨ C` |
| `simulate_CTMC(...)` | CTMC simulation with likelihood ratio for IS |
| `AlphaBetaMLP` | Neural network that predicts α and β parameters |
| `train_mlp_cross_entropy(...)` | Training with Cross-Entropy Method |
| `evaluate_model(...)` | Evaluation and comparison IS vs naive Monte Carlo |
| `save_results_to_file(...)` | Saves results to `results/{tree_type}/` |
| `plot_results_extended(...)` | Generates loss and parameters plots |
