# Adaptive Importance Sampling for Rare Event Simulation in Fault Trees

Rare event simulation in Fault Tree Analysis using adaptive Importance Sampling, optimized with neural networks (MLP) and Cross-Entropy method.

## Preset Configurations

| Preset | Structure | P estimated | Alpha range |
|--------|-----------|-------------|-------------|
| `medium` | Hierarchical OR-OR | ~10⁻⁴ | [5, 30] |
| `hard` | Deep AND (4 comp) | ~10⁻⁵ | [2, 10] |
| `very_hard` | Deep AND (6 comp) | ~10⁻⁸ | [2, 5] |
| `extreme` | Deep AND (8 comp) | ~10⁻¹¹ | [1.5, 4] |

## Main Functions

| Function | Description |
|----------|-------------|
| `OptimizedConfig` | Configuration class with preset fault trees, IS parameters (α, β ranges), and training hyperparameters |
| `simulate_CTMC(...)` | CTMC simulation with likelihood ratio for IS |
| `AlphaBetaMLP` | Neural network predicting α and β parameters per component |
| `train_mlp_cross_entropy(config)` | Training with Cross-Entropy Method and Policy Gradient |
| `evaluate_model(model, config)` | Evaluation and comparison IS vs naive Monte Carlo |
| `save_results_to_file(...)` | Saves results to `results/{tree_type}/` |
| `plot_results_extended(...)` | Generates loss and parameter evolution plots |
