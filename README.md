# Adaptive Importance Sampling for Rare Event Simulation in Fault Trees

Simulazione di eventi rari in Fault Tree Analysis mediante Importance Sampling adattivo, ottimizzato con reti neurali (MLP) e metodo Cross-Entropy.

## Funzioni principali

| Funzione | Descrizione |
|----------|-------------|
| `fault_tree(state)` | Logica booleana del Top Event: **SPOF** `A ∨ B ∨ (C ∧ D ∧ E)`, **Bridge** `[(A ∨ B) ∧ (C ∨ D)] ∨ E`, **2oo3** `(A ∧ B) ∨ (A ∧ C) ∨ (B ∧ C)`, **Generic** `(A ∧ B) ∨ C` |
| `simulate_CTMC(...)` | Simulazione CTMC con likelihood ratio per IS |
| `AlphaBetaMLP` | Rete neurale che predice i parametri α e β |
| `train_mlp_cross_entropy(...)` | Training con Cross-Entropy Method |
| `evaluate_model(...)` | Valutazione e confronto IS vs Monte Carlo naive |
| `save_results_to_file(...)` | Salva i risultati in `results/{tree_type}/` |
| `plot_results_extended(...)` | Genera grafici di loss e parametri |
