* **`fault_tree(state)`**: Definisce la logica booleana del sistema. Determina se la combinazione degli stati dei componenti (0 = funzionante, 1 = guasto) attiva il "Top Event" per diverse topologie (Generic, 2oo3, Bridge, SPOF).
* **`simulate_CTMC(lambda_, mu_, alpha, beta, T)`**: Gestisce la simulazione stocastica "Continuous Time Markov Chain". Genera le traiettorie temporali applicando i parametri di biasing $\alpha$ e $\beta$ per accelerare la comparsa di eventi rari.
* **`extract_features(trajs, lambda_, mu_)`**: Elabora i dati grezzi delle traiettorie simulate per produrre i vettori numerici (feature) utilizzati come input dalla rete neurale.
* **`AlphaBetaMLP`**: Classe che definisce la struttura della rete neurale (Multi-Layer Perceptron). Riceve lo stato del sistema e predice i valori ottimali di $\alpha$ e $\beta$.
* **`sample_parameters(alpha_raw, beta_raw, noise_std)`**: Introduce variabilità (rumore) nei parametri predetti dalla rete per favorire l'esplorazione di diverse configurazioni durante il training.
* **`cross_entropy_loss_ML(samples_data, rho)`**: Calcola la performance dei campioni e identifica l' "Elite Set" (i campioni migliori), calcolando la media delle performance in scala logaritmica per la stabilità numerica.
* **`train_mlp_cross_entropy(...)`**: Coordina l'intero processo di addestramento. Gestisce i cicli di simulazione, l'aggiornamento dei pesi della rete neurale e il salvataggio del modello migliore.
* **`plot_loss(loss_history, elite_history, tree_type)`**: Genera e salva i grafici relativi all'andamento della Loss e della performance dell'Elite Set durante le epoche di training.
* **`plot_alpha_beta(alpha_hist, beta_hist, tree_type)`**: Produce i grafici affiancati per l'evoluzione temporale dei parametri di biasing ($\alpha$) e correzione ($\beta$) per ogni singolo componente.
* **`plot_weights_distribution(active_weights, tree_type)`**: Crea un istogramma della distribuzione logaritmica dei pesi Importance Sampling per visualizzare la riduzione della varianza.
* **`estimate_probability(trajs)`**: Fornisce una stima preliminare della probabilità di guasto basata sulla frequenza degli eventi nelle traiettorie.
* **`compare_MC_IS(...)`**: Funzione di validazione finale che confronta i risultati ottenuti tramite Monte Carlo standard con quelli dell'Importance Sampling ottimizzato.

1. **Inizializzazione e Setup**: Il sistema rileva l'hardware disponibile (CPU/CUDA) e definisce i parametri nominali dei componenti ($\lambda, \mu$) e la topologia dell'albero dei guasti desiderata.
2. **Ciclo di Training (Cross-Entropy Method)**: 
    * La MLP riceve in input le caratteristiche delle traiettorie attuali.
    * Vengono campionati nuovi parametri di biasing ($\alpha, \beta$) introducendo del rumore per favorire l'esplorazione.
    * Il simulatore CTMC genera nuove traiettorie; quelle che portano al guasto del sistema entrano nell' "Elite Set".
    * La rete neurale viene aggiornata tramite backpropagation per massimizzare la probabilità di osservare eventi rari nelle epoche successive.
3. **Analisi e Visualizzazione**: Al termine del training, vengono generati grafici che mostrano la convergenza della Loss e l'evoluzione temporale dei parametri ottimali appresi per ogni componente.
4. **Valutazione Finale**: Il modello ottimizzato viene utilizzato per eseguire una simulazione su larga scala (500.000 campioni). I risultati vengono confrontati con il metodo Monte Carlo standard per validare l'efficacia della riduzione della varianza e l'accuratezza della stima IS.
