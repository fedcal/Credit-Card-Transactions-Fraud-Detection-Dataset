---
layout: home
title: Home
nav_order: 1
description: >-
  Pipeline ML production-friendly per la fraud detection su transazioni con
  carta di credito. Classificazione sbilanciata, AUC-PR, ottimizzazione
  della soglia decisionale su matrice di costi, split temporale walk-forward.
permalink: /
---

<div class="hero-banner" markdown="0">
  <h1>Credit Card Fraud Detection &mdash; ML Pipeline</h1>
  <p>
    Dal CSV Kaggle a <code>predict_fraud()</code>: feature temporali e
    geografiche, aggregati expanding per cliente (no leakage), Logistic
    Regression e Random Forest con <code>class_weight='balanced'</code>,
    walk-forward time-series CV, ottimizzazione della soglia decisionale
    su matrice di costi business.
  </p>
</div>

## In sintesi

Progetto di riferimento del percorso **Machine Learning Engineer** di
[DataMasters](https://datamasters.it/)/Skiller. Implementa l'intero flusso
di lavoro di un classificatore tabular su un problema **altamente
sbilanciato** (~0.5% di frodi) e con **forte dimensione temporale**, dal
dato grezzo all'inferenza, con focus su **rigorosità metodologica**,
**prevenzione del leakage temporale** e **scelta della soglia operativa**
guidata dai costi di business.

<div class="kpi-grid" markdown="0">
  <div class="kpi-card">
    <div class="kpi-label">AUC-PR (RandomForest)</div>
    <div class="kpi-value">~0.85</div>
    <div>holdout out-of-time</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Recall @ soglia ottimale</div>
    <div class="kpi-value">~0.78</div>
    <div>frodi intercettate</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Precision @ soglia ottimale</div>
    <div class="kpi-value">~0.74</div>
    <div>F1 ~ 0.76</div>
  </div>
</div>

## Repository GitHub

- **Nome del repository**: `credit-card-fraud-detection`
- **URL**: [github.com/fedcal/credit-card-fraud-detection](https://github.com/fedcal/credit-card-fraud-detection)
- **Documentazione (questo sito)**: pubblicata via **GitHub Pages** dalla cartella
  [`/docs`](https://github.com/fedcal/credit-card-fraud-detection/tree/main/docs).

{: .note }
> La documentazione viene servita direttamente dai file Markdown della cartella
> `docs/`, processati da Jekyll con il tema **Just the Docs**.
> Ogni push su `main` aggiorna automaticamente il sito.

## Quick start

```bash
git clone https://github.com/fedcal/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python3 -m venv venv && source venv/bin/activate
pip install -e ".[notebooks]"

# Scarica fraudTrain.csv e fraudTest.csv da Kaggle in data/raw/
# https://www.kaggle.com/datasets/kartik2112/fraud-detection

fraud-train --quick     # smoke-test ~1-2 min su 50k righe
fraud-train             # full pipeline ~10-30 min su 1.5M righe
```

Inferenza programmatica:

```python
from fraud_pipeline.inference import predict_fraud, example_transaction

tx = example_transaction()
res = predict_fraud(tx)
print(res)
# {'fraud_probability': 0.0034, 'is_fraud': False, 'threshold': 0.20}
```

L'API `predict_fraud()` accetta sia dizionari (singola transazione) sia
`pd.DataFrame` (batch).

## Risultati di riferimento (full tuning)

Su test set out-of-time Kaggle (~555k transazioni, ~0.4% frodi). I numeri
esatti dipendono dal seed e dal sample, ma l'ordine di grandezza atteso è:

| Modello              | AUC-PR | Recall@t* | Precision@t* | F1@t*  |
|:--|--:|--:|--:|--:|
| **RandomForest**     | ~0.85  | ~0.78     | ~0.74        | ~0.76  |
| LogisticRegression   | ~0.55  | ~0.70     | ~0.05        | ~0.10  |
| XGBoost (opzionale)  | ~0.88  | ~0.82     | ~0.78        | ~0.80  |

{: .tip }
> Il modello lineare ha alto recall ma bassa precision (troppi falsi allarmi);
> il Random Forest cattura le interazioni non lineari (importo + orario +
> distanza) e ottiene un trade-off molto migliore. La soglia ottimale `t*`
> (cost-based) tipicamente cade fra 0.15 e 0.40, ben distante dal default 0.5.

## Mappa della documentazione

### [Teoria](teoria/)

Fondamenti per leggere i risultati del progetto:

- [Classificazione sbilanciata](teoria/01_classificazione_sbilanciata/) — class imbalance, accuracy paradox, AUC-PR, strategie di mitigazione.
- [Metriche fraud detection](teoria/02_metriche_fraud_detection/) — recall, precision, F1, F-beta, AUC-PR, MCC.
- [Feature engineering temporali](teoria/03_feature_engineering_temporali/) — timestamp, distanza Haversine, aggregati expanding.
- [Split temporale & leakage](teoria/04_split_temporale_e_leakage/) — TimeSeriesSplit, walk-forward CV, anti-pattern.
- [Modelli supervisionati](teoria/05_modelli_supervisionati/) — LogReg, RandomForest, XGBoost, gestione sbilanciamento.
- [Threshold tuning & costi](teoria/06_threshold_tuning_e_costi/) — matrice di costi, soglia ottimale, deployment.

### [Scelte tecniche](scelte_tecniche/)

Decisioni architetturali e di modellazione:

- [Architettura](scelte_tecniche/architettura/) — moduli `src/fraud_pipeline/`, flusso dati, CLI.
- [Scelte di modellazione](scelte_tecniche/scelte_modello/) — trade-off espliciti, validation strategy, drift.

## Stack tecnologico

| Layer | Tecnologie |
|:--|:--|
| Linguaggio | Python 3.11+ |
| ML | scikit-learn, xgboost (opzionale) |
| Data | pandas, numpy, scipy |
| Plotting | matplotlib, seaborn |
| Notebook | jupyter, jupytext |
| Persistenza | joblib |
| Documentazione | Jekyll + Just the Docs |

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso
*Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti:
[**federicocalo.dev**](https://federicocalo.dev){: .btn .btn-purple }
