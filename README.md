# Modelli Predittivi per l’Identificazione di Frodi nelle Transazioni con Carta di Credito

 Simulare il lavoro di un team dati che supporta una banca/emittente di carte di credito con un modello che dovrà segnalare le transazioni sospette in tempo quasi-reale permettendo di ridurre le frodi non intercettate, mantenendo però un tasso di falsi allarmi accettabile dal business.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-Just%20the%20Docs-blueviolet.svg)](https://fedcal.github.io/credit-card-fraud-detection/)

## Dettagli

### Dataset

Userete il dataset “Credit Card Transactions Fraud Detection Dataset” disponibile su Kaggle. Si tratta di transazioni simulate ma realistiche tra carte di 1.000 clienti e circa 800 merchant, nel periodo 01/01/2019–31/12/2020, con circa 1,5 milioni di righe complessive nei file fraudTrain.csv e fraudTest.csv. (Kaggle)

Ogni transazione contiene informazioni su timestamp, importo, cliente, merchant, categoria, localizzazione, e un’etichetta che indica se la transazione è fraudolenta o legittima.

### Contesto e problema

L’obiettivo è simulare il lavoro di un team dati che supporta una banca/emittente di carte di credito con un modello che dovrà segnalare le transazioni sospette in tempo quasi-reale permettendo di ridurre le frodi non intercettate, mantenendo però un tasso di falsi allarmi accettabile dal business.

Si deve tenere conto del forte sbilanciamento nei dati (la percentuale di frodi è molto bassa rispetto alle transazioni legittime, come tipicamente accade in questi problemi) e della natura temporale dei dati, essendo le transazioni distribuite su due anni.

### Obiettivi

Alla fine del progetto, lo studente dovrebbe essere in grado di:

- Impostare correttamente un problema di classificazione altamente sbilanciato, scegliendo metriche adeguate e strategie di validazione coerenti.
- Progettare feature ingegnerizzate su dati transazionali e temporali (cliente, merchant, tempo, geolocalizzazione).
- Confrontare almeno 2 modelli di machine learning supervisionati, valutandone pro e contro.
- Costruire una mini-pipeline con preprocessing, salvataggio del modello e inferenza su nuove transazioni.

### Obiettivo tecnico del progetto

Progettare, implementare e documentare una pipeline completa che:

- Carica e prepara i dati (train + test) nel rispetto della dimensione temporale.
- Esegue preprocessing e feature engineering (di base e/o avanzata).
- Addestra almeno due tipologie di modelli di machine learning.
- Valida i modelli con un set di metriche mirate al contesto “fraud detection”.
- Produce una funzione o script di “serving” che, dato un batch di nuove transazioni, restituisce una probabilità di frode e una decisione (fraud/legit).
- Fornisce una lettura critica dei risultati, considerando anche aspetti di interpretabilità e possibili evoluzioni future (es. oversampling avanzato, GAN, metodi sequenziali, graph-based). (arXiv)

### Fasi di lavoro consigliate

Fase 1: Comprensione del business e analisi esplorativa
Definire una “storia” di business: quanto costa una frode non intercettata? Quanto costa bloccare erroneamente una transazione legittima?

Formalizzare una matrice dei costi o almeno un rapporto di priorità e tenerla come riferimento per l’ottimizzazione della soglia.

EDA:

- Distribuzione delle classi.
- Distribuzione dell’importo.
- Pattern temporali.
- Analisi per categoria e merchant.
- Analisi geografica cliente–merchant.

Fase 2: Preprocessing e feature engineering “base”

- Gestire valori mancanti.
- Codificare variabili categoriche.
- Normalizzare/scalare feature numeriche.
- Creare feature temporali.
- Creare feature geografiche.
- Creare aggregati per cliente e merchant.

Fase 3: Feature engineering avanzata

- Pattern temporali per cliente: tempo dall’ultima transazione, frequenze recenti, anomalie rispetto alla storia.
- Pattern per merchant: tasso storico frodi, combinazioni categorie, nuovi merchant per il cliente.
- Indicatori di anomalia relativa: distanza insolita, orario inconsueto.
- Opzionale: feature sequence-based (HMM, sequenze temporali). (arXiv)

L’obiettivo non è implementare modelli complessi, ma ragionare come se si modellassero sequenze.

Fase 4: Modellazione

- Almeno due modelli supervisionati: Logistic Regression, Random Forest, Gradient Boosting.
- Una strategia di gestione dello sbilanciamento: class_weight, sampling, SMOTE.

Fase 5: Validazione, metriche e ottimizzazione soglia

- Split temporale.
- Metriche: recall frodi, precision frodi, F1 frodi, AUC-PR.
- Curve precision-recall e ottimizzazione soglia in base ai costi.
- Discussione trade-off.

Fase 6: Interpretabilità e analisi errori

- Feature importance, coefficienti, SHAP/LIME.
- Analisi errori: false negative e false positive.
- Connessione con possibili azioni di business.

Fase 7: Proto-pipeline di produzione

- Strutturare codice riutilizzabile.
- Salvare modello e preprocessori.
- Script di inferenza su nuove transazioni.
- Facoltativo: idee di monitoring (drift, retrain).

### Deliverable richiesti
- Notebook o script Python con pipeline completa.
- Report di 3–4 pagine che descriva:
  - contesto di business,
  - scelte di preprocessing e feature engineering,
  - modelli provati e metriche,
  - analisi errori e feature più influenti,
  - considerazioni su leakage, drift e sbilanciamento.

### Criteri di valutazione

- Solidità impostazione e metriche.
- Qualità EDA e feature engineering.
- Rigorosità validazione.
- Profondità discussione critica.
- Qualità codice e report.
- Creatività nelle estensioni.

---

## Repository GitHub

**Nome del repository pubblico**: `credit-card-fraud-detection`
URL: <https://github.com/fedcal/credit-card-fraud-detection>

Il sito documentazione è generato con **Jekyll + Just the Docs** (mobile-first, SEO-ready, dark mode, MathJax) ed è servito da GitHub Pages direttamente dalla cartella [`/docs`](docs/). Da abilitare in *Settings → Pages → Source = Deploy from a branch, Branch = `main` / `/docs`* alla prima volta. Ogni push su `main` aggiorna automaticamente il sito.

## Documentazione completa

Sito statico con teoria, scelte tecniche e diagramma architetturale:
<https://fedcal.github.io/credit-card-fraud-detection/>

## Quick start

### 1. Setup

```bash
git clone https://github.com/fedcal/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[notebooks]"       # installa il pacchetto + dipendenze notebook
# Opzionale: aggiungi XGBoost
pip install -e ".[notebooks,xgboost]"
```

### 2. Scaricare il dataset Kaggle

Il dataset *non* è committato (peso ~470 MB, licenza Kaggle). Scaricalo da:

> <https://www.kaggle.com/datasets/kartik2112/fraud-detection>

e copia i due file in `data/raw/`:

```
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

### 3. Pipeline completa (training + valutazione)

```bash
fraud-train               # full pipeline su tutto il dataset (~10-30 min)
fraud-train --quick       # smoke-test su 50k righe campionate (~30 secondi)
```

Output:

- `reports/models/best_model.joblib` — pipeline serializzata pronta per l'inferenza
- `reports/cv_summary.csv` — confronto modelli con AUC-PR / recall / precision
- `reports/holdout_metrics.json` — metriche test set per ogni modello
- `reports/threshold_analysis.csv` — analisi della soglia e costi
- `reports/figures/*.png` — curve PR/ROC, confusion matrix, calibration

### 4. Inferenza

```python
from fraud_pipeline.inference import predict_fraud, example_transaction

tx = example_transaction()
result = predict_fraud(tx)
print(result)
# {'fraud_probability': 0.0034, 'is_fraud': False, 'threshold': 0.20}
```

L'API accetta dizionari (singola transazione) o `pd.DataFrame` (batch).

### 5. Notebook didattici

```bash
python scripts/build_notebooks.py        # rigenera i 4 .ipynb da sorgente Python
jupyter lab notebooks/
```

I 4 notebook sono pensati per essere letti in sequenza:

1. **`01_eda_class_imbalance.ipynb`** — EDA, distribuzione classi, pattern temporali e geografici.
2. **`02_feature_engineering.ipynb`** — feature temporali, distanza geografica, aggregati cliente/merchant.
3. **`03_models_baseline_vs_ensemble.ipynb`** — Logistic Regression, RandomForest, (XGBoost) e gestione sbilanciamento.
4. **`04_threshold_tuning_and_errors.ipynb`** — curve PR, scelta soglia su matrice di costi, analisi errori.

## Struttura del repository

```
src/fraud_pipeline/        Libreria Python installabile
├── config.py                Path, costanti, iperparametri, seed
├── data.py                  Load CSV, parse datetime, schema check
├── features.py              FraudFeatureEngineer (transformer sklearn)
├── preprocessing.py         ColumnTransformer (numeriche/categoriche)
├── models.py                Pipeline candidate (LogReg, RandomForest, XGBoost)
├── tuning.py                Walk-forward CV temporale, scoring AUC-PR
├── threshold.py             Ottimizzazione soglia su matrice costi
├── evaluation.py            PR, ROC, confusion matrix, classification report
├── inference.py             predict_fraud() per nuove transazioni
└── pipeline.py              Orchestrator + CLI fraud-train

notebooks/                 Documentazione esecutiva
docs/
├── teoria/                  6 spiegazioni didattiche
└── scelte_tecniche/         Architettura, decisioni di modello
data/                      Dataset Kaggle (gitignored, da scaricare manualmente)
reports/                   Output: figures, metrics, models (gitignored)
```

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di DataMasters/Skiller (2026).

Per altri progetti, contatti e portfolio: <https://federicocalo.dev>.

## Licenza

[MIT License](LICENSE) © 2026 Federico Calò.

Il dataset *Credit Card Transactions Fraud Detection* è di proprietà di chi lo ha pubblicato su Kaggle ed è soggetto ai termini d'uso indicati nella relativa pagina Kaggle.
