---
layout: default
title: Architettura
parent: Scelte tecniche
nav_order: 1
description: >-
  Mappa dei moduli `src/fraud_pipeline/`, flusso dei dati, CLI, file
  persistenti e razionali delle decisioni di organizzazione del codice.
---

# Architettura della pipeline
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Vista a 3 layer

La pipeline è organizzata in tre layer ortogonali:

```
┌─────────────────────────────────────────────────────────┐
│ Application layer                                       │
│   CLI (fraud-train, fraud-predict)                      │
│   Notebook 01–04 (didattici)                            │
│   pytest (test_features, test_inference)                │
└─────────────────────────────────────────────────────────┘
                          ↓ chiama
┌─────────────────────────────────────────────────────────┐
│ Pipeline orchestration                                  │
│   pipeline.run_full_pipeline()                          │
│   inference.predict_fraud()                             │
└─────────────────────────────────────────────────────────┘
                          ↓ compone
┌─────────────────────────────────────────────────────────┐
│ Domain modules (single responsibility)                  │
│   data | features | preprocessing | models              │
│   tuning | threshold | evaluation                       │
│   config (costanti, niente logica)                      │
└─────────────────────────────────────────────────────────┘
```

Ogni layer dipende solo da quelli sotto. La separazione consente:

- **Test in isolamento** (mocking dei layer inferiori).
- **Sostituzione locale**: cambiare modello implica toccare solo `models.py`.
- **Riutilizzo**: i notebook chiamano le stesse funzioni della CLI.

## 2. Mappa dei moduli `src/fraud_pipeline/`

| Modulo            | Responsabilità                                            | Linee |
|-------------------|-----------------------------------------------------------|-------|
| `config.py`       | Path filesystem, costanti, schema dataset, iperparametri default, dataclass `PipelineConfig` | ~150 |
| `data.py`         | I/O CSV, schema validation, ordinamento cronologico, split temporale, smoke-test sampling | ~170 |
| `features.py`     | `FraudFeatureEngineer` (transformer sklearn): temporali, geografiche, aggregati cliente | ~210 |
| `preprocessing.py`| `ColumnTransformer`: numeric (imputer + scaler), nominal (imputer + OneHotEncoder) | ~110 |
| `models.py`       | Pipeline candidate (LogReg, RF, opzionalmente XGBoost) con `class_weight` | ~135 |
| `tuning.py`       | `TimeSeriesSplit` + GridSearchCV / RandomizedSearchCV su scoring `average_precision` | ~190 |
| `threshold.py`    | `CostMatrix`, `optimal_threshold_by_cost`, `threshold_sweep` | ~190 |
| `evaluation.py`   | `compute_metrics`, curve PR/ROC, confusion matrix, feature importance | ~220 |
| `inference.py`    | `predict_fraud(tx)`, lazy load model + threshold, CLI `fraud-predict` | ~190 |
| `pipeline.py`     | Orchestratore + entry point CLI `fraud-train` | ~280 |

Tutti i moduli sono **piccoli e focalizzati** (≤300 righe). Nessun monolite.

## 3. Flusso dati end-to-end

```
                       ┌─────────────────────────────────┐
                       │ data/raw/fraudTrain.csv         │
                       │ data/raw/fraudTest.csv          │
                       └────────────┬────────────────────┘
                                    │
                                    ↓ data.load_train_test()
                       ┌─────────────────────────────────┐
                       │ DataFrame ordinato cronologic.  │
                       │ + schema validato                │
                       └────────────┬────────────────────┘
                                    │
                                    ↓ data.split_features_target()
                            ┌───────┴───────┐
                            ↓               ↓
                       X_train,       X_test,
                       y_train         y_test
                            │               │
                            ↓               ↓
                Pipeline:                   |
                  feature_engineer  ━━━━━━━┓│
                       (sklearn)           ││
                  preprocessor             ││  fit -> transform -> predict
                  model (LogReg/RF/XGB)    ││
                                           ┃│
                            │              ▼│
                            ↓ TimeSeriesSplit CV
                       Tuning (AUC-PR)      |
                            |               |
                            ↓               |
                       best_model           |
                            |               |
                            ↓               |
                       predict_proba(X_test)│
                            |               |
                            ↓               ↓
                       CostMatrix     [MODEL]
                            |               |
                            ↓               ↓
                       optimal_threshold    |
                            |               ↓
                            └────→ Persistenza ────→ reports/models/
                                                       ├── best_model.joblib
                                                       └── threshold.json
```

## 4. Pipeline sklearn: tre step concatenati

L'oggetto `sklearn.pipeline.Pipeline` finale ha la struttura:

```python
Pipeline(steps=[
    ("feature_engineer", FraudFeatureEngineer()),     # custom transformer
    ("preprocessor", ColumnTransformer(...)),         # imputer + scaler + OHE
    ("model", LogisticRegression(...)),               # o RF/XGB
])
```

### Perché un singolo `Pipeline`

1. **`fit_predict_proba`**: l'API sklearn `pipeline.predict_proba(X)` esegue automaticamente: feature engineering → preprocessing → modello. Niente codice custom.
2. **No leakage in CV**: ogni fold della `TimeSeriesSplit` esegue `pipeline.fit` solo sul subset di training.
3. **Serializzazione bit-identica**: `joblib.dump(pipeline, "model.joblib")` salva tutto. L'inferenza è esattamente quella del training.

## 5. CLI

Due entry point dichiarati in `pyproject.toml`:

```toml
[project.scripts]
fraud-train = "fraud_pipeline.pipeline:main_train"
fraud-predict = "fraud_pipeline.inference:main_predict"
```

### `fraud-train`

```bash
fraud-train                      # full pipeline
fraud-train --quick              # smoke-test 50k righe
fraud-train --xgboost            # include XGBoost
fraud-train --cost-fn 200 --cost-fp 10  # cost matrix custom
```

Output:
- `reports/models/best_model.joblib`
- `reports/models/threshold.json`
- `reports/cv_summary.csv`
- `reports/holdout_metrics.json`
- `reports/threshold_analysis.csv`
- `reports/figures/*.png`

### `fraud-predict`

```bash
fraud-predict --input transactions.csv --output predictions.csv
fraud-predict --input transactions.csv --output predictions.csv --threshold 0.20
```

Legge il CSV, applica il modello, scrive un CSV con colonne `fraud_probability`, `is_fraud`, `threshold`.

## 6. File persistenti: separazione di concerns

| Cartella               | Contenuto                                  | Gitignored?   |
|------------------------|--------------------------------------------|---------------|
| `data/raw/`            | CSV Kaggle originali                       | Sì (470 MB)   |
| `data/processed/`      | (riservato per snapshot intermedi)         | Sì            |
| `data/external/`       | (riservato per dati esterni)               | Sì            |
| `reports/models/`      | `.joblib` + `threshold.json`               | Sì (>200 MB)  |
| `reports/figures/`     | Plot PNG (PR, ROC, confusion matrix)       | Sì            |
| `reports/cv_summary.csv` | Tabella confronto modelli                | Sì            |
| `reports/holdout_metrics.json` | Metriche test out-of-time          | Sì            |
| `reports/threshold_analysis.csv` | Sweep di soglie                  | Sì            |

Tutto ciò che è ricostruibile da `fraud-train` viene **gitignored** (vedi `.gitignore`). Il repo committato resta sotto pochi MB.

## 7. Notebook didattici

I 4 notebook in `notebooks/` sono **rigenerabili** da `scripts/build_notebooks.py`. Vantaggi:

- **Sorgente in Python**: diff Git puliti.
- **Riproducibilità**: chiunque rigenera notebook identici.
- **No metadata casuali**: kernel locale, output cache non committati.

```bash
python scripts/build_notebooks.py
jupyter lab notebooks/
```

I notebook **importano** funzioni dai moduli `src/fraud_pipeline/`: niente duplicazione di logica. Sono "pagine eseguibili" della documentazione.

## 8. Test

`tests/` contiene smoke test minimi:

- `test_features.py`: feature engineer produce le colonne attese, non muta l'input, non leak-a sul cliente.
- `test_inference.py`: `predict_fraud` accetta dict/DataFrame, gestisce threshold, alignment colonne.

Eseguibili con:

```bash
pytest tests/ -v
```

I test che richiedono il modello reale (`test_predict_fraud_with_real_model`) sono `pytest.mark.skipif` se `best_model.joblib` non esiste — non bloccano lo sviluppo locale.

## 9. CI/CD: GitHub Actions

`.github/workflows/docs.yml` builda il sito MkDocs Material e lo pubblica su GitHub Pages a ogni push su `main`.

Tre job sequenziali:

1. **build**: `mkdocs build --strict` (fallisce se ci sono link rotti).
2. **upload**: artifact `./site`.
3. **deploy**: `actions/deploy-pages@v4`.

Trigger: `push` a `main` o `workflow_dispatch` (manuale).

## 10. Riproducibilità

Tre garanzie:

1. **Stesso seed**: `RANDOM_SEED = 42` in `config.py`, propagato a sklearn.
2. **Stesse versioni**: `pyproject.toml` pinna a range minor (es. `numpy>=2.0,<3.0`).
3. **Schema check**: `data._validate_schema` fallisce esplicitamente se i CSV cambiano forma.

Risultato: `fraud-train --quick` produce metriche identiche su esecuzioni successive (entro tolleranza float64 dovuta al thread parallelism `n_jobs=-1`; per riproducibilità bit-perfect impostare `n_jobs=1`).

## 11. Estensioni future

In ordine di valore aggiunto:

- [ ] **Target encoding** per `merchant` (con prior bayesiano e validation set).
- [ ] **Calibrazione probabilità** via `CalibratedClassifierCV` (`isotonic` o `sigmoid`).
- [ ] **API REST** con FastAPI + Docker per inferenza online.
- [ ] **Drift detection** (KS-test mensile sulla distribuzione di score).
- [ ] **Monitoring dashboard** (Streamlit/Grafana).
- [ ] **Walk-forward sliding window** in alternativa a `TimeSeriesSplit` espandente.
- [ ] **SHAP values** per explanability per-transazione.
- [ ] **Ensemble stacking** LogReg + RF con meta-learner LogReg.
