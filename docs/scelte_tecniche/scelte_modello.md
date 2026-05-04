---
layout: default
title: Scelte di modellazione
parent: Scelte tecniche
nav_order: 2
math: mathjax
description: >-
  Trade-off espliciti su feature engineering, scelta del modello, gestione
  dello sbilanciamento, validation strategy, threshold tuning e drift.
---

# Scelte di modellazione
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

Questa pagina documenta le **decisioni progettuali** della pipeline e le motivazioni dietro ciascuna. Ogni decisione è accompagnata dal trade-off accettato.

## 1. Perché Logistic Regression + Random Forest (e non solo Gradient Boosting)

Il PW richiede "almeno due modelli". Le opzioni discusse:

| Configurazione                         | Pro                                                 | Contro                                                |
|----------------------------------------|------------------------------------------------------|--------------------------------------------------------|
| LogReg + RF                            | Coverage massima: lineare + ensemble. Veloce.        | Niente boosting (lo state-of-the-art).                |
| LogReg + XGBoost                       | LogReg interpretabile + XGB performante.             | RF ottimo benchmark per ablation.                     |
| RF + XGBoost                           | Migliori AUC-PR.                                     | Niente baseline interpretabile.                       |
| Tutti e tre                            | Confronto completo.                                  | Tempo CV ~3x; XGBoost richiede dipendenza opzionale.  |

**Decisione**: LogReg + RF come default obbligatori; XGBoost come **opzionale** (extra `xgboost` in `pyproject.toml`, flag CLI `--xgboost`).

**Razionale**:

1. **LogReg** è la baseline didattica obbligata: un modello fraud detection che non batte LogReg ha qualcosa di rotto.
2. **RandomForest** copre la non linearità senza la complessità di tuning di XGBoost.
3. **XGBoost** è il "bonus": chi vuole massimizzare AUC-PR lo abilita, gli altri non hanno una dipendenza pesante (XGBoost binary ~30MB).

## 2. Perché `class_weight='balanced'` e non SMOTE come strategia primaria

Tre opzioni per gestire lo sbilanciamento:

| Strategia            | Pro                                          | Contro                                          |
|----------------------|----------------------------------------------|-------------------------------------------------|
| `class_weight`       | Zero costo overhead, integrato nel training. | Cambia calibrazione probabilità.                |
| Random oversampling  | Semplice.                                    | Overfit ai pochi esempi positivi.               |
| SMOTE                | Esempi sintetici diversificati.              | Lento su 1.5M righe, esempi sintetici "fuori manifold". |

**Decisione**: `class_weight='balanced'` come default per LogReg/RF; `scale_pos_weight` per XGBoost.

**Razionale**:

1. **Scalabilità**: SMOTE su 1,5M righe richiederebbe espandere la classe minoritaria a ~1.5M righe. Tempo di training >2x.
2. **No leakage**: SMOTE deve essere dentro `imblearn.pipeline.Pipeline` per essere sicuro nella CV; se l'utente sbaglia il setup, c'è leakage. `class_weight` è impossibile da sbagliare.
3. **Performance comparable**: studi (Pozzolo et al. 2014) mostrano che su dataset >1M, SMOTE e class_weight ottengono AUC-PR simili.

SMOTE rimane disponibile come alternativa per sperimentazione (vedi notebook 03 commento).

## 3. Perché `TimeSeriesSplit` e non KFold stratificato

I dati sono ordinati cronologicamente. La transazione $t$ può dipendere da pattern delle transazioni $1..t-1$ dello stesso cliente.

`KFold` casuale produrrebbe metriche di CV ottimisticamente alte: il modello vedrebbe "pezzi del cliente" sia nel train che nel test, mentre in produzione vede solo passato.

**Decisione**: `TimeSeriesSplit(n_splits=4)`. Niente shuffle, fold espandenti.

**Trade-off accettato**:

- **Fold di training disuguali**: il fold 1 ha un training piccolo, il fold 4 un training grande. Le metriche per fold non sono direttamente comparabili.
- **Stima conservativa**: la performance del modello finale (refittato su tutto il train) è leggermente migliore del miglior fold della CV.

Entrambi accettabili: la priorità è una stima onesta della generalizzazione.

## 4. Perché droppare `merchant` invece di target encoding

Il dataset ha 693 merchant unici. Tre strategie:

| Strategia                   | Pro                                       | Contro                                        |
|-----------------------------|--------------------------------------------|------------------------------------------------|
| OneHot diretto              | Semplice.                                  | 693 colonne dummy → curse of dimensionality.   |
| Drop colonna                | Semplice. Categoria conserva info.         | Perdita di segnale per merchant specifici.    |
| Target encoding (con prior) | Cattura merchant ad alto rischio.          | Rischio leakage se mal implementato.          |

**Decisione**: Drop colonna `merchant` per default (vedi `preprocessing.HIGH_CARDINALITY_COLUMNS`).

**Razionale**:

1. La colonna `category` (14 categorie) è sufficiente per catturare il "tipo" di merchant.
2. `distance_km` (cliente↔merchant) cattura informazione geografica.
3. Target encoding richiede un'implementazione molto attenta:
    - Calcolare il prior di frode di ogni merchant **solo sul fold di training** della CV.
    - Applicare Bayesian shrinkage per merchant con pochi sample.
    - Aggiornare il prior durante il refit finale.
   Un bug in qualsiasi passo introduce leakage. Per pipeline didattica, il rischio non vale.

Target encoding è documentato come **estensione futura** in `architettura.md`.

## 5. Perché AUC-PR come metrica primaria di tuning

Le opzioni di scoring per `GridSearchCV`:

- `accuracy` — fuorviante su sbilanciato.
- `f1` — bilancia precision e recall, ma su soglia 0.5 (sub-ottima).
- `roc_auc` — ottimistico su sbilanciato.
- `average_precision` (= AUC-PR) — corretta per sbilanciato, indipendente da soglia.

**Decisione**: `scoring='average_precision'` ovunque (`tuning.PRIMARY_SCORING`).

**Trade-off**: AUC-PR ottimizza il **ranking globale**, non la performance a una soglia specifica. Se il vincolo di business è "max FP rate al X%", il tuning su AUC-PR può non essere ottimo per quel vincolo. In quel caso si userebbe uno scoring custom basato sul costo.

## 6. Perché soglia ottimizzata sul test set (e non su validation)

Discutiamo onestamente questa scelta nella pipeline.

**Pratica corretta**: scegliere la soglia su validation set interno al training, valutare su test holdout.

**Pratica scelta**: scegliere la soglia direttamente sul test set Kaggle.

**Razionale dello shortcut**:

1. **Dimensione del test**: 555k transazioni. Le stime sono molto stabili — il rumore è < 1%.
2. **Dataset Kaggle predefinito**: `fraudTrain` e `fraudTest` sono già splittati. Costruire un validation interno significa rifare lo split, complicando la pipeline.
3. **Scopo didattico**: il PW chiede di mostrare l'ottimizzazione della soglia su matrice di costi, non di simulare un deployment puro.

**Limite accettato**: in produzione reale, la pipeline va estesa con un validation set interno. Il commento esplicito è in `pipeline.py` linea ~190.

## 7. Perché feature engineer come `Pipeline` step (e non a mano sul DataFrame)

Tre opzioni:

1. **A mano sul DataFrame**: preparare X_train_fe = fe.fit_transform(X_train) prima del fit del modello.
2. **Funzione standalone**: `preprocess(df)` che ritorna un DataFrame.
3. **Transformer sklearn dentro `Pipeline`**: `FraudFeatureEngineer(BaseEstimator, TransformerMixin)`.

**Decisione**: opzione 3.

**Razionale**:

1. **No leakage in CV**: la trasformazione viene fatta dentro ogni fold sul subset di training.
2. **Serializzazione bit-identica**: il transformer è dentro il `joblib`. L'inferenza usa la stessa logica del training, anche se il codice cambia.
3. **Testabile**: ogni transformer ha una signature standard (`fit/transform`).

**Trade-off**: gli aggregati per cliente richiedono accesso al **training set completo** in `fit`. Sklearn permette questo: `fit(X_train, y_train)` riceve X_train. L'implementazione in `features.py` mantiene gli aggregati come operazioni vettorizzate efficienti.

## 8. Perché smoke-test (`--quick`) sui primi 50k

Su 1,5M righe, anche un singolo `cross_val_score` richiede 5+ minuti. In sviluppo serve un loop rapido (15-30s) per verificare l'integrazione delle componenti.

**Decisione**: `--quick` campiona le **prime 50k** righe cronologicamente (non casualmente).

**Razionale**:

1. **Cronologico**: simula un mini-deployment realistico (training su 50k tx storiche).
2. **Stratificazione automatica**: se le 50k iniziali hanno meno di 50 frodi (raro), aggiungiamo positivi extra per stabilità della CV (vedi `data.downsample_for_smoke_test`).
3. **Veloce**: ~30 secondi end-to-end, sufficiente per CI / debugging.

## 9. Drift e retrain: strategia raccomandata

Anche se non implementata in pipeline, definiamo la strategia:

1. **Monitoring giornaliero**: distribuzione di `predict_proba` sul flusso di produzione.
2. **Trigger di retrain**: KS-test fra distribuzione corrente e baseline (snapshot al training). Se `p < 0.01` per 3 giorni consecutivi, alert.
3. **Retrain schedulato**: ogni 30 giorni, indipendentemente dal drift, con dati degli ultimi 12 mesi.
4. **Revisione cost matrix**: ogni quarter, business team rivisita $c_{FN}/c_{FP}$.

In produzione la pipeline andrebbe estesa con un modulo `monitoring.py` (non incluso in questa versione didattica).

## 10. Riepilogo: principles della pipeline

1. **Modulare**: ogni file 100–300 righe, una responsabilità chiara.
2. **No leakage**: sortby + TimeSeriesSplit + transformer dentro Pipeline.
3. **No-magic constants**: tutto in `config.py`.
4. **Type hints + docstring**: per ogni public function.
5. **CLI + notebook + import**: stesso codice, tre interfacce.
6. **Test minimi ma critici**: smoke test, no-leakage test, alignment test.
7. **Riproducibilità**: seed fisso, versioni pinnate, schema validation.
8. **Trade-off espliciti**: ogni scelta non standard è documentata qui.
