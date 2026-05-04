---
layout: default
title: Split temporale & leakage
parent: Teoria
nav_order: 4
math: mathjax
description: >-
  Perché KFold casuale è sbagliato sui dati transazionali, come usare
  TimeSeriesSplit / walk-forward CV, e quali aggregati possono leakare
  informazione futura nei modelli di fraud detection.
---

# Split temporale e data leakage
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Il problema della causalità

Il dataset Kaggle Fraud copre 2 anni (2019–2020). Le transazioni sono ordinate cronologicamente. In **produzione**, il modello viene addestrato su dati passati e usato per predire transazioni future. La validazione deve simulare questo deployment: train sul passato, test sul futuro.

Tre patologie da evitare:

1. **Shuffle casuale prima dello split**: la transazione di domani può finire nel training set, e il modello "impara dal futuro".
2. **Aggregati temporali leak-canti**: feature come "media globale dell'importo per cliente" includono il futuro.
3. **Tuning iperparametri su tutto il dataset**: ogni decisione presa sul dataset completo (es. SMOTE su train+test) porta segnale dal test al modello.

Questa pagina copre la (1) e (3). La (2) è in [Feature engineering temporali](03_feature_engineering_temporali.md).

## 2. Perché KFold casuale è sbagliato sui dati transazionali

`KFold(shuffle=True)` mescola le righe casualmente e crea $K$ fold. Ogni fold viene usato come test, gli altri come train. Su dati IID (campioni indipendenti) è la scelta corretta.

Sui dati **temporali** è errato:

- I fold di test contengono transazioni interleaved con quelle di train. Il modello ha "visto" un pattern del cliente nel train, e nel test prevede una transazione successiva dello **stesso cliente**. La metrica di CV è sovra-stimata.
- La performance osservata in CV non riflette quella in produzione (dove il modello vede solo passato).

## 3. TimeSeriesSplit: la scelta corretta

`sklearn.model_selection.TimeSeriesSplit` produce fold "espandenti":

```
fold 1: |─ train ─|test1|
fold 2: |── train ──|test2|
fold 3: |─── train ───|test3|
fold 4: |──── train ────|test4|
```

Ogni test set è cronologicamente **dopo** il rispettivo training set. Le metriche aggregate sui $K$ fold sono una stima onesta della performance fuori-tempo.

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

cv = TimeSeriesSplit(n_splits=4)
# Importante: NESSUN shuffle. I dati devono gia' essere ordinati per
# trans_date_trans_time prima di chiamare fit/cross_val_score.
scores = cross_val_score(pipeline, X_train, y_train,
                         scoring="average_precision", cv=cv)
```

### Walk-forward con sliding window

Una variante più strict è il **walk-forward con finestra fissa** (training su una finestra di N mesi che scorre):

```
fold 1: |─ 6m train ─|1m test|
fold 2:        |─ 6m train ─|1m test|
fold 3:               |─ 6m train ─|1m test|
```

Sklearn non lo offre direttamente; va implementato con `BaseCrossValidator`. Più realistico per simulare deployment con retrain mensile.

## 4. Test set: il dataset Kaggle ha già lo split

Il dataset Kaggle è già splittato in `fraudTrain.csv` e `fraudTest.csv` con un cutoff temporale fisso. Questo è il nostro **holdout test set out-of-time**: non viene mai usato durante tuning o feature engineering. Le metriche su `fraudTest.csv` sono la stima finale di generalizzazione.

In `pipeline.py`, lo schema è:

```
fraudTrain.csv  (training)
    |
    └── TimeSeriesSplit(n_splits=4)  → tuning iperparametri
    └── refit su tutto fraudTrain   → modello finale
        |
        └── valutazione su fraudTest.csv  → metriche holdout
```

## 5. Tipologie di data leakage

Il leakage può sgattaiolare in molti modi. Tre da conoscere bene:

### 5.1 Target leakage

Una feature contiene direttamente l'informazione del target. Es: `is_fraud_flagged_yesterday` è quasi `is_fraud`. Nel dataset Kaggle non è un problema perché `is_fraud` è l'unica colonna obiettivo, ma in dataset enterprise è il bug più frequente.

### 5.2 Train-test contamination

Statistiche calcolate su `train+test` invece che solo train. Esempio classico: `StandardScaler` fittato sull'intero dataset:

```python
# SBAGLIATO
scaler.fit(X)              # vede media/std del test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# CORRETTO
scaler.fit(X_train)        # vede solo train
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Il modo robusto è usare `Pipeline(steps=[("scaler", StandardScaler()), ("model", ...)])`: sklearn gestisce il `fit` solo sul train interno alla CV.

### 5.3 Leakage temporale

Specifico per dati ordinati nel tempo. Esempio:

- "Probabilità di frode di questo cliente" calcolata come `mean(is_fraud)` su tutto il train. Per il primo fold del walk-forward CV, include tx future.
- "Media degli importi" come `groupby("cc_num").transform("mean")`: include il presente nella propria media (vedi sezione precedente).

## 6. Strategie difensive

| Difesa                                        | Come                                                  |
|-----------------------------------------------|--------------------------------------------------------|
| Sort cronologico esplicito                    | `df.sort_values("trans_date_trans_time")` prima di tutto |
| `TimeSeriesSplit` (no shuffle)                | `cv = TimeSeriesSplit(n_splits=4)`                    |
| `Pipeline` con tutti gli step                 | scaler/encoder/imputer dentro la pipeline             |
| Aggregati con `shift(1) + expanding`          | vedi `features.py` linea 145-185                      |
| Test holdout out-of-time isolato              | `fraudTest.csv` mai toccato fino al final eval         |
| Test no-leakage automatici                    | vedi `tests/test_features.py`                         |

## 7. Concept drift: il problema dopo il deployment

Una volta in produzione, il modello incontra un nuovo nemico: il **concept drift**. Le frodi evolvono — i fraudster cambiano tattica, i pattern temporali si spostano. Un modello addestrato su 2019 potrebbe degradare su 2024.

Strategie di monitoring:

- **Distribuzione score in produzione**: se la distribuzione di `predict_proba` si sposta nel tempo, il modello sta vedendo dati "diversi" da quelli del training (covariate shift).
- **Recall stimato**: tracciare la frazione di frodi confermate (es. via report da clienti) sul totale dei flag. Cala? Drift.
- **Retrain schedulato**: ogni 1–3 mesi, ri-addestrare con dati recenti.

Implementazione di base: salvare in `reports/monitoring/` la distribuzione di score per giorno/settimana e fare alert se KS-test fra distribuzione corrente e baseline diventa significativo.

## 8. Sintesi

| Anti-pattern                                  | Fix                                                   |
|-----------------------------------------------|-------------------------------------------------------|
| `KFold(shuffle=True)`                         | `TimeSeriesSplit(n_splits=K)`                         |
| `train_test_split` casuale                    | Split cronologico (`temporal_train_test_split`)        |
| `.groupby(...).transform("mean")` per aggregati storici | `.shift(1).expanding().mean()`              |
| Encoder fittato su tutto il dataset           | Encoder dentro `Pipeline`                              |
| Tuning su test set                            | Tuning su CV interna al training                      |
| Modello statico in produzione                 | Retrain schedulato + monitoring                       |

## 9. Riferimenti

- **Kaufman, S., Rosset, S., Perlich, C.** (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance*, ACM TKDD 6(4).
- **Bergmeir, C. & Benítez, J. M.** (2012). *On the use of cross-validation for time series predictor evaluation*, Information Sciences 191.
- **Sklearn user guide**: [TimeSeriesSplit](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split), [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline).
- **Pozzolo, A. D., et al.** (2018). *Credit card fraud detection: a realistic modeling and a novel learning strategy*, IEEE TNNLS 29(8). Discussione di drift e calibrazione.
