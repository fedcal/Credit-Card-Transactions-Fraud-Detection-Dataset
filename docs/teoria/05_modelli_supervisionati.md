---
layout: default
title: Modelli supervisionati
parent: Teoria
nav_order: 5
math: mathjax
description: >-
  Logistic Regression, Random Forest e Gradient Boosting (XGBoost) a
  confronto: assunzioni, gestione dello sbilanciamento, calibrazione,
  trade-off di interpretabilità.
---

# Modelli supervisionati per fraud detection
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Tre famiglie complementari

Sui dati tabulari di fraud detection, tre famiglie di modelli sono lo standard:

1. **Logistic Regression** — modello lineare regolarizzato. Baseline interpretabile.
2. **Random Forest** — ensemble di alberi (bagging). Cattura non linearità e interazioni.
3. **Gradient Boosting** (XGBoost, LightGBM) — alberi sequenziali su residui. Tipicamente lo stato dell'arte sui tabulari.

In questa pipeline confrontiamo Logistic Regression e Random Forest come modelli obbligatori (Fase 4 della traccia) e includiamo XGBoost come opzionale.

## 2. Logistic Regression

### Modello

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

con loss negativa log-likelihood (cross-entropy):

$$
\mathcal{L} = -\sum_i \left[ y_i \log p_i + (1-y_i) \log (1 - p_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2
$$

dove $C$ è l'inverso della forza di regolarizzazione (in sklearn la convenzione è $\lambda = 1/C$).

### Pro

- **Interpretabile**: ogni coefficiente è il **log-odds ratio** della feature corrispondente. Per gli stakeholder business: "un'unità in più di `is_night` aumenta l'odds di frode di un fattore $e^{w_j}$".
- **Veloce**: `solver='liblinear'` o `'lbfgs'` su 1.5M righe gira in pochi secondi.
- **Probabilità calibrate** quando le feature sono ben scalate.

### Contro

- **Modella solo relazioni lineari** (nello spazio delle feature trasformate). Le interazioni vanno costruite a mano (es. `hour * distance_km`).
- **Non cattura non monotonicità** senza feature derivate (es. picco di rischio a 02:00 con valle alle 09:00).
- Su problemi sbilanciati richiede sempre `class_weight='balanced'` e scaling.

### Iperparametri chiave

| Parametro       | Effetto                                       | Default |
|-----------------|-----------------------------------------------|---------|
| `C`             | Inverso regolarizzazione (basso = piu' regolarizzato) | 1.0   |
| `penalty`       | `l1` / `l2` / `elasticnet`                    | `l2`    |
| `solver`        | `liblinear` (L1+L2), `lbfgs`, `saga`          | `lbfgs` |
| `class_weight`  | `'balanced'` per dataset sbilanciati          | None    |
| `max_iter`      | Tipicamente 1000–5000 con scaling             | 100     |

## 3. Random Forest

### Modello

Un Random Forest è una **media di $T$ decision tree** addestrati su:

1. **Bootstrap sample** del training set (campionamento con rimpiazzo).
2. **Sottoinsieme casuale di feature** ad ogni split (`max_features='sqrt'`).

$$
\hat{p}(y=1 \mid \mathbf{x}) = \frac{1}{T} \sum_{t=1}^T p_t(\mathbf{x})
$$

dove $p_t$ è la frequenza di classe 1 nelle foglie raggiunte dall'albero $t$.

### Pro

- **No assunzioni** sulla forma funzionale: cattura interazioni e non linearità senza feature engineering esplicito.
- **No scaling necessario**: gli alberi sono invarianti per trasformazioni monotone.
- **Robusto agli outlier** e ai dati rumorosi.
- **Feature importance** out-of-the-box.

### Contro

- **Probabilità non calibrate**: tendono ad essere sovrastimate al 0/1 (vedi Calibration sotto).
- **Lento in inference** rispetto a un lineare: $T$ alberi da attraversare per ogni predizione.
- **Memoria**: 400 alberi su 1.5M righe può occupare 200+ MB serializzato.

### Iperparametri chiave

| Parametro             | Effetto                                              | Default      |
|-----------------------|------------------------------------------------------|--------------|
| `n_estimators`        | Numero alberi (piu' = meglio, plateau dopo 200-500)  | 100          |
| `max_depth`           | Profondita' max alberi (None = espande fino a foglia pura) | None  |
| `min_samples_leaf`    | Min sample per foglia (regolarizzazione)             | 1            |
| `max_features`        | Sub-sampling feature per split (`'sqrt'` standard)   | `'sqrt'`     |
| `class_weight`        | `'balanced_subsample'` per dataset sbilanciati       | None         |

### Bagging vs Boosting: lezione concettuale

Il Random Forest è un **bagging**: ogni albero è addestrato indipendentemente sul suo bootstrap. Il bias dell'ensemble è quello dei singoli alberi, ma la varianza si riduce di un fattore $1/T$ (per alberi correlati, più lentamente).

XGBoost è un **boosting**: gli alberi sono addestrati in sequenza, ognuno predicendo i residui del precedente. Riduce sia bias che varianza, ed è tipicamente più potente.

## 4. Gradient Boosting (XGBoost)

### Modello

Funzione obiettivo iterativa:

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})
$$

dove $\eta$ è il learning rate e $h_m$ è un albero che minimizza una loss del second'ordine sui residui di $F_{m-1}$. XGBoost aggiunge regolarizzazione L1/L2 sui pesi delle foglie e tree pruning.

### Pro

- **State-of-the-art** sui tabulari di queste dimensioni.
- **Veloce** con `tree_method='hist'`: discretizza le feature in bin, esegue split su bin invece che valori puntuali.
- **Gestisce nativamente** missing values e categorical encoding (con LightGBM ancora di più).
- **`scale_pos_weight`** = analogo di `class_weight` per fraud.

### Contro

- **Più sensibile al tuning**: 7+ iperparametri rilevanti.
- **Probabilità necessitano calibrazione** se si usa `scale_pos_weight`.
- **Risk overfit** se `learning_rate` alto e `n_estimators` elevato senza early stopping.

### Iperparametri chiave

| Parametro             | Effetto                                              | Range tipico |
|-----------------------|------------------------------------------------------|--------------|
| `n_estimators`        | Numero alberi                                        | 200-1500     |
| `max_depth`           | Profondita' alberi (boost vuole alberi sotti, max=6) | 3-7          |
| `learning_rate` (eta) | Passo di shrinkage (basso + tanti alberi = meglio)   | 0.01-0.1     |
| `subsample`           | Frazione righe per albero (regolarizzazione)         | 0.7-1.0      |
| `colsample_bytree`    | Frazione colonne per albero                          | 0.7-1.0      |
| `reg_lambda`          | L2 sui pesi delle foglie                             | 1-10         |
| `scale_pos_weight`    | Peso classe positiva (~ neg/pos ratio)               | 1-200        |

## 5. Gestione dello sbilanciamento: 3 strategie

### 5.1 `class_weight='balanced'`

Standard per LogReg e RF. Sklearn calcola automaticamente:

$$
w_c = \frac{N}{K \cdot N_c}
$$

I gradienti durante il training sono pesati: la classe rara ha più "voce in capitolo". Per RF, la variante `'balanced_subsample'` ricalcola i pesi su ogni bootstrap (più robusta).

### 5.2 `scale_pos_weight` (XGBoost)

Equivalente per XGBoost. Va impostato a `n_negativi / n_positivi` (~200 sul dataset Kaggle).

### 5.3 Resampling (SMOTE) come alternativa

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline

pipeline = ImblearnPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression()),
])
```

NOTA: SMOTE va dentro la `imblearn.pipeline.Pipeline` (non `sklearn.pipeline.Pipeline`) per garantire che l'oversampling avvenga **solo sul training fold** della CV. Non integrarlo correttamente è una causa comune di leakage.

## 6. Calibrazione delle probabilità

Su modelli con `class_weight='balanced'` o `scale_pos_weight`, le probabilità prodotte da `predict_proba` non corrispondono più alla prevalenza reale. Per esempio, un modello potrebbe restituire 0.5 per una transazione che ha solo 2% di probabilità reale di essere frode.

Conseguenza: la **soglia di default 0.5** è ulteriormente subottimale.

### Calibration plot

Si plotta la frazione di positivi reali per bin di score predetto. Un modello calibrato ha la curva sulla bisettrice. RF con `balanced` tende ad essere underconfident (curve sotto la bisettrice nei bin alti); LogReg con scaling è tipicamente già ben calibrato.

### Correzione: `CalibratedClassifierCV`

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(rf_pipeline, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)
```

`method='isotonic'` (per dataset grandi) o `'sigmoid'` (Platt scaling, per dataset piccoli).

In questa pipeline non applichiamo calibrazione di default (va integrata se l'output di `predict_proba` viene usato come input a un secondo modello). Se l'unico uso è la decisione binaria via soglia, l'**ottimizzazione della soglia** sull'output non calibrato è equivalente.

## 7. Confronto: pro/contro per fraud detection

| Aspetto                          | LogReg          | RandomForest      | XGBoost           |
|----------------------------------|-----------------|-------------------|-------------------|
| AUC-PR atteso (Kaggle Fraud)     | ~0.55           | ~0.85             | ~0.88             |
| Interpretabilita'                | Alta (coefficienti) | Media (importance) | Bassa (SHAP serve) |
| Tempo training (1.5M righe)      | ~10s            | ~3-5min           | ~1-3min           |
| Tempo inferenza (1k righe)       | ~ms             | ~10-50ms          | ~5-20ms           |
| Calibrazione probabilita'        | Buona           | Scarsa            | Scarsa            |
| Robustezza a feature mancanti    | Bassa (richiede imputer) | Alta     | Altissima (nativa) |
| Tuning effort                    | Basso (1 param) | Medio (3-4 param) | Alto (6+ param)   |

## 8. Selezione del modello in pipeline

In `pipeline.py`, il selettore segue una regola semplice: **il modello con miglior AUC-PR sulla cross-validation temporale** vince. Il best model (insieme alla soglia ottimale) viene serializzato come `best_model.joblib` per l'inferenza.

Decisione esplicita: non scegliamo il modello "migliore in produzione" che potrebbe essere il meno performante ma il più interpretabile. Per pipeline di banca dove l'interpretabilità è obbligatoria (es. GDPR, "right to explanation"), si potrebbe forzare la scelta sulla LogReg accettando un AUC-PR più basso.

## 9. Riferimenti

- **Hastie, T., Tibshirani, R., Friedman, J.** *The Elements of Statistical Learning*, Springer, capp. 4 (LogReg), 15 (RF), 10 (Boosting).
- **Chen, T. & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*, KDD.
- **Breiman, L.** (2001). *Random Forests*, Machine Learning 45(1).
- **Niculescu-Mizil, A. & Caruana, R.** (2005). *Predicting Good Probabilities with Supervised Learning*, ICML — calibrazione.
- **scikit-learn user guide**: [Linear models](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), [Ensembles](https://scikit-learn.org/stable/modules/ensemble.html), [Calibration](https://scikit-learn.org/stable/modules/calibration.html).
