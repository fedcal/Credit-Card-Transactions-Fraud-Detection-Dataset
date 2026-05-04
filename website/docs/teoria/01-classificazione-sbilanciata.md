---
sidebar_position: 1
title: Classificazione sbilanciata
description: |
  Class imbalance, accuracy paradox, strategie: weighting, sampling, threshold.
---

# Classificazione sbilanciata

## 1. Il problema

Nel dataset Kaggle Fraud Detection, su circa 1,5 milioni di transazioni solo lo **0,5%** è etichettato come frode. Questo è il regime di **classificazione altamente sbilanciata**: la classe positiva (frode) è 200 volte più rara di quella negativa.

In questa situazione, modelli e metriche standard si comportano in modo controintuitivo. Bisogna riformulare l'intera pipeline — dal training, alla validazione, alla scelta della soglia decisionale.

## 2. Perché l'accuracy non basta

L'accuracy è la frazione di predizioni corrette:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Su un dataset con 0,5% di frodi, un classificatore costante che predice **sempre `0` (legit)** ottiene:

$$
\text{Accuracy} = \frac{0 + 0{,}995 \cdot N}{N} = 99{,}5\%
$$

Sembra ottimo, ma il **recall sulle frodi è zero**. Tutte le frodi vengono mancate. Per questo l'accuracy è una metrica fuorviante (e silenziosamente disastrosa) sui dataset sbilanciati.

:::warning Regola pratica
Quando la classe positiva ha prevalenza ≪ 50%, l'accuracy va **sempre** ignorata. Sostituiscila con AUC-PR + recall + precision sulla classe minoritaria.
:::

## 3. La curva PR e l'AUC-PR

La curva **Precision-Recall** è il vero strumento diagnostico per problemi sbilanciati. Per ogni soglia $t \in [0, 1]$ si calcola:

$$
\text{Precision}(t) = \frac{TP(t)}{TP(t) + FP(t)}, \qquad
\text{Recall}(t) = \frac{TP(t)}{TP(t) + FN(t)}
$$

L'**AUC-PR** (Area Under the PR Curve, anche detta Average Precision) è l'integrale della curva: una sintesi numerica della qualità del ranking del modello.

### Differenza fondamentale rispetto all'AUC-ROC

L'AUC-ROC misura la separazione fra le distribuzioni di score della classe positiva e negativa. Per un dataset bilanciato è ottima. Su un dataset estremamente sbilanciato, invece:

- L'AUC-ROC **resta ottimisticamente alta** anche se il modello è quasi inutile in pratica, perché il denominatore della specificità (numero di negativi) è enorme.
- L'AUC-PR **rispecchia la difficoltà reale**: con prevalenza 0,5%, un classificatore casuale ha AUC-PR ≈ 0,005, non 0,5.

:::info Esempio — Perché AUC-ROC inganna su frodi
Modello A: AUC-ROC = 0,98 → sembra eccellente.
AUC-PR = 0,12 → solo il 12% delle predizioni positive sono vere frodi alla soglia ottimale.
Su 100.000 transazioni reali, blocchi 8.000 transazioni e ne becchi 80 vere frodi. Inutilizzabile in produzione.
:::

### Calcolo

`scikit-learn` fornisce direttamente:

```python
from sklearn.metrics import average_precision_score
ap = average_precision_score(y_true, y_proba)
```

Internamente è equivalente a `auc(recall, precision)` calcolato sui punti della `precision_recall_curve`.

## 4. Tre strategie di gestione

Ci sono tre approcci ortogonali (combinabili) per mitigare lo sbilanciamento.

### 4.1 Class weighting

Assegnare pesi al loss function inversamente proporzionali alla frequenza:

$$
w_c = \frac{N}{K \cdot N_c}
$$

dove $N$ è la dimensione del dataset, $K$ il numero di classi, $N_c$ il numero di esempi della classe $c$. In sklearn:

```python
LogisticRegression(class_weight="balanced")
RandomForestClassifier(class_weight="balanced_subsample")
XGBClassifier(scale_pos_weight=N_negativi / N_positivi)
```

**Pro**: zero costo aggiuntivo, nessuna copia del dataset, integrato nel training.
**Contro**: cambia la calibrazione (le probabilità non corrispondono più alla prevalenza reale).

### 4.2 Resampling

Tre varianti:

- **Random oversampling**: duplica esempi della classe minoritaria. Rischio: overfit ai pochi esempi positivi.
- **Random undersampling**: elimina esempi della classe maggioritaria. Rischio: butta segnale informativo.
- **SMOTE** (*Synthetic Minority Oversampling Technique*): genera esempi sintetici interpolando fra k-NN della classe minoritaria nello spazio delle feature.

```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

**Pro di SMOTE**: aggiunge varietà, non solo duplicati.
**Contro**: lento su dataset grandi (1,5M righe), e gli esempi sintetici in alto-dimensionale possono finire in zone "improbabili" dello spazio.

:::note La nostra scelta
In questa pipeline usiamo `class_weight='balanced'` (e `scale_pos_weight` per XGBoost) come strategia primaria. È **scalabile**, **veloce** e **non altera la distribuzione dei dati**. SMOTE è valutato come ablation nel notebook 03.
:::

### 4.3 Threshold tuning

A differenza delle precedenti due strategie (che agiscono sul training), questa agisce **dopo** che il modello ha prodotto le probabilità. La soglia 0,5 di default è quasi sempre subottimale: si sceglie la soglia ottimale post-hoc su un set di validazione (vedi [pagina dedicata](06-threshold-tuning-costi.md)).

## 5. Effetti sulla validazione

Lo sbilanciamento ha conseguenze anche sulla **strategia di cross-validation**:

- **`StratifiedKFold`** (e non `KFold`): garantisce che ogni fold abbia la stessa proporzione di positivi. Senza stratificazione, su 5 fold si rischia di averne uno con zero positivi e metriche `NaN`.
- Per i problemi temporali (come Fraud), **`TimeSeriesSplit`** prevale comunque sulla stratificazione (vedi [Split temporale](04-split-temporale-leakage.md)).

## 6. Sintesi

| Aspetto             | Pratica corretta                                         | Anti-pattern                |
|---------------------|----------------------------------------------------------|-----------------------------|
| Metrica primaria    | AUC-PR (Average Precision)                               | Accuracy                    |
| Metrica secondaria  | Recall, Precision, F1, F2 sulla classe positiva          | AUC-ROC isolata             |
| Loss in training    | `class_weight='balanced'` o `scale_pos_weight`           | Loss non pesata             |
| CV split            | StratifiedKFold (o TimeSeriesSplit per dati temporali)   | KFold semplice              |
| Soglia decisionale  | Ottimizzata post-hoc su matrice costi                    | Default 0,5                 |

## 7. Riferimenti

- **He, H. & Garcia, E. A.** (2009). *Learning from Imbalanced Data*, IEEE TKDE 21(9). Survey classico.
- **Saito, T. & Rehmsmeier, M.** (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLoS ONE 10(3). Argomento per AUC-PR vs AUC-ROC.
- **Chawla et al.** (2002). *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR 16. Paper originale di SMOTE.
- **scikit-learn user guide**: [Imbalanced classes](https://scikit-learn.org/stable/modules/svm.html#unbalanced-problems), [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html).
