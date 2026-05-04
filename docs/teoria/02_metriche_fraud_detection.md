---
layout: default
title: Metriche fraud detection
parent: Teoria
nav_order: 2
math: mathjax
description: >-
  Recall, Precision, F1, F-beta, AUC-PR, MCC: quando usare quale, come
  interpretarle e perché il costo asimmetrico FN/FP cambia tutto.
---

# Metriche per fraud detection
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. La confusion matrix come fondamento

Ogni metrica di classificazione binaria deriva dalla **confusion matrix**:

|                       | Predetto Legit (0) | Predetto Fraud (1) |
|-----------------------|--------------------|--------------------|
| **Reale Legit (0)**   | TN (true negative) | FP (false positive)|
| **Reale Fraud (1)**   | FN (false negative)| TP (true positive) |

Le quattro celle hanno significati di business molto diversi:

- **TP**: frode correttamente identificata e bloccata. Risparmiamo l'importo.
- **FN**: frode mancata. Costo = perdita media su una frode.
- **FP**: transazione legittima bloccata. Costo = chargeback friction + lavoro umano + customer experience.
- **TN**: transazione legittima passata. Outcome neutro.

In fraud detection **`cost(FN) ≫ cost(FP)`** (tipicamente 20–100×). Questo asimmetria è il vero driver delle scelte di metrica e soglia.

## 2. Le tre metriche fondamentali

### 2.1 Recall (sensitivity, TPR)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Frazione di frodi reali che il modello riesce a identificare.

- Recall = 1: il modello becca **tutte** le frodi (ma può avere molti falsi allarmi).
- Recall = 0: il modello non becca nessuna frode.

In fraud detection è la metrica più diretta sull'obiettivo di business: "quante frodi stiamo evitando".

### 2.2 Precision (PPV)

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Frazione di allarmi che corrispondono a frodi vere.

- Precision = 1: ogni allarme è una vera frode (ma magari ne stiamo mancando tante).
- Precision = 0: tutti gli allarmi sono falsi.

In fraud detection misura la **qualità degli allarmi**: precision bassa = troppe transazioni legittime bloccate = clienti irritati.

### 2.3 F1-score (media armonica)

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Compromesso fra le due. Il massimo si ottiene quando precision = recall.

!!! warning "F1 da' uguale peso a precision e recall"
    Se in fraud detection `cost(FN) ≫ cost(FP)`, F1 sotto-pesa il recall. La generalizzazione corretta è F-beta.

## 3. F-beta: dare peso esplicito al recall

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

- $\beta = 1$: equivalente a F1.
- $\beta = 2$: il recall pesa 4× la precision (F2). Consigliato per fraud detection.
- $\beta = 0{,}5$: la precision pesa 4× il recall (F0.5). Usato in spam filtering, dove FP costa molto.

```python
from sklearn.metrics import fbeta_score
f2 = fbeta_score(y_true, y_pred, beta=2.0)
```

## 4. AUC-PR: la metrica primaria

L'**Area Under the Precision-Recall Curve** misura la qualità del ranking del modello indipendentemente da una soglia specifica.

$$
\text{AUC-PR} = \int_0^1 \text{Precision}(\text{Recall}) \, d(\text{Recall})
$$

`scikit-learn` la calcola via `average_precision_score`:

```python
from sklearn.metrics import average_precision_score
ap = average_precision_score(y_true, y_proba)
```

### Range e baseline

- **Random classifier**: AUC-PR ≈ prevalenza della classe positiva.
- **Modello perfetto**: AUC-PR = 1.

Su Kaggle Fraud (~0.5% frodi), un AUC-PR > 0,3 è già un buon segnale; > 0,7 è eccellente.

## 5. Costo atteso: la metrica di business

Per problemi con costo asimmetrico, la vera funzione obiettivo è il **costo atteso**:

$$
\mathbb{E}[\text{cost}] = c_{\text{FN}} \cdot FN + c_{\text{FP}} \cdot FP + c_{\text{TN}} \cdot TN + c_{\text{TP}} \cdot TP
$$

Con valori indicativi $c_{\text{FN}} = 120$ (perdita media frode), $c_{\text{FP}} = 5$ (chargeback friction), $c_{\text{TN}} = c_{\text{TP}} = 0$.

Il vantaggio: questa metrica:

1. Tiene conto dell'asimmetria.
2. È in unità monetarie comprensibili al business.
3. Permette di scegliere la **soglia ottimale** in modo principled (vedi [Threshold tuning](06_threshold_tuning_e_costi.md)).

## 6. AUC-ROC: usare con cautela

$$
\text{AUC-ROC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})
$$

Su dataset bilanciati è eccellente. Su dataset sbilanciati come fraud (0,5% positivi):

- L'AUC-ROC tende a essere **artificialmente alta** perché il denominatore `FPR = FP / (FP + TN)` ha `TN` enorme.
- È poco sensibile alle differenze fra modelli quando entrambi sono "ragionevoli" sul ranking globale.

Si riporta comunque per completezza ma **non è la metrica primaria**.

## 7. Matthews Correlation Coefficient (MCC)

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

Range $[-1, +1]$: -1 = predizione invertita, 0 = casuale, +1 = perfetto. Più stabile di F1 quando le classi sono molto sbilanciate, perché coinvolge tutte e quattro le celle della confusion matrix. È usato come tie-breaker quando AUC-PR fra modelli è simile.

## 8. Tabella riassuntiva: quale metrica per quale scopo

| Scopo                                    | Metrica primaria | Note                                            |
|------------------------------------------|------------------|-------------------------------------------------|
| Confronto modelli (ranking globale)      | **AUC-PR**       | Indipendente dalla soglia                       |
| Decisione "quanto blocco a soglia $t$?"  | Precision @ $t$  | "Quanti dei blocchi sono veri?"                 |
| Decisione "quante frodi prendo?"         | Recall @ $t$     | "Quante frodi su tutte intercetto?"             |
| Confronto fra modelli sbilanciato        | F2 (recall-leaning) | Rispecchia priorita' fraud detection         |
| Decisione finale di business             | Expected cost    | In dollari, comprensibile a tutti               |
| Tie-break o sanity check                 | MCC              | Robusto sui 4 quadranti                         |
| Comunicazione esecutiva                  | Recall + Precision | Due numeri, intuitivi                         |
| **Da non usare come primaria**           | Accuracy, AUC-ROC| Fuorvianti su dataset sbilanciati               |

## 9. Diagnostica: leggere errori, non solo numeri

Le metriche aggregate non bastano. Bisogna **diagnosticare gli errori** del modello:

- **Distribuzione delle FN**: importi piccoli? Categorie specifiche? Orari notturni? L'analisi rivela pattern non catturati e suggerisce nuove feature.
- **Distribuzione delle FP**: tipi di transazioni "border-line" del cliente? Possono indicare bisogno di una soglia per-customer.
- **Score distribution per classe**: se le distribuzioni di `predict_proba` sulle classi 0 e 1 sono ben separate, il modello discrimina; se sono sovrapposte, c'è poco segnale (o feature insufficienti).

Vedi `notebooks/04_threshold_tuning_and_errors.ipynb` per l'implementazione concreta.

## 10. Riferimenti

- **Saito, T. & Rehmsmeier, M.** (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLoS ONE 10(3).
- **Powers, D. M. W.** (2011). *Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation*, JMLT 2(1).
- **Chicco, D. & Jurman, G.** (2020). *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation*, BMC Genomics 21(6).
- **scikit-learn user guide**: [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
