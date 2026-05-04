---
sidebar_position: 6
title: Threshold tuning & costi
description: |
  Decision threshold come iperparametro, matrice costi asimmetrica.
---

# Threshold tuning su matrice di costi

## 1. Il punto: 0.5 è sbagliato

Tutti i classificatori probabilistici producono uno score $p \in [0, 1]$. La decisione binaria si ottiene confrontando $p$ con una **soglia** $t$:

$$
\hat{y} = \begin{cases} 1 & \text{se } p \ge t \\ 0 & \text{altrimenti} \end{cases}
$$

`scikit-learn` di default usa $t = 0{,}5$. Su problemi bilanciati con probabilità calibrate, è una scelta sensata.

In **fraud detection**, $t = 0{,}5$ è quasi sempre subottimale per due ragioni convergenti:

1. **Sbilanciamento**: con `class_weight='balanced'` o `scale_pos_weight`, le probabilità non corrispondono più alla prevalenza reale. Un score 0.4 può corrispondere a 2% di probabilità reale di frode.
2. **Costo asimmetrico**: $c_{\text{FN}} \gg c_{\text{FP}}$. Avere recall basso (perdere frodi) costa molto di più di avere precision bassa (falsi allarmi).

Conseguenza: la soglia ottimale è tipicamente più bassa di 0.5 (es. 0.15–0.30), per "sbilanciare" la decisione verso il recall.

## 2. La matrice dei costi

Ogni cella della confusion matrix ha un costo:

|              | Predetto Legit (0) | Predetto Fraud (1) |
|--------------|--------------------|--------------------|
| Reale Legit  | $c_{TN}$           | $c_{FP}$           |
| Reale Fraud  | $c_{FN}$           | $c_{TP}$           |

Convenzione tipica:

- $c_{TN} = 0$ — transazione legittima passata, nessun costo.
- $c_{TP} = 0$ — frode bloccata, nessun costo (anzi, un guadagno: ma per semplicità lo lasciamo a 0 o lo si riassorbe in $c_{FN}$).
- $c_{FP}$ = costo di bloccare una transazione legittima. Stima: $5–$15 fra customer experience, chargeback, lavoro umano sulla revisione.
- $c_{FN}$ = costo di mancare una frode. Stima: importo medio della frode, $80–$200.

Nel modulo `fraud_pipeline.threshold` queste sono in `CostMatrix`:

```python
@dataclass(frozen=True)
class CostMatrix:
    cost_fn: float = 120.0   # frode mancata
    cost_fp: float = 5.0     # legit bloccata
    cost_tn: float = 0.0
    cost_tp: float = 0.0
```

## 3. Costo atteso come funzione della soglia

Per ogni soglia $t$:

$$
\text{Cost}(t) = c_{FN} \cdot FN(t) + c_{FP} \cdot FP(t) + c_{TN} \cdot TN(t) + c_{TP} \cdot TP(t)
$$

dove $FN(t), FP(t), \ldots$ dipendono dalla soglia.

La soglia ottimale è semplicemente:

$$
t^* = \arg\min_t \text{Cost}(t)
$$

### Scelta naive: griglia uniforme

Si campiona $t \in \{0{,}01, 0{,}02, \ldots, 0{,}99\}$ e si calcola `Cost(t)` per ognuno.

```python
thresholds = np.linspace(0.01, 0.99, 200)
costs = [expected_cost(y_true, (proba >= t).astype(int), cost) for t in thresholds]
best_t = thresholds[np.argmin(costs)]
```

Funziona ma è $O(N \cdot G)$ dove $N$ è la dimensione del dataset e $G$ il numero di soglie.

### Scelta efficiente: breakpoint della PR curve

Le soglie "interessanti" sono solo quelle che cambiano la confusion matrix. Sono $N + 1$ in totale (una per ogni record + i bordi). Implementazione efficiente in `threshold.optimal_threshold_by_cost`:

```python
order = np.argsort(-y_proba)        # Sort decreasing
y_sorted = y_true[order]
cum_tp = np.cumsum(y_sorted)        # TP cumulativo
cum_fp = np.arange(1, N+1) - cum_tp # FP cumulativo
fn_curve = total_pos - cum_tp
fp_curve = cum_fp
cost_curve = c_fn * fn_curve + c_fp * fp_curve
best_k = np.argmin(cost_curve)
best_threshold = (proba_sorted[best_k-1] + proba_sorted[best_k]) / 2
```

$O(N \log N)$ per il sort e $O(N)$ per il sweep — gestibile su milioni di righe.

## 4. Visualizzazione del trade-off

Il **threshold sweep** mostra precision, recall, F1, F2 e cost al variare della soglia. Si individuano quattro punti chiave:

- **Soglia di max recall** (vicino a 0): recall ~1, precision crolla.
- **Soglia di max precision** (vicino a 1): pochissimi flag ma quasi tutti veri.
- **Soglia di max F1**: punto di equilibrio simmetrico.
- **Soglia di min cost**: dipende dal rapporto $c_{FN}/c_{FP}$.

Per fraud detection con $c_{FN}/c_{FP} = 24$, la soglia di min cost è generalmente **più bassa** della soglia di max F1, perché privilegiamo il recall.

## 5. Il punto di precisione minima accettabile

In molti deployment reali, c'è un **vincolo di business**: "non possiamo bloccare più di X transazioni legittime al giorno". Questo si traduce in un vincolo di precision (o, equivalentemente, di FP rate).

Il problema diventa:

$$
\max_t \text{Recall}(t) \quad \text{s.t.} \quad \text{Precision}(t) \ge p_0
$$

Soluzione: si scorre la PR curve e si prende il $t$ con recall massimo fra quelli con precision $\ge p_0$.

```python
mask = precision_curve >= 0.5  # vincolo: 50% di precision
best_idx = np.argmax(recall_curve[mask])
best_threshold = threshold_curve[mask][best_idx]
```

## 6. Validation strategy: dove scegliere la soglia?

**Errore comune**: scegliere la soglia sul test set finale.

Questo introduce un mini-leakage: si sta tunando un iperparametro (la soglia) sul set che dovrebbe misurare la generalizzazione.

**Pratica corretta**:

1. Train e tune iperparametri su `train` (con time-series CV).
2. Scelta della soglia su un **validation set interno** al training (es. ultimo 10% del train, cronologicamente).
3. Valutazione finale su `test` con la soglia scelta al passo 2.

In questa pipeline, per semplicità didattica, scegliamo la soglia direttamente sul test set Kaggle. È una scelta accettabile perché il dataset è grande (le stime sono stabili) e per il PW serve confrontare modelli, non deployment in produzione. In una pipeline produzione si dovrebbe invece introdurre un set di validation interno (vedi `pipeline.py` linea ~190 per il commento esplicito).

## 7. Implicazioni operative

### 7.1 Calibrazione vs threshold

Esistono due modi per "spostare" il punto di decisione:

1. **Calibrare le probabilità** (`CalibratedClassifierCV`) → usare ancora soglia 0.5.
2. **Lasciare le probabilità raw** → spostare la soglia.

Sono matematicamente equivalenti per la decisione binaria. Differiscono se:

- Le probabilità servono **a valle** (es. ranker che pesa più transazioni). In quel caso serve la calibrazione.
- Si vuole usare una soglia "a vista d'uomo" comprensibile (es. "blocca se score > 0.5"). In quel caso conviene calibrare.

### 7.2 Multi-soglia per segmento

In deployment avanzati si usa una soglia **per segmento di transazione**:

- Soglia bassa (alta sensibilità) per importi alti, perché un FN costa di più.
- Soglia alta (alta specificità) per importi bassi, dove i falsi allarmi infastidiscono il cliente per niente.

Implementazione: una `dict` `{segment: threshold}` salvata accanto al modello. Vedi `inference._load_threshold` per la struttura estendibile.

### 7.3 Drift della soglia ottimale

Il rapporto $c_{FN}/c_{FP}$ può cambiare nel tempo (es. nuova policy aziendale, costi aggiornati). La soglia va ri-ottimizzata periodicamente, anche senza retrain del modello. Il modello produce score; la soglia traduce score in decisioni — sono entità separate.

## 8. Sintesi del flusso

```
                          Training
                              |
                              ↓
                    [Pipeline.fit(X_train, y_train)]
                              |
              ┌───────────────┴───────────────┐
              |                               |
              ↓                               ↓
    Modello (fitted)              Validation set (interno)
              |                               |
              |                  predict_proba(X_val) → scores
              |                               |
              |                  ┌────────────┴────────────┐
              |                  ↓                         ↓
              |          CostMatrix                 PR curve
              |                  |                         |
              |                  └─────────┬───────────────┘
              |                            ↓
              |             optimal_threshold_by_cost(...)
              |                            |
              |                            ↓
              |              ┌───────  threshold = 0.18  ──┐
              |              ↓                              ↓
              |       Salva model.joblib              Salva threshold.json
              |
              ↓
        Inference: predict_fraud(tx) →
            { fraud_probability, is_fraud, threshold }
```

## 9. Riferimenti

- **Elkan, C.** (2001). *The Foundations of Cost-Sensitive Learning*, IJCAI. Paper canonico sul cost-sensitive thresholding.
- **Provost, F. & Fawcett, T.** (2001). *Robust Classification for Imprecise Environments*, Machine Learning 42(3). ROC convex hull e multi-threshold.
- **Bahnsen, A. C., Aouada, D., Ottersten, B.** (2014). *Example-dependent cost-sensitive logistic regression for credit scoring*, ICMLA. Cost matrix dipendente dall'esempio.
- **scikit-learn user guide**: [Tuning the decision threshold](https://scikit-learn.org/stable/auto_examples/model_selection/plot_tuned_decision_threshold.html).
