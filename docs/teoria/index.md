---
layout: default
title: Teoria
nav_order: 2
has_children: true
permalink: /teoria/
description: >-
  Fondamenti teorici della pipeline Credit Card Fraud Detection:
  classificazione sbilanciata, metriche per fraud detection, feature
  engineering temporali, split temporale & leakage, modelli supervisionati,
  threshold tuning su matrice di costi.
---

# Teoria

I sei articoli di questa sezione costruiscono progressivamente le basi
necessarie per leggere il progetto **Credit Card Fraud Detection Pipeline**
e per ragionare in modo critico sui risultati ottenuti su un problema
altamente sbilanciato e con forte dimensione temporale.

## Percorso consigliato di lettura

| Capitolo | Titolo | Concetti chiave |
|:--|:--|:--|
| 1 | [Classificazione sbilanciata](01_classificazione_sbilanciata/) | Class imbalance, accuracy paradox, AUC-PR, strategie di mitigazione |
| 2 | [Metriche fraud detection](02_metriche_fraud_detection/) | Recall, Precision, F1, F-beta, AUC-PR, MCC, costo asimmetrico FN/FP |
| 3 | [Feature engineering temporali](03_feature_engineering_temporali/) | Decomposizione timestamp, distanza Haversine, aggregati expanding |
| 4 | [Split temporale & leakage](04_split_temporale_e_leakage/) | TimeSeriesSplit, walk-forward CV, anti-pattern temporali |
| 5 | [Modelli supervisionati](05_modelli_supervisionati/) | Logistic Regression, Random Forest, Gradient Boosting, `class_weight` |
| 6 | [Threshold tuning & costi](06_threshold_tuning_e_costi/) | Soglia ottimale, matrice di costi, costo atteso, deployment |

{: .note }
> Ogni capitolo è autocontenuto: leggi nell'ordine se vuoi una progressione
> didattica, oppure salta direttamente al capitolo che ti serve.

## Riferimenti trasversali

- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (2009).
- He, H. & Garcia, E. A. (2009) — *Learning from Imbalanced Data*, IEEE TKDE.
- Saito, T. & Rehmsmeier, M. (2015) — *The Precision-Recall Plot Is More
  Informative than the ROC Plot When Evaluating Binary Classifiers on
  Imbalanced Datasets*, PLoS ONE.
- Bahnsen, A. C. et al. (2016) — *Feature engineering strategies for credit
  card fraud detection*, Expert Systems with Applications.
