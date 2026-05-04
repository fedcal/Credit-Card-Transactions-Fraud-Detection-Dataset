---
sidebar_position: 1
title: Introduzione
description: |
  Pipeline ML per fraud detection con classificazione sbilanciata, feature temporali, ottimizzazione della soglia di decisione.
slug: /intro
---

# Credit Card Fraud — ML Pipeline

Pipeline ML didattica e **production-friendly** per fraud detection: dataset Kaggle credit card, feature engineering temporale e geo, gestione di un imbalance estremo (0.5% frodi), modelli supervisionati (LogReg, RandomForest, XGBoost) con `class_weight='balanced'`, ottimizzazione della soglia su matrice costi asimmetrica.

:::tip In una riga
*Da transazioni grezze a `predict_fraud()` con feature temporali, gestione sbilanciamento e soglia di decisione ottimizzata sui costi.*
:::

## Repository GitHub

| Item | Link |
|---|---|
| Repo | [`fedcal/Credit-Card-Transactions-Fraud-Detection-Dataset`](https://github.com/fedcal/Credit-Card-Transactions-Fraud-Detection-Dataset) |
| Documentazione | [https://fedcal.github.io/Credit-Card-Transactions-Fraud-Detection-Dataset/](https://fedcal.github.io/Credit-Card-Transactions-Fraud-Detection-Dataset/) |
| Licenza | MIT |
| Stack docs | Docusaurus 3 + TypeScript + KaTeX |

## Mappa della documentazione

### [Teoria](/docs/category/teoria)

1. [Classificazione sbilanciata](./teoria/01-classificazione-sbilanciata.md) — Class imbalance, accuracy paradox, strategie: weighting, sampling, threshold.
2. [Metriche per fraud detection](./teoria/02-metriche-fraud-detection.md) — Precision, recall, F1, F-beta, ROC-AUC, PR-AUC: cosa misurano e quando preferire una all'altra.
3. [Feature engineering temporali](./teoria/03-feature-engineering-temporali.md) — Time of day, day of week, time-since-last, geo features.
4. [Split temporale & leakage](./teoria/04-split-temporale-leakage.md) — Time-aware split, prevenzione del leakage in transazioni temporali.
5. [Modelli supervisionati](./teoria/05-modelli-supervisionati.md) — LogReg, RandomForest, XGBoost: pro/contro per fraud detection.
6. [Threshold tuning & costi](./teoria/06-threshold-tuning-costi.md) — Decision threshold come iperparametro, matrice costi asimmetrica.

### [Scelte tecniche](/docs/category/scelte-tecniche)

- [Architettura del progetto](./scelte-tecniche/architettura.md) — Moduli fraud_pipeline/, flusso dati, CLI fraud-train e fraud-predict.
- [Scelte di modellazione: razionale](./scelte-tecniche/scelte-modello.md) — Razionale LogReg+RF+XGB, class_weight, soglia, gestione drift.

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti: [**federicocalo.dev**](https://federicocalo.dev).
