---
layout: default
title: Scelte tecniche
nav_order: 3
has_children: true
permalink: /scelte_tecniche/
description: >-
  Decisioni architetturali e di modellazione del progetto Credit Card
  Fraud Detection Pipeline, con trade-off espliciti e razionali documentati.
---

# Scelte tecniche

Questa sezione documenta **come** è costruito il progetto e **perché** ogni
componente è stata progettata in un certo modo. È pensata per chi vuole
estendere o adattare la pipeline a un dominio simile (anti-money laundering,
detection di abusi, classificazione sbilanciata in generale).

## Capitoli

| Capitolo | Titolo | Cosa contiene |
|:--|:--|:--|
| 1 | [Architettura](architettura/) | Moduli `src/fraud_pipeline/`, flusso dati, CLI, dipendenze fra componenti. |
| 2 | [Scelte di modellazione](scelte_modello/) | Selezione famiglie di modelli, gestione sbilanciamento, validation walk-forward, threshold tuning, gestione drift. |

{: .tip }
> Per la teoria sottostante (algoritmi, metriche, leakage temporale) consulta
> la sezione **[Teoria](../teoria/)**.
