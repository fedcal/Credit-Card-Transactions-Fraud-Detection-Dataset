"""Pipeline ML end-to-end per il rilevamento di frodi nelle transazioni con
carta di credito (Kaggle Fraud Detection Dataset, ~1.5M righe, ~0.5% frodi).

API pubbliche principali:

    from fraud_pipeline.pipeline import run_full_pipeline
    from fraud_pipeline.inference import predict_fraud
    from fraud_pipeline.data import load_train_test

Per il dettaglio del flusso e delle scelte tecniche, vedi
`docs/scelte_tecniche/architettura.md` e i notebook in `notebooks/`.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
