"""Inferenza end-to-end su nuove transazioni.

Espone `predict_fraud(transactions)` come da specifica del project work
(restituisce probabilita' di frode + decisione binaria).

L'API supporta:
    - Singola transazione come `dict`.
    - Batch come `pd.DataFrame`.
    - File CSV via la CLI `fraud-predict --input file.csv --output preds.csv`.

Il modello viene caricato da disco una sola volta tramite cache LRU.
La soglia decisionale e' letta da `threshold.json` salvato accanto al
modello (se presente), altrimenti si usa `DEFAULT_DECISION_THRESHOLD`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import DEFAULT_DECISION_THRESHOLD, MODELS_DIR

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
DEFAULT_THRESHOLD_PATH: Path = MODELS_DIR / "threshold.json"


@lru_cache(maxsize=4)
def _load_model(model_path: str) -> Any:
    """Carica modello serializzato (cache LRU per evitare I/O ripetuti).

    `lru_cache` accetta solo argomenti hashable: per questo riceve `str`,
    non `Path`.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modello non trovato a {path}. "
            "Esegui prima `fraud-train` per addestrare e serializzare il modello."
        )
    logger.info("Carico modello da %s", path)
    return joblib.load(path)


def _load_threshold(threshold_path: Path = DEFAULT_THRESHOLD_PATH) -> float:
    """Carica la soglia ottimale salvata, fallback al default."""
    if threshold_path.exists():
        try:
            data = json.loads(threshold_path.read_text())
            t = float(data.get("threshold", DEFAULT_DECISION_THRESHOLD))
            return t
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Impossibile leggere threshold da %s (%s), uso default.",
                           threshold_path, exc)
    return DEFAULT_DECISION_THRESHOLD


def _expected_columns(model: Any) -> list[str] | None:
    """Estrae i nomi colonna attesi dal modello serializzato.

    La pipeline e' `(feature_engineer, preprocessor, model)`. Il primo
    step espone `feature_names_in_` con le colonne del DataFrame
    pre-feature-engineering: quelle sono le colonne che l'utente deve
    fornire (anche se non tutte: le mancanti vengono imputate dal
    preprocessor interno).
    """
    try:
        first_step = list(model.named_steps.values())[0]
        return list(first_step.feature_names_in_)
    except (AttributeError, IndexError):
        return None


def _align_to_expected(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Riempie le colonne mancanti con NaN, scarta le extra, riordina.

    Per le colonne mancanti l'imputer della pipeline sostituira' i NaN
    con il valore appropriato (mediana per numeriche, "unknown" per
    nominali). `predict_fraud` puo' essere chiamata con un input parziale.
    """
    return df.reindex(columns=expected)


def predict_fraud(
    transactions: dict[str, Any] | pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    threshold: float | None = None,
    threshold_path: Path = DEFAULT_THRESHOLD_PATH,
) -> dict | list[dict]:
    """Predice probabilita' di frode + decisione su una o piu' transazioni.

    Args:
        transactions: dict (singola) o DataFrame (batch). Le colonne
            attese sono quelle del CSV Kaggle (`amt`, `category`,
            `trans_date_trans_time`, `lat`, `long`, `merch_lat`,
            `merch_long`, `cc_num`, `dob`, ...). Le colonne mancanti
            vengono imputate.
        model_path: path al `best_model.joblib`.
        threshold: soglia decisionale. Se None, viene letta da `threshold_path`.

    Returns:
        Dict per input singolo:
            {
                "fraud_probability": float in [0,1],
                "is_fraud": bool,
                "threshold": float
            }
        Lista di dict per batch.
    """
    # Validazione tipo BEFORE load model: cosi' un input invalido non
    # tenta nemmeno il caricamento del .joblib (errore piu' diagnostico).
    if isinstance(transactions, dict):
        df = pd.DataFrame([transactions])
        is_single = True
    elif isinstance(transactions, pd.DataFrame):
        df = transactions.copy()
        is_single = False
    else:
        raise TypeError(
            f"transactions deve essere dict o DataFrame, ricevuto "
            f"{type(transactions).__name__}."
        )

    model = _load_model(str(model_path))
    if threshold is None:
        threshold = _load_threshold(threshold_path)

    expected = _expected_columns(model)
    if expected is not None:
        df = _align_to_expected(df, expected)

    probabilities = model.predict_proba(df)[:, 1]
    decisions = (probabilities >= threshold).astype(bool)

    results = [
        {
            "fraud_probability": float(p),
            "is_fraud": bool(d),
            "threshold": float(threshold),
        }
        for p, d in zip(probabilities, decisions, strict=True)
    ]
    if is_single:
        return results[0]
    return results


def example_transaction() -> dict[str, Any]:
    """Esempio realistico di input per smoke-test della funzione `predict_fraud`."""
    return {
        "trans_date_trans_time": "2020-06-15 14:32:00",
        "cc_num": 4992346398478123,
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "shopping_pos",
        "amt": 75.42,
        "first": "John",
        "last": "Doe",
        "gender": "M",
        "street": "123 Main St",
        "city": "Springfield",
        "state": "IL",
        "zip": "62701",
        "lat": 39.78,
        "long": -89.65,
        "city_pop": 116250,
        "job": "Engineer",
        "dob": "1985-07-22",
        "trans_num": "abc123",
        "unix_time": 1592231520,
        "merch_lat": 39.85,
        "merch_long": -89.50,
    }


# --- CLI fraud-predict ---

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predice frodi su un CSV di transazioni.",
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="CSV di transazioni in input (formato Kaggle).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="CSV in output con colonne fraud_probability, is_fraud.",
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help="Path al modello .joblib (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Soglia decisionale (default: caricata da threshold.json).",
    )
    return parser


def main_predict() -> int:
    """Entry point CLI `fraud-predict`."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = _build_arg_parser().parse_args()
    if not args.input.exists():
        logger.error("File non trovato: %s", args.input)
        return 1

    df = pd.read_csv(args.input, low_memory=False)
    logger.info("Carico %d transazioni da %s", len(df), args.input)
    preds = predict_fraud(df, model_path=args.model, threshold=args.threshold)
    out_df = pd.DataFrame(preds)
    out_df.to_csv(args.output, index=False)
    logger.info("Scritte %d predizioni in %s", len(out_df), args.output)
    return 0


__all__ = [
    "predict_fraud",
    "example_transaction",
    "main_predict",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_THRESHOLD_PATH",
]
