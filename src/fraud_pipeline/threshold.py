"""Ottimizzazione della soglia decisionale su matrice di costi.

Sui problemi sbilanciati la soglia 0.5 e' quasi sempre subottimale: i
modelli (specie con `class_weight='balanced'`) producono distribuzioni di
probabilita' che non corrispondono alla soglia massima di un'utility
business. Qui calcoliamo la soglia che minimizza il costo atteso, dato:

    cost(t) = FN(t) * cost_FN + FP(t) * cost_FP

con `cost_FN >> cost_FP` (perdere una frode costa piu' che bloccare una
transazione legittima).

API principale:
    optimal_threshold_by_cost(y_true, y_proba, cost_FN, cost_FP) -> float
    threshold_sweep(y_true, y_proba) -> DataFrame (precision, recall, F1, cost) per ogni t
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from .config import (
    DEFAULT_COST_FALSE_NEGATIVE,
    DEFAULT_COST_FALSE_POSITIVE,
    DEFAULT_COST_TRUE_NEGATIVE,
    DEFAULT_COST_TRUE_POSITIVE,
)

logger = logging.getLogger(__name__)


# Numero di soglie campionate quando non si usa la curva PR di sklearn.
DEFAULT_THRESHOLD_GRID_SIZE: int = 200


@dataclass(frozen=True)
class CostMatrix:
    """Matrice dei costi per una decisione fraud/legit.

    Convenzione (in unita' monetarie):
        - cost_fn: costo di mancare una frode (e' la perdita media).
        - cost_fp: costo di bloccare una transazione legittima (chargeback,
          customer experience, lavoro umano sulla revisione).
        - cost_tn / cost_tp: di solito zero o piccoli (verifica automatica).
    """
    cost_fn: float = DEFAULT_COST_FALSE_NEGATIVE
    cost_fp: float = DEFAULT_COST_FALSE_POSITIVE
    cost_tn: float = DEFAULT_COST_TRUE_NEGATIVE
    cost_tp: float = DEFAULT_COST_TRUE_POSITIVE

    @property
    def fn_to_fp_ratio(self) -> float:
        return self.cost_fn / max(self.cost_fp, 1e-9)


def _validate_proba(y_proba: np.ndarray) -> np.ndarray:
    """Schiaccia in [0,1] e verifica forma."""
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim == 2:
        # Probabilita' di classe 1 (positiva).
        y_proba = y_proba[:, 1]
    return np.clip(y_proba, 0.0, 1.0)


def expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost: CostMatrix,
) -> float:
    """Costo atteso totale dato il vettore di predizioni binarie."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (
        cost.cost_fn * fn + cost.cost_fp * fp
        + cost.cost_tn * tn + cost.cost_tp * tp
    )


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Iterable[float] | None = None,
    cost: CostMatrix | None = None,
) -> pd.DataFrame:
    """Per ogni soglia, calcola precision/recall/F1/F2 e costo atteso.

    Returns:
        DataFrame con righe ordinate per soglia crescente.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = _validate_proba(y_proba)
    if cost is None:
        cost = CostMatrix()
    if thresholds is None:
        # Sample uniforme da 0.01 a 0.99.
        thresholds = np.linspace(0.01, 0.99, DEFAULT_THRESHOLD_GRID_SIZE)

    rows: list[dict] = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
        c = expected_cost(y_true, y_pred, cost)
        rows.append({
            "threshold": float(t),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2,
            "expected_cost": c,
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def optimal_threshold_by_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost: CostMatrix | None = None,
) -> tuple[float, float]:
    """Trova la soglia che minimizza il costo atteso.

    Usa direttamente i breakpoint di `precision_recall_curve` (ognuno
    corrisponde a un'effettiva configurazione di confusion matrix), molto
    piu' efficiente di una griglia uniforme.

    Returns:
        (best_threshold, best_cost)
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = _validate_proba(y_proba)
    cost = cost or CostMatrix()

    _precision, _recall, thresholds = precision_recall_curve(y_true, y_proba)
    # `thresholds` ha len(N-1) rispetto a precision/recall; aggiungiamo 0 per
    # coprire il caso "tutti positivi".
    thresholds = np.concatenate([[0.0], thresholds])

    # Vettorizzato: per ogni soglia calcola FN, FP via cumsum (efficiente
    # su milioni di righe).
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    proba_sorted = y_proba[order]

    n_pos_total = y_sorted.sum()
    n_neg_total = len(y_sorted) - n_pos_total

    # Per ogni soglia t, "predetti positivi" sono i record con proba >= t,
    # ovvero i primi k del sorted (dove k = numero di proba_sorted >= t).
    # Costruiamo la curva direttamente lungo k = 0..N.
    cum_tp = np.concatenate([[0], np.cumsum(y_sorted)])               # TP after k positives
    cum_fp = np.concatenate([[0], np.arange(1, len(y_sorted) + 1) - np.cumsum(y_sorted)])
    fn_curve = n_pos_total - cum_tp
    fp_curve = cum_fp
    tp_curve = cum_tp
    tn_curve = n_neg_total - cum_fp
    cost_curve = (
        cost.cost_fn * fn_curve + cost.cost_fp * fp_curve
        + cost.cost_tn * tn_curve + cost.cost_tp * tp_curve
    )
    best_k = int(np.argmin(cost_curve))
    if best_k == 0:
        # Nessun positivo predetto. Soglia = max(proba) + epsilon.
        best_threshold = float(proba_sorted[0]) + 1e-6 if len(proba_sorted) else 1.0
    elif best_k >= len(proba_sorted):
        best_threshold = 0.0
    else:
        # La soglia che separa best_k record positivi dal resto:
        # qualunque valore in [proba_sorted[best_k], proba_sorted[best_k-1]).
        best_threshold = float((proba_sorted[best_k - 1] + proba_sorted[best_k]) / 2.0)
    best_cost = float(cost_curve[best_k])
    logger.info(
        "Soglia ottimale (cost-based): t=%.4f, cost=%.2f (TP=%d FP=%d FN=%d TN=%d)",
        best_threshold, best_cost, tp_curve[best_k], fp_curve[best_k],
        fn_curve[best_k], tn_curve[best_k],
    )
    return best_threshold, best_cost


def optimal_threshold_by_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """Soglia che massimizza F1. Alternativa cost-free se non si ha il costo."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = _validate_proba(y_proba)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Drop l'ultimo punto (recall=0, precision=1, t=inf).
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def confusion_matrix_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Confusion matrix dato un threshold, restituita come dict per JSON-friendliness."""
    y_pred = (_validate_proba(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


__all__ = [
    "CostMatrix",
    "expected_cost",
    "threshold_sweep",
    "optimal_threshold_by_cost",
    "optimal_threshold_by_f1",
    "confusion_matrix_at_threshold",
]
