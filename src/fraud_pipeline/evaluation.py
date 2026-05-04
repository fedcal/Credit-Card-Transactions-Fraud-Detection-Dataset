"""Metriche, curve PR/ROC, confusion matrix e diagnostica grafica.

Tutte le metriche sono calcolate sulla classe positiva (`is_fraud=1`).
La metrica primaria sui problemi sbilanciati e' l'AUC-PR (Average
Precision); riportiamo anche AUC-ROC come baseline ma con la nota che
e' troppo ottimistica con prevalenza ~0.5%.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import FIGURES_DIR

logger = logging.getLogger(__name__)


@dataclass
class FraudMetrics:
    """Insieme di metriche calcolate alla soglia data."""
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    auc_pr: float
    auc_roc: float
    n_predicted_positive: int
    n_true_positive: int
    n_false_positive: int
    n_false_negative: int
    n_true_negative: int

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
            "auc_pr": self.auc_pr,
            "auc_roc": self.auc_roc,
            "n_predicted_positive": self.n_predicted_positive,
            "n_true_positive": self.n_true_positive,
            "n_false_positive": self.n_false_positive,
            "n_false_negative": self.n_false_negative,
            "n_true_negative": self.n_true_negative,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> FraudMetrics:
    """Calcola un set completo di metriche fraud-detection-oriented.

    Args:
        y_true: etichette binarie.
        y_proba: probabilita' classe 1 (output di `predict_proba(X)[:,1]`).
        threshold: soglia per binarizzare le probabilita'.

    Returns:
        FraudMetrics dataclass.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return FraudMetrics(
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        f2=float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        auc_pr=float(average_precision_score(y_true, y_proba)),
        auc_roc=float(roc_auc_score(y_true, y_proba)),
        n_predicted_positive=int((y_pred == 1).sum()),
        n_true_positive=int(tp),
        n_false_positive=int(fp),
        n_false_negative=int(fn),
        n_true_negative=int(tn),
    )


def evaluate_on_holdout(
    fitted_estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> FraudMetrics:
    """Valuta un estimator gia' fittato sull'holdout test set."""
    y_proba = fitted_estimator.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_proba, threshold=threshold)
    logger.info(
        "Holdout @t=%.3f: P=%.3f R=%.3f F1=%.3f F2=%.3f AUC-PR=%.4f AUC-ROC=%.4f",
        metrics.threshold, metrics.precision, metrics.recall, metrics.f1,
        metrics.f2, metrics.auc_pr, metrics.auc_roc,
    )
    return metrics


def compare_models(
    metrics_by_model: dict[str, FraudMetrics],
) -> pd.DataFrame:
    """Tabella di confronto modelli ordinata per AUC-PR decrescente."""
    df = pd.DataFrame({
        name: m.as_dict() for name, m in metrics_by_model.items()
    }).T
    df.index.name = "model"
    return df.sort_values("auc_pr", ascending=False)


# --- Diagnostica grafica ---

def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall curve",
    save_path: Path | None = None,
) -> plt.Figure:
    """Curva precision-recall. Per dataset sbilanciati e' piu' informativa
    della ROC: mostra il trade-off effettivo nelle decisioni."""
    y_true = np.asarray(y_true).astype(int)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()  # frazione di positivi (random classifier).

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"AUC-PR = {ap:.4f}")
    ax.axhline(baseline, color="grey", linestyle="--", lw=1,
               label=f"Random (prev = {baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        logger.info("PR curve salvata: %s", save_path)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC curve",
    save_path: Path | None = None,
) -> plt.Figure:
    """Curva ROC. Da leggere con cautela su dataset sbilanciati."""
    y_true = np.asarray(y_true).astype(int)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC-ROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        logger.info("ROC curve salvata: %s", save_path)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion matrix",
    save_path: Path | None = None,
) -> plt.Figure:
    """Confusion matrix con annotazioni numeriche."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Legit (0)", "Fraud (1)"])
    ax.set_yticks([0, 1], ["Legit (0)", "Fraud (1)"])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        logger.info("Confusion matrix salvata: %s", save_path)
    return fig


def get_top_feature_importance(
    fitted_pipeline,
    feature_names,
    top_n: int = 25,
) -> pd.DataFrame:
    """Estrae le top-N feature importance per modelli tree-based.

    Funziona con pipeline `(preprocessor, model)`. Cerca il regressor
    finale e cerca `feature_importances_` (RF, XGBoost) o `coef_` (LogReg).
    """
    final_model = fitted_pipeline.named_steps.get("model", None)
    if final_model is None:
        raise AttributeError("La pipeline non contiene uno step 'model'.")

    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
    elif hasattr(final_model, "coef_"):
        importances = np.abs(final_model.coef_).ravel()
    else:
        raise AttributeError(
            f"Il modello {type(final_model).__name__} non espone "
            "feature_importances_ ne' coef_."
        )

    names = list(feature_names)
    if len(importances) != len(names):
        names = [f"f_{i}" for i in range(len(importances))]
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Top feature importance",
    save_path: Path | None = None,
) -> plt.Figure:
    """Barh delle top feature importance."""
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(importance_df))))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        logger.info("Feature importance salvata: %s", save_path)
    return fig


def make_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Wrapper su `sklearn.metrics.classification_report`. Output testuale."""
    return classification_report(y_true, y_pred, target_names=["legit", "fraud"], digits=4)


__all__ = [
    "FraudMetrics",
    "compute_metrics",
    "evaluate_on_holdout",
    "compare_models",
    "plot_pr_curve",
    "plot_roc_curve",
    "plot_confusion_matrix",
    "get_top_feature_importance",
    "plot_feature_importance",
    "make_classification_report",
    "FIGURES_DIR",
]
