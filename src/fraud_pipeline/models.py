"""Definizione dei modelli candidati e delle pipeline complete.

Tre famiglie di modelli con caratteristiche complementari:

- **LogisticRegression** (lineare regolarizzato): baseline interpretabile.
  Coefficienti = log-odds, leggibili. Veloce. Performance limitata su
  pattern non lineari.

- **RandomForest** (ensemble bagging di alberi): cattura interazioni
  non lineari, robusto, senza scaling necessario. Tendenzialmente
  sotto-performa il gradient boosting su tabular ma e' piu' stabile.

- **XGBoost** (gradient boosting, opzionale): tipicamente lo state-of-the-art
  su dataset tabulari di queste dimensioni. Importato lazy: il modulo non
  fallisce se xgboost non e' installato.

Ognuno e' esposto come `sklearn.pipeline.Pipeline` con il preprocessor
condiviso. Gestiamo lo sbilanciamento via `class_weight='balanced'`:
piu' robusto e veloce di SMOTE su 1.5M righe.
"""
from __future__ import annotations

import logging
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED

logger = logging.getLogger(__name__)


def logreg_pipeline(
    preprocessor: ColumnTransformer,
    use_class_weight: bool = True,
) -> Pipeline:
    """Logistic Regression con regolarizzazione L2.

    Note:
        - `solver='liblinear'` e' efficiente su dataset di medie dimensioni
          e supporta L1/L2.
        - `class_weight='balanced'` riassegna pesi inversamente proporzionali
          alla frequenza delle classi: standard per dataset sbilanciati.
        - `max_iter=2000` per garantire convergenza con feature scalate.
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                class_weight="balanced" if use_class_weight else None,
                max_iter=2000,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


def random_forest_pipeline(
    preprocessor: ColumnTransformer,
    use_class_weight: bool = True,
    n_estimators: int = 200,
) -> Pipeline:
    """Random Forest classifier (no scaling necessario, ma riusiamo lo stesso preproc).

    `class_weight='balanced_subsample'` calcola i pesi su ogni bootstrap
    sample, piu' robusto di 'balanced' per RF.
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_leaf=1,
                max_features="sqrt",
                n_jobs=-1,
                class_weight="balanced_subsample" if use_class_weight else None,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


def xgboost_pipeline(
    preprocessor: ColumnTransformer,
    scale_pos_weight: float | None = None,
) -> Pipeline:
    """XGBoost classifier. Import lazy: solleva ImportError chiaro se mancante.

    `scale_pos_weight` e' l'analogo di class_weight per XGBoost. Tipicamente
    impostato a `n_negatives / n_positives` (~ 200 per Kaggle Fraud).
    """
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "XGBoost non e' installato. Aggiungi l'extra:\n"
            "    pip install -e \".[notebooks,xgboost]\"\n"
            "oppure rimuovi 'XGBoost' dalla lista dei modelli candidati."
        ) from exc

    spw = scale_pos_weight if scale_pos_weight is not None else 200.0

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                scale_pos_weight=spw,
                tree_method="hist",
                eval_metric="aucpr",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
    ])


def get_all_pipelines(
    preprocessor: ColumnTransformer,
    use_class_weight: bool = True,
    include_xgboost: bool = False,
    scale_pos_weight: float | None = None,
) -> dict[str, Pipeline]:
    """Restituisce tutte le pipeline candidate (dict ordinato).

    Args:
        preprocessor: ColumnTransformer condiviso (vedi `preprocessing.py`).
        use_class_weight: applica class_weight='balanced' su LogReg/RF.
        include_xgboost: include XGBoost se installato.
        scale_pos_weight: per XGBoost, ratio neg/pos.
    """
    pipelines: dict[str, Pipeline] = {
        "LogisticRegression": logreg_pipeline(preprocessor, use_class_weight),
        "RandomForest": random_forest_pipeline(preprocessor, use_class_weight),
    }
    if include_xgboost:
        try:
            pipelines["XGBoost"] = xgboost_pipeline(preprocessor, scale_pos_weight)
        except ImportError as exc:
            logger.warning("XGBoost non disponibile, salto: %s", exc)
    return pipelines


__all__ = [
    "logreg_pipeline",
    "random_forest_pipeline",
    "xgboost_pipeline",
    "get_all_pipelines",
]
