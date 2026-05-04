"""Hyperparameter tuning con time-series cross-validation.

Punto chiave didattico: NON usare `KFold` casuale su questo problema.
Le feature aggregate per cliente (`customer_mean_amt_so_far`, ecc.)
contengono informazione sequenziale: shuffling violerebbe la causalita'
temporale e gonfierebbe le metriche.

Usiamo `TimeSeriesSplit` di sklearn:
    fold 1:  [────train────][test1]
    fold 2:  [─────train─────][test2]
    fold 3:  [──────train──────][test3]
    ...

Lo scoring primario e' `average_precision` (= AUC-PR), corretto per
classificazione altamente sbilanciata. NON usare AUC-ROC come metrica
primaria: e' troppo ottimistico quando la classe positiva e' rara.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .config import (
    LOGREG_PARAM_GRID,
    RF_PARAM_GRID,
    XGB_PARAM_GRID,
    DEFAULT_CONFIG,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


# Scoring sklearn corrispondente all'AUC-PR (Average Precision).
PRIMARY_SCORING: str = "average_precision"


@dataclass
class TuningResult:
    """Risultato del tuning di un singolo modello."""
    model_name: str
    best_estimator: Pipeline
    best_params: dict[str, Any]
    best_score: float                     # AUC-PR (Average Precision) sul CV
    cv_results: dict[str, Any]
    duration_seconds: float


def _make_cv(config: PipelineConfig) -> TimeSeriesSplit:
    """TimeSeriesSplit standard. Niente gap fra train e test (ok per fraud)."""
    return TimeSeriesSplit(n_splits=config.cv_splits)


def tune_grid(
    name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> TuningResult:
    """Tuning esauriente via GridSearchCV. Usato per LogReg e RF (grid piccola)."""
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=PRIMARY_SCORING,
        cv=_make_cv(config),
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        refit=True,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    duration = time.perf_counter() - t0
    logger.info(
        "[%s] GridSearch completato in %.1fs. Best AUC-PR=%.4f",
        name, duration, search.best_score_,
    )
    return TuningResult(
        model_name=name,
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        cv_results=search.cv_results_,
        duration_seconds=duration,
    )


def tune_random(
    name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 20,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> TuningResult:
    """RandomizedSearchCV: campionamento dalla grid. Usato per XGBoost."""
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=PRIMARY_SCORING,
        cv=_make_cv(config),
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        refit=True,
        random_state=config.random_state,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    duration = time.perf_counter() - t0
    logger.info(
        "[%s] RandomizedSearch completato in %.1fs. Best AUC-PR=%.4f",
        name, duration, search.best_score_,
    )
    return TuningResult(
        model_name=name,
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        cv_results=search.cv_results_,
        duration_seconds=duration,
    )


def tune_all_models(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
    xgb_n_iter: int = 15,
) -> dict[str, TuningResult]:
    """Esegue il tuning di tutti i modelli registrati.

    Strategia per modello:
        - LogisticRegression -> GridSearchCV (grid piccola).
        - RandomForest       -> GridSearchCV.
        - XGBoost            -> RandomizedSearchCV (grid grande).
    """
    results: dict[str, TuningResult] = {}
    for name, pipeline in pipelines.items():
        if name == "LogisticRegression":
            results[name] = tune_grid(name, pipeline, LOGREG_PARAM_GRID, X, y, config)
        elif name == "RandomForest":
            results[name] = tune_grid(name, pipeline, RF_PARAM_GRID, X, y, config)
        elif name == "XGBoost":
            results[name] = tune_random(
                name, pipeline, XGB_PARAM_GRID, X, y,
                n_iter=xgb_n_iter, config=config,
            )
        else:
            raise ValueError(f"Tuning strategy non definita per modello '{name}'.")
    return results


def summarize_tuning(results: dict[str, TuningResult]) -> pd.DataFrame:
    """Tabella riepilogativa ordinata per AUC-PR decrescente."""
    rows = [
        {
            "model": r.model_name,
            "auc_pr_cv": r.best_score,
            "duration_s": round(r.duration_seconds, 1),
            "best_params": r.best_params,
        }
        for r in results.values()
    ]
    return pd.DataFrame(rows).sort_values("auc_pr_cv", ascending=False).reset_index(drop=True)


__all__ = [
    "PRIMARY_SCORING",
    "TuningResult",
    "tune_grid",
    "tune_random",
    "tune_all_models",
    "summarize_tuning",
]
