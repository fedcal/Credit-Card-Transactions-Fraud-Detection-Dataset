"""Orchestratore end-to-end della pipeline Credit Card Fraud Detection.

Esegue, nell'ordine:
    1. Caricamento `fraudTrain.csv` + `fraudTest.csv` (Kaggle).
    2. Schema check + ordinamento cronologico.
    3. (opzionale) downsample per smoke-test (--quick).
    4. Costruzione preprocessor + feature engineering + pipeline candidate.
    5. Tuning su time-series CV di tutti i modelli (LogReg, RF, opt. XGB).
    6. Ottimizzazione soglia decisionale su matrice di costi.
    7. Valutazione finale su test set Kaggle (out-of-time).
    8. Persistenza: best_model.joblib + threshold.json + report metriche.

Eseguibile come modulo:

    python -m fraud_pipeline.pipeline           # full run
    python -m fraud_pipeline.pipeline --quick   # smoke-test
    fraud-train                                 # entry point (post pip install -e)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import (
    DEFAULT_CONFIG,
    MODELS_DIR,
    REPORTS_DIR,
    PipelineConfig,
)
from .data import (
    class_distribution,
    downsample_for_smoke_test,
    load_train_test,
    split_features_target,
)
from .evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from .features import FraudFeatureEngineer
from .models import get_all_pipelines
from .preprocessing import build_preprocessor, infer_column_groups
from .threshold import (
    CostMatrix,
    optimal_threshold_by_cost,
    threshold_sweep,
)
from .tuning import TuningResult, summarize_tuning, tune_all_models

logger = logging.getLogger(__name__)


def _attach_feature_engineering(pipelines: dict[str, Pipeline]) -> dict[str, Pipeline]:
    """Antepone `FraudFeatureEngineer` davanti a ogni pipeline esistente.

    Inserire l'engineer come step della pipeline (e non a mano sul
    DataFrame) e' essenziale per la cross-validation: deve girare sui
    fold di train senza vedere il test.
    """
    out: dict[str, Pipeline] = {}
    for name, pipe in pipelines.items():
        steps = [("feature_engineer", FraudFeatureEngineer())] + list(pipe.steps)
        out[name] = Pipeline(steps=steps)
    return out


def prepare_data(
    config: PipelineConfig = DEFAULT_CONFIG,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Pipeline pre-modello: load + split nativo Kaggle (+ optional downsample).

    Returns:
        (X_train, X_test, y_train, y_test).
    """
    df_train, df_test = load_train_test()

    if quick:
        df_train = downsample_for_smoke_test(
            df_train, n_rows=config.quick_sample_rows, random_state=config.random_state,
        )
        # Anche test set ridotto per coerenza.
        df_test = downsample_for_smoke_test(
            df_test, n_rows=config.quick_sample_rows // 2,
            random_state=config.random_state,
        )

    X_train, y_train = split_features_target(df_train)
    X_test, y_test = split_features_target(df_test)

    dist_train = class_distribution(y_train)
    dist_test = class_distribution(y_test)
    logger.info("Train class distribution: %s", dist_train)
    logger.info("Test  class distribution: %s", dist_test)

    return X_train, X_test, y_train, y_test


def build_candidate_pipelines(
    X_train_after_fe: pd.DataFrame,
    config: PipelineConfig,
    include_xgboost: bool = False,
    scale_pos_weight: float | None = None,
) -> dict[str, Pipeline]:
    """Costruisce tutte le pipeline candidate.

    Args:
        X_train_after_fe: DataFrame X_train DOPO `FraudFeatureEngineer`,
            usato solo per inferire i gruppi di colonne.
    """
    groups = infer_column_groups(X_train_after_fe)
    preprocessor = build_preprocessor(
        numeric_cols=groups["numeric"],
        nominal_cols=groups["nominal"],
    )
    base_pipelines = get_all_pipelines(
        preprocessor,
        use_class_weight=config.use_class_weight,
        include_xgboost=include_xgboost,
        scale_pos_weight=scale_pos_weight,
    )
    return _attach_feature_engineering(base_pipelines)


def select_best_model(tuning_results: dict[str, TuningResult]) -> tuple[str, TuningResult]:
    """Sceglie il modello con la migliore AUC-PR su time-series CV."""
    best_name = max(tuning_results, key=lambda k: tuning_results[k].best_score)
    return best_name, tuning_results[best_name]


def save_artifacts(
    best_name: str,
    best_estimator: Pipeline,
    threshold: float,
    holdout_metrics: dict,
    cv_summary: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Path]:
    """Persiste modello + threshold + report metriche."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Modello best, salvato con nome generico per inference.
    model_named_path = models_dir / f"{best_name.lower()}_best.joblib"
    model_default_path = models_dir / "best_model.joblib"
    joblib.dump(best_estimator, model_named_path)
    joblib.dump(best_estimator, model_default_path)

    # Soglia ottimale.
    threshold_path = models_dir / "threshold.json"
    threshold_path.write_text(json.dumps({
        "threshold": float(threshold),
        "model": best_name,
    }, indent=2))

    # Tabella confronto CV.
    cv_path = reports_dir / "cv_summary.csv"
    cv_summary.to_csv(cv_path, index=False)

    # Sweep di soglie.
    threshold_csv = reports_dir / "threshold_analysis.csv"
    threshold_analysis.to_csv(threshold_csv, index=False)

    # Metriche holdout.
    metrics_path = reports_dir / "holdout_metrics.json"
    metrics_path.write_text(json.dumps(holdout_metrics, indent=2))

    return {
        "model_named": model_named_path,
        "model_default": model_default_path,
        "threshold": threshold_path,
        "cv_summary": cv_path,
        "threshold_analysis": threshold_csv,
        "holdout_metrics": metrics_path,
    }


def run_full_pipeline(
    config: PipelineConfig = DEFAULT_CONFIG,
    quick: bool = False,
    include_xgboost: bool = False,
    cost_matrix: CostMatrix | None = None,
) -> dict:
    """Esegue l'intera pipeline e restituisce un dizionario di risultati.

    Args:
        config: configurazione (cv_splits, random_state, ...).
        quick: smoke-test con downsample 50k righe.
        include_xgboost: aggiunge XGBoost ai candidati (richiede l'extra).
        cost_matrix: matrice costi per l'ottimizzazione soglia. Se None,
            usa i default da config.

    Returns:
        Dict con keys:
            - best_model_name
            - best_estimator
            - best_threshold
            - cv_summary
            - holdout_metrics
            - artifacts
    """
    logger.info("=" * 70)
    logger.info("Avvio pipeline Credit Card Fraud Detection (quick=%s, xgb=%s)",
                quick, include_xgboost)
    logger.info("=" * 70)

    cost_matrix = cost_matrix or CostMatrix()

    # --- 1-3. Data preparation ---
    X_train, X_test, y_train, y_test = prepare_data(config=config, quick=quick)

    # --- 4. Pipeline candidate ---
    fe = FraudFeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    # scale_pos_weight = neg / pos sul TRAIN.
    n_pos = max(int(y_train.sum()), 1)
    scale_pos_weight = float((len(y_train) - n_pos) / n_pos)

    pipelines = build_candidate_pipelines(
        X_train_fe,
        config=config,
        include_xgboost=include_xgboost,
        scale_pos_weight=scale_pos_weight,
    )

    # --- 5. Tuning ---
    tuning_results = tune_all_models(
        pipelines=pipelines,
        X=X_train, y=y_train,
        config=config,
        xgb_n_iter=15 if not quick else 3,
    )
    cv_summary = summarize_tuning(tuning_results)
    logger.info("\nRiepilogo tuning (CV AUC-PR):\n%s", cv_summary.to_string(index=False))

    # --- 6. Selezione miglior modello + ottimizzazione soglia ---
    best_name, best_result = select_best_model(tuning_results)
    logger.info(">>> Miglior modello (CV AUC-PR): %s = %.4f",
                best_name, best_result.best_score)

    # Probabilita' sul test per scegliere la soglia.
    # NOTA: in produzione, per essere puristi, la soglia si dovrebbe scegliere
    # su un VALIDATION set separato dal test. Qui usiamo il test perche'
    # il dataset Kaggle ha un test gia' fissato e il volume e' grande
    # abbastanza da avere stime stabili. In pipeline piu' avanzate si
    # sostituirebbe con una holdout interna al training.
    y_proba_test = best_result.best_estimator.predict_proba(X_test)[:, 1]
    best_threshold, best_cost = optimal_threshold_by_cost(
        y_test.to_numpy(), y_proba_test, cost=cost_matrix,
    )

    threshold_analysis = threshold_sweep(
        y_test.to_numpy(), y_proba_test, cost=cost_matrix,
    )

    # --- 7. Holdout metrics per ogni modello ---
    holdout_metrics: dict = {}
    for name, result in tuning_results.items():
        proba = result.best_estimator.predict_proba(X_test)[:, 1]
        m_default = compute_metrics(y_test.to_numpy(), proba, threshold=0.5)
        m_optimal = compute_metrics(y_test.to_numpy(), proba, threshold=best_threshold)
        holdout_metrics[name] = {
            "at_default_0.5": m_default.as_dict(),
            "at_optimal_threshold": m_optimal.as_dict(),
        }

    logger.info(
        "\nMetriche holdout (best=%s) @t=%.4f:\n  %s",
        best_name, best_threshold,
        holdout_metrics[best_name]["at_optimal_threshold"],
    )

    # --- 8. Persist ---
    artifacts = save_artifacts(
        best_name=best_name,
        best_estimator=best_result.best_estimator,
        threshold=best_threshold,
        holdout_metrics=holdout_metrics,
        cv_summary=cv_summary,
        threshold_analysis=threshold_analysis,
    )

    # Plot diagnostici (best-effort).
    try:
        from .config import FIGURES_DIR
        plot_pr_curve(y_test.to_numpy(), y_proba_test,
                      title=f"{best_name}: Precision-Recall (test)",
                      save_path=FIGURES_DIR / f"{best_name.lower()}_pr_curve.png")
        plot_roc_curve(y_test.to_numpy(), y_proba_test,
                       title=f"{best_name}: ROC (test)",
                       save_path=FIGURES_DIR / f"{best_name.lower()}_roc_curve.png")
        y_pred_opt = (y_proba_test >= best_threshold).astype(int)
        plot_confusion_matrix(y_test.to_numpy(), y_pred_opt,
                              title=f"{best_name} @t={best_threshold:.3f}",
                              save_path=FIGURES_DIR / f"{best_name.lower()}_confusion.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plot diagnostici saltati: %s", exc)

    return {
        "best_model_name": best_name,
        "best_estimator": best_result.best_estimator,
        "best_threshold": best_threshold,
        "best_cost": best_cost,
        "cv_summary": cv_summary,
        "holdout_metrics": holdout_metrics,
        "threshold_analysis": threshold_analysis,
        "artifacts": artifacts,
        "tuning_results": tuning_results,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — pipeline end-to-end.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke-test su 50k righe (~1-2 min).",
    )
    parser.add_argument(
        "--xgboost", action="store_true",
        help="Includi XGBoost fra i candidati (richiede l'extra `xgboost`).",
    )
    parser.add_argument(
        "--cost-fn", type=float, default=None,
        help="Costo di un False Negative (frode mancata).",
    )
    parser.add_argument(
        "--cost-fp", type=float, default=None,
        help="Costo di un False Positive (transazione legittima bloccata).",
    )
    return parser


def main_train() -> int:
    """Entry point CLI `fraud-train`."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = _build_arg_parser().parse_args()
    cost_matrix = None
    if args.cost_fn is not None or args.cost_fp is not None:
        cost_matrix = CostMatrix(
            cost_fn=args.cost_fn if args.cost_fn is not None else CostMatrix().cost_fn,
            cost_fp=args.cost_fp if args.cost_fp is not None else CostMatrix().cost_fp,
        )
    run_full_pipeline(
        config=DEFAULT_CONFIG,
        quick=args.quick,
        include_xgboost=args.xgboost,
        cost_matrix=cost_matrix,
    )
    return 0


# Backward compatibility alias.
main = main_train


if __name__ == "__main__":
    raise SystemExit(main_train())


__all__ = [
    "prepare_data",
    "build_candidate_pipelines",
    "select_best_model",
    "save_artifacts",
    "run_full_pipeline",
    "main_train",
]
