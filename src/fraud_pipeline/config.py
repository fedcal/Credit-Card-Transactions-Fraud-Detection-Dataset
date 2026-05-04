"""Configurazione globale della pipeline Credit Card Fraud Detection.

Centralizza path, costanti, schema dataset e iperparametri di default.
Mantiene il codice pulito (no magic numbers/path sparsi) e facilita
l'esecuzione riproducibile da CLI o notebook.

Convenzione: tutto ciò che non e' una funzione o classe vive qui o in un
file `*_config.py` dedicato. Mai costanti hardcoded dentro il codice di
business.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Filesystem layout ---
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = REPORTS_DIR / "models"

# --- Dataset Kaggle ---
TRAIN_FILENAME: str = "fraudTrain.csv"
TEST_FILENAME: str = "fraudTest.csv"
TARGET_COLUMN: str = "is_fraud"

# Colonne attese nel CSV originale (Kaggle: kartik2112/fraud-detection).
# Usato come schema-check al caricamento.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "first", "last", "gender", "street", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time",
    "merch_lat", "merch_long", "is_fraud",
)

# Colonne ID/PII da NON usare come feature (privacy + nessun valore predittivo).
ID_COLUMNS: tuple[str, ...] = (
    "Unnamed: 0",  # indice scritto da pandas in alcuni dump
    "trans_num",
    "first", "last", "street",
)

# --- Riproducibilità ---
RANDOM_SEED: int = 42

# --- Validation strategy ---
# Numero di fold per walk-forward time series CV.
TIME_SERIES_CV_SPLITS: int = 4

# Frazione del dataset usata come test set quando si usa lo split temporale
# (gli ultimi `TEST_TEMPORAL_FRACTION` per data sono test).
TEST_TEMPORAL_FRACTION: float = 0.20

# --- Smoke-test (--quick) ---
# Numero di righe campionate (cronologicamente: prime N) per smoke test.
QUICK_SAMPLE_ROWS: int = 50_000

# --- Threshold di default per `predict_fraud` ---
# 0.5 e' subottimale per dataset sbilanciati: la soglia ottimale (vedi
# `threshold.py`) viene salvata accanto al modello e usata in inferenza.
DEFAULT_DECISION_THRESHOLD: float = 0.5

# --- Cost matrix di esempio (in euro/dollari) ---
# Il PW chiede di formalizzare un'asimmetria di costo. Valori indicativi:
# perdere una frode media costa ~120 EUR; bloccare ingiustamente una
# transazione legittima costa ~5 EUR (chargeback friction + customer
# experience).
DEFAULT_COST_FALSE_NEGATIVE: float = 120.0
DEFAULT_COST_FALSE_POSITIVE: float = 5.0
DEFAULT_COST_TRUE_NEGATIVE: float = 0.0
DEFAULT_COST_TRUE_POSITIVE: float = 0.0


@dataclass(frozen=True)
class PipelineConfig:
    """Iperparametri e flag della pipeline.

    Frozen=True per evitare mutazioni accidentali dopo l'inizializzazione.
    Per cambiare un parametro si crea un nuovo oggetto (immutabilita').
    """
    random_state: int = RANDOM_SEED
    test_temporal_fraction: float = TEST_TEMPORAL_FRACTION
    cv_splits: int = TIME_SERIES_CV_SPLITS
    n_jobs: int = -1
    verbose: int = 1
    use_class_weight: bool = True
    quick_sample_rows: int = QUICK_SAMPLE_ROWS


DEFAULT_CONFIG: PipelineConfig = PipelineConfig()


# --- Iperparametri di default per i modelli candidati ---
# Volutamente piccoli per consentire esecuzione su laptop in tempi
# didattici. Ampliarli per produzione/competizione.

LOGREG_PARAM_GRID: dict[str, list] = {
    "model__C": [0.1, 1.0, 10.0],
    "model__penalty": ["l2"],
}

RF_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 16],
    "model__min_samples_leaf": [1, 5],
    "model__max_features": ["sqrt"],
}

XGB_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [300, 600],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__reg_lambda": [1.0, 5.0],
}


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "EXTERNAL_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "MODELS_DIR",
    "TRAIN_FILENAME",
    "TEST_FILENAME",
    "TARGET_COLUMN",
    "EXPECTED_COLUMNS",
    "ID_COLUMNS",
    "RANDOM_SEED",
    "TIME_SERIES_CV_SPLITS",
    "TEST_TEMPORAL_FRACTION",
    "QUICK_SAMPLE_ROWS",
    "DEFAULT_DECISION_THRESHOLD",
    "DEFAULT_COST_FALSE_NEGATIVE",
    "DEFAULT_COST_FALSE_POSITIVE",
    "DEFAULT_COST_TRUE_NEGATIVE",
    "DEFAULT_COST_TRUE_POSITIVE",
    "PipelineConfig",
    "DEFAULT_CONFIG",
    "LOGREG_PARAM_GRID",
    "RF_PARAM_GRID",
    "XGB_PARAM_GRID",
]
