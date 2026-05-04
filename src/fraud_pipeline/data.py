"""Caricamento, validazione e split temporale del dataset Kaggle Fraud Detection.

Single responsibility: tutto cio' che riguarda I/O dati grezzi, parsing del
timestamp e creazione di train/test set vive qui. Nessuna logica di
preprocessing applicata: quella sta in `features.py` / `preprocessing.py`.

Il dataset Kaggle (`kartik2112/fraud-detection`) e' gia' splittato in
fraudTrain.csv (set 2019-Q1..2020-Q2) e fraudTest.csv (resto). Manteniamo
quello split nativo come default.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_COLUMNS,
    ID_COLUMNS,
    RAW_DIR,
    TARGET_COLUMN,
    TEST_FILENAME,
    TEST_TEMPORAL_FRACTION,
    TRAIN_FILENAME,
)

logger = logging.getLogger(__name__)

DATETIME_COLUMN: Final[str] = "trans_date_trans_time"
DOB_COLUMN: Final[str] = "dob"


def _read_csv(path: Path) -> pd.DataFrame:
    """Carica un CSV Kaggle parserizzando la colonna datetime principale.

    Lasciamo `dob` come stringa: la parsiamo dentro `features.py` per
    derivare l'eta' del cliente al momento della transazione.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File non trovato: {path}\n"
            "Scarica il dataset da Kaggle "
            "(https://www.kaggle.com/datasets/kartik2112/fraud-detection) "
            f"e copialo in {path.parent}/."
        )
    logger.info("Lettura CSV: %s", path)
    df = pd.read_csv(
        path,
        parse_dates=[DATETIME_COLUMN],
        low_memory=False,
    )
    return df


def _validate_schema(df: pd.DataFrame, source: str) -> None:
    """Verifica che le colonne attese siano presenti e che il target sia binario.

    Tollera colonne extra (es. `Unnamed: 0` da export pandas) che
    vengono droppate piu' avanti.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] colonne mancanti: {missing}. "
            "Verifica di avere il CSV originale Kaggle."
        )
    target_values = set(df[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"[{source}] target '{TARGET_COLUMN}' deve essere binario {{0,1}}, "
            f"trovati: {target_values}"
        )


def load_train_test(
    raw_dir: Path = RAW_DIR,
    train_name: str = TRAIN_FILENAME,
    test_name: str = TEST_FILENAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica i due CSV Kaggle, valida lo schema, ordina cronologicamente.

    Returns:
        (df_train, df_test) DataFrame ordinati per `trans_date_trans_time`.
    """
    df_train = _read_csv(raw_dir / train_name)
    df_test = _read_csv(raw_dir / test_name)

    _validate_schema(df_train, source=train_name)
    _validate_schema(df_test, source=test_name)

    # Ordina cronologicamente: indispensabile per qualsiasi feature
    # engineering basato su finestre temporali (rolling, expanding).
    df_train = df_train.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    df_test = df_test.sort_values(DATETIME_COLUMN).reset_index(drop=True)

    logger.info(
        "Caricati: train=%d righe (frodi=%.3f%%), test=%d righe (frodi=%.3f%%)",
        len(df_train), 100 * df_train[TARGET_COLUMN].mean(),
        len(df_test), 100 * df_test[TARGET_COLUMN].mean(),
    )
    return df_train, df_test


def load_combined(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Carica train + test concatenati e ordinati cronologicamente.

    Utile quando si vuole rigenerare lo split temporale custom (es.
    ablation studies) o per EDA su tutto il dataset.
    """
    df_train, df_test = load_train_test(raw_dir=raw_dir)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_TEMPORAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporale: gli ultimi `test_fraction` per data finiscono in test.

    A differenza di `train_test_split` casuale di sklearn, qui rispettiamo
    la dimensione temporale. Niente shuffle: la transazione N+1 e' sempre
    posteriore alla N. Questa e' la modalita' che simula meglio il
    deployment reale (modello addestrato sul passato, valutato sul futuro).
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction deve essere in (0,1), ricevuto {test_fraction}")
    df_sorted = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_fraction))
    train = df_sorted.iloc[:cutoff].copy()
    test = df_sorted.iloc[cutoff:].copy()
    logger.info(
        "Split temporale: train=%d (fino a %s), test=%d (da %s)",
        len(train), train[DATETIME_COLUMN].max(),
        len(test), test[DATETIME_COLUMN].min(),
    )
    return train, test


def split_features_target(
    df: pd.DataFrame, drop_id: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (feature) da y (target binaria).

    Le colonne in `ID_COLUMNS` sono droppate per default: sono PII o
    identificatori non predittivi. La colonna datetime principale viene
    invece MANTENUTA in X: e' usata da `FraudFeatureEngineer` per derivare
    feature temporali e poi droppata nel preprocessor.
    """
    columns_to_drop: list[str] = [TARGET_COLUMN]
    if drop_id:
        columns_to_drop.extend(c for c in ID_COLUMNS if c in df.columns)
    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def downsample_for_smoke_test(
    df: pd.DataFrame, n_rows: int, random_state: int,
) -> pd.DataFrame:
    """Per smoke test (`--quick`): prende le prime `n_rows` cronologiche.

    Non usiamo sample casuale: dovremmo poi ri-ordinare il DataFrame.
    Prendere le prime righe in ordine temporale e' piu' realistico.
    """
    if n_rows >= len(df):
        return df
    df_sorted = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    sub = df_sorted.iloc[:n_rows].copy()
    logger.info("Smoke-test sample: %d righe (di %d), seed=%d", len(sub), len(df), random_state)
    # Forza una percentuale minima di frodi (almeno 50) altrimenti la CV
    # con stratify temporale puo' avere fold tutti zero.
    n_fraud = int(sub[TARGET_COLUMN].sum())
    if n_fraud < 50:
        # Aggiungi alcune frodi prese cronologicamente da df.
        extra_fraud = df.loc[df[TARGET_COLUMN] == 1].iloc[: 50 - n_fraud]
        sub = pd.concat([sub, extra_fraud], ignore_index=True)
        sub = sub.sort_values(DATETIME_COLUMN).reset_index(drop=True)
        logger.info("Smoke-test: aggiunti %d positivi extra per stabilita' CV", len(extra_fraud))
    return sub


def class_distribution(y: pd.Series | np.ndarray) -> dict[str, float]:
    """Statistiche di sbilanciamento. Utile in log e EDA."""
    y_arr = np.asarray(y)
    n = len(y_arr)
    n_pos = int((y_arr == 1).sum())
    return {
        "n": n,
        "n_positives": n_pos,
        "n_negatives": n - n_pos,
        "positive_rate": n_pos / n if n > 0 else 0.0,
        "imbalance_ratio": (n - n_pos) / max(n_pos, 1),
    }


__all__ = [
    "DATETIME_COLUMN",
    "DOB_COLUMN",
    "load_train_test",
    "load_combined",
    "temporal_train_test_split",
    "split_features_target",
    "downsample_for_smoke_test",
    "class_distribution",
]
