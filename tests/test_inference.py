"""Smoke test della funzione `predict_fraud`.

Senza modello reale, validiamo la logica di alignment colonne, gestione
dict vs DataFrame e propagazione del threshold. Per il path "modello reale"
serve `fraud-train`, che richiede il dataset Kaggle: skipped se mancante.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline.inference import (
    DEFAULT_MODEL_PATH,
    _align_to_expected,
    example_transaction,
    predict_fraud,
)


def _fake_model_with_proba(proba: list[float]) -> MagicMock:
    """Crea un mock di pipeline con `predict_proba` pre-determinato."""
    model = MagicMock()
    proba_arr = np.array([[1 - p, p] for p in proba])
    model.predict_proba.return_value = proba_arr
    # La funzione `_expected_columns` cerca named_steps.
    first_step = MagicMock()
    first_step.feature_names_in_ = np.array(
        list(example_transaction().keys()), dtype=object,
    )
    model.named_steps = {"feature_engineer": first_step}
    return model


def test_example_transaction_has_required_keys() -> None:
    """L'esempio deve contenere le colonne attese dal preprocessor."""
    tx = example_transaction()
    required = {
        "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
        "lat", "long", "merch_lat", "merch_long", "dob",
    }
    assert required.issubset(tx.keys())


def test_align_to_expected_fills_missing_with_nan() -> None:
    """Le colonne mancanti devono diventare NaN."""
    df = pd.DataFrame({"a": [1], "c": [3]})
    aligned = _align_to_expected(df, expected=["a", "b", "c"])
    assert list(aligned.columns) == ["a", "b", "c"]
    assert pd.isna(aligned["b"].iloc[0])


def test_align_to_expected_drops_extra_columns() -> None:
    """Colonne extra non nel training devono essere rimosse."""
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [99]})
    aligned = _align_to_expected(df, expected=["a", "b"])
    assert list(aligned.columns) == ["a", "b"]
    assert "extra" not in aligned.columns


@patch("fraud_pipeline.inference._load_model")
@patch("fraud_pipeline.inference._load_threshold")
def test_predict_fraud_single_dict(mock_load_threshold, mock_load_model) -> None:
    """Input dict singolo deve produrre output dict (non lista)."""
    mock_load_threshold.return_value = 0.5
    mock_load_model.return_value = _fake_model_with_proba([0.85])

    result = predict_fraud(example_transaction())

    assert isinstance(result, dict)
    assert "fraud_probability" in result
    assert "is_fraud" in result
    assert "threshold" in result
    assert result["fraud_probability"] == pytest.approx(0.85)
    assert result["is_fraud"] is True
    assert result["threshold"] == pytest.approx(0.5)


@patch("fraud_pipeline.inference._load_model")
@patch("fraud_pipeline.inference._load_threshold")
def test_predict_fraud_below_threshold(mock_load_threshold, mock_load_model) -> None:
    """Probabilita' < threshold => is_fraud = False."""
    mock_load_threshold.return_value = 0.5
    mock_load_model.return_value = _fake_model_with_proba([0.20])
    result = predict_fraud(example_transaction())
    assert result["is_fraud"] is False


@patch("fraud_pipeline.inference._load_model")
@patch("fraud_pipeline.inference._load_threshold")
def test_predict_fraud_explicit_threshold_overrides_loaded(
    mock_load_threshold, mock_load_model,
) -> None:
    """Threshold esplicito ha priorita' su quella caricata da disco."""
    mock_load_threshold.return_value = 0.5  # ignorata
    mock_load_model.return_value = _fake_model_with_proba([0.30])
    result = predict_fraud(example_transaction(), threshold=0.20)
    assert result["threshold"] == pytest.approx(0.20)
    assert result["is_fraud"] is True  # 0.30 >= 0.20


@patch("fraud_pipeline.inference._load_model")
@patch("fraud_pipeline.inference._load_threshold")
def test_predict_fraud_batch_dataframe(mock_load_threshold, mock_load_model) -> None:
    """Input DataFrame produce list[dict] di lunghezza coerente."""
    mock_load_threshold.return_value = 0.5
    mock_load_model.return_value = _fake_model_with_proba([0.10, 0.95, 0.40])

    df = pd.DataFrame([example_transaction()] * 3)
    results = predict_fraud(df)

    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)
    assert results[0]["is_fraud"] is False
    assert results[1]["is_fraud"] is True
    assert results[2]["is_fraud"] is False


def test_predict_fraud_invalid_input_raises_type_error() -> None:
    """Tipi diversi da dict/DataFrame devono fallire con TypeError."""
    with pytest.raises(TypeError):
        predict_fraud(["not", "valid"])  # type: ignore[arg-type]


@pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="Modello non addestrato. Esegui `fraud-train` per abilitare il test.",
)
def test_predict_fraud_with_real_model() -> None:
    """Smoke test end-to-end con il modello reale (skipped se non presente)."""
    result = predict_fraud(example_transaction())
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert isinstance(result["is_fraud"], bool)
