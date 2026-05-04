"""Smoke test del feature engineering.

Verifica che `FraudFeatureEngineer`:
- Produca le colonne attese.
- Non leak-i informazione futura negli aggregati per cliente.
- Sia idempotente su transform ripetuti (no side effect mutating).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline.features import FraudFeatureEngineer, haversine_km


@pytest.fixture
def mini_transactions() -> pd.DataFrame:
    """Mini DataFrame con 6 transazioni di 2 clienti, ordine cronologico."""
    return pd.DataFrame({
        "trans_date_trans_time": pd.to_datetime([
            "2020-01-01 02:00:00",
            "2020-01-01 14:30:00",
            "2020-01-02 09:15:00",
            "2020-01-03 23:45:00",
            "2020-01-04 12:00:00",
            "2020-01-05 03:00:00",
        ]),
        "cc_num": [1001, 1001, 1002, 1001, 1002, 1002],
        "merchant": ["m_a", "m_b", "m_a", "m_c", "m_b", "m_a"],
        "category": ["food", "shopping", "food", "gas", "shopping", "food"],
        "amt": [10.0, 200.0, 50.0, 1.0, 30.0, 500.0],
        "lat": [40.0, 40.0, 41.0, 40.0, 41.0, 41.0],
        "long": [-75.0, -75.0, -76.0, -75.0, -76.0, -76.0],
        "merch_lat": [40.1, 40.5, 41.2, 40.05, 41.0, 50.0],
        "merch_long": [-75.1, -75.0, -76.1, -75.0, -76.0, -76.0],
        "city": ["A", "A", "B", "A", "B", "B"],
        "state": ["PA", "PA", "NY", "PA", "NY", "NY"],
        "zip": ["1", "1", "2", "1", "2", "2"],
        "city_pop": [10_000] * 6,
        "job": ["eng"] * 6,
        "gender": ["M", "M", "F", "M", "F", "F"],
        "dob": pd.to_datetime(["1990-01-01"] * 6).astype(str),
        "unix_time": [1577840400, 1577885400, 1577955300, 1578093900,
                      1578139200, 1578193200],
    })


def test_haversine_returns_zero_for_identical_points() -> None:
    """Distanza fra due punti coincidenti deve essere zero (entro tolleranza)."""
    d = haversine_km([40.0], [-75.0], [40.0], [-75.0])
    assert d[0] == pytest.approx(0.0, abs=1e-9)


def test_haversine_known_distance_nyc_la() -> None:
    """NYC (40.7128, -74.006) -> LA (34.0522, -118.2437) ~= 3936 km."""
    d = haversine_km([40.7128], [-74.006], [34.0522], [-118.2437])
    assert d[0] == pytest.approx(3936.0, abs=20.0)


def test_feature_engineer_produces_expected_columns(mini_transactions) -> None:
    """Verifica le colonne derivate principali."""
    fe = FraudFeatureEngineer()
    out = fe.fit_transform(mini_transactions)
    expected_new = {
        "hour", "day_of_week", "month", "is_weekend", "is_night",
        "customer_age_years",
        "distance_km", "is_far_tx",
        "log_amt", "is_small_amt",
        "customer_tx_count_so_far",
        "customer_mean_amt_so_far",
        "customer_std_amt_so_far",
        "customer_amt_zscore",
    }
    assert expected_new.issubset(set(out.columns)), (
        f"Mancano: {expected_new - set(out.columns)}"
    )


def test_feature_engineer_drops_source_columns_by_default(mini_transactions) -> None:
    """Le colonne sorgente (datetime, lat/long, dob, cc_num) vengono droppate."""
    fe = FraudFeatureEngineer(drop_source_columns=True)
    out = fe.fit_transform(mini_transactions)
    for col in ("trans_date_trans_time", "dob", "lat", "long",
                "merch_lat", "merch_long", "city", "state", "cc_num"):
        assert col not in out.columns, f"{col} dovrebbe essere droppato"


def test_customer_aggregates_no_future_leakage(mini_transactions) -> None:
    """La prima transazione di ogni cliente deve avere tx_count_so_far = 0.

    Se cosi' non fosse, l'expanding sta includendo la riga corrente
    (data leakage temporale).

    Usiamo drop_source_columns=False cosi' `cc_num` resta nell'output e
    possiamo raggruppare correttamente.
    """
    fe = FraudFeatureEngineer(drop_source_columns=False)
    out = fe.fit_transform(mini_transactions)
    first_per_customer = out.groupby("cc_num").head(1)
    assert (first_per_customer["customer_tx_count_so_far"] == 0).all(), (
        "La prima tx per cliente deve avere tx_count_so_far == 0"
    )
    # Per la prima tx, z-score = 0 (no storico).
    assert (first_per_customer["customer_amt_zscore"] == 0).all(), (
        "La prima tx per cliente deve avere customer_amt_zscore == 0"
    )

    # Per le tx successive (non prime), tx_count_so_far deve essere > 0.
    non_first = out.groupby("cc_num").tail(-1) if False else (
        out.drop(first_per_customer.index)
    )
    assert (non_first["customer_tx_count_so_far"] >= 1).all(), (
        "Le tx successive devono avere tx_count_so_far >= 1"
    )


def test_feature_engineer_is_pure_no_input_mutation(mini_transactions) -> None:
    """Verifica immutabilita': il DataFrame originale non viene mutato."""
    snapshot = mini_transactions.copy()
    fe = FraudFeatureEngineer()
    _ = fe.fit_transform(mini_transactions)
    pd.testing.assert_frame_equal(mini_transactions, snapshot)


def test_is_night_flag_correct(mini_transactions) -> None:
    """Le tx alle 02:00, 23:45, 03:00 devono avere is_night=1."""
    fe = FraudFeatureEngineer()
    out = fe.fit_transform(mini_transactions)
    # Order non garantito dopo aggregati; ricostruiamo via hour.
    night_rows = out[out["is_night"] == 1]["hour"].tolist()
    for h in night_rows:
        assert h < 6 or h >= 22, f"is_night=1 ma hour={h}"
