"""Costruzione del preprocessor sklearn (ColumnTransformer).

Combina due branch parallele:

- **Numeriche**: imputer mediana + StandardScaler. Lo scaling e' necessario
  per la Logistic Regression; non e' dannoso per i modelli tree-based
  (sono invarianti per trasformazioni monotone delle feature). Mantenerlo
  nel preprocessor unico semplifica la pipeline.

- **Nominali (alta cardinalita')**: il dataset Kaggle ha 14 categorie di
  merchant e 693 merchant unici. Per evitare OneHot esplosivo (>700 colonne)
  usiamo:
    * `OneHotEncoder(min_frequency=N)` per le categorie con cardinalita'
      contenuta (`category`, `gender`, `job`).
    * Drop diretto per gli identificatori ad altissima cardinalita'
      (`merchant`, `cc_num` gia' droppato in feature engineering).

Tutto e' contenuto in un `ColumnTransformer` per due ragioni:

1. **No data leakage**: imputer/encoder/scaler calcolano statistiche solo
   sul training set e le riapplicano in test/inferenza.
2. **Riproducibilita'**: serializzando con joblib, l'inferenza e'
   bit-identica al training.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# Colonne nominali ad alta cardinalita' che vanno DROPPATE nel preprocessor
# (no OneHot diretto: troppe dummy). In una pipeline piu' avanzata si
# userebbe target encoding con prior — vedi `docs/scelte_tecniche/scelte_modello.md`.
HIGH_CARDINALITY_COLUMNS: tuple[str, ...] = (
    "merchant",  # 693 unique
)


def _build_numeric_branch() -> Pipeline:
    """Branch numerica: imputer mediana + StandardScaler.

    Lo scaling e' applicato anche per i modelli tree-based: e' inerte e
    ci permette di riusare lo stesso preprocessor.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])


def _build_nominal_branch() -> Pipeline:
    """Branch nominale: imputer 'unknown' + OneHotEncoder con min_frequency.

    `min_frequency=10` rimuove categorie ultra-rare (≤9 occorrenze nel
    training): riduce dimensionalita' e rumore. Le categorie inattese in
    test/inferenza vengono ignorate (`handle_unknown='ignore'`).
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                min_frequency=10,
            ),
        ),
    ])


def build_preprocessor(
    numeric_cols: Sequence[str],
    nominal_cols: Sequence[str],
) -> ColumnTransformer:
    """Costruisce il `ColumnTransformer` completo dato il grouping di colonne.

    Args:
        numeric_cols: colonne numeriche continue/discrete.
        nominal_cols: colonne categoriche senza ordine (category, gender, job, ...).

    Returns:
        ColumnTransformer parametrizzato. Va `fit` su X_train e poi
        `transform` su X_test e dati di inferenza.
    """
    transformers: list[tuple] = []
    if numeric_cols:
        transformers.append(("num", _build_numeric_branch(), list(numeric_cols)))
    if nominal_cols:
        transformers.append(("nom", _build_nominal_branch(), list(nominal_cols)))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Esplicito: ogni colonna non gestita viene scartata.
        verbose_feature_names_out=False,
    )
    logger.info(
        "Preprocessor pronto: %d numeriche, %d nominali (one-hot).",
        len(numeric_cols), len(nominal_cols),
    )
    return preprocessor


def infer_column_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    """Inferisce i due gruppi di colonne dal DataFrame post-feature-engineering.

    Strategia:
        - droppa colonne ad altissima cardinalita' (vedi HIGH_CARDINALITY_COLUMNS);
        - le rimanenti `object` sono nominali;
        - le rimanenti numeriche sono... numeriche.
    """
    cols = [c for c in X.columns if c not in HIGH_CARDINALITY_COLUMNS]
    sub = X[cols]
    nominal_cols = list(sub.select_dtypes(include=["object", "category"]).columns)
    numeric_cols = list(sub.select_dtypes(include="number").columns)
    return {
        "numeric": numeric_cols,
        "nominal": nominal_cols,
    }


__all__ = [
    "HIGH_CARDINALITY_COLUMNS",
    "build_preprocessor",
    "infer_column_groups",
]
