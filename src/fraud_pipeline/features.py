"""Feature engineering domain-specific per Credit Card Fraud Detection.

Tutte le trasformazioni qui sono `BaseEstimator` + `TransformerMixin` per
essere componibili dentro `sklearn.pipeline.Pipeline`. Vantaggi:

1. **No leakage**: `fit_transform` su train, `transform` su test.
2. **Riproducibilita'**: serializzabili con joblib insieme al modello.
3. **Testabilita'**: ogni transformer ha una signature standard.

Le feature sono progettate per essere INTERPRETABILI ed efficaci
soprattutto contro frodi del tipo "card-not-present" e "test-amount":
small charge seguito da grande transazione.

ATTENZIONE — leakage temporale:
    Tutti gli aggregati cliente/merchant sono calcolati con `expanding()`
    sui dati ordinati cronologicamente, in modo che la riga `t` veda solo
    statistiche calcolate su righe `< t`. NON usare `groupby().mean()`
    globale: includerebbe il futuro nella media e gonfierebbe le metriche.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


# Raggio terrestre per la formula di Haversine (km).
EARTH_RADIUS_KM: float = 6371.0


def haversine_km(
    lat1: np.ndarray | pd.Series,
    lon1: np.ndarray | pd.Series,
    lat2: np.ndarray | pd.Series,
    lon2: np.ndarray | pd.Series,
) -> np.ndarray:
    """Distanza Haversine (km) fra due punti su sfera terrestre.

    Usa formula numericamente stabile con `arcsin(sqrt(...))`. Vettorizzata
    per operare su intere colonne pandas.
    """
    lat1_r = np.radians(np.asarray(lat1, dtype=float))
    lon1_r = np.radians(np.asarray(lon1, dtype=float))
    lat2_r = np.radians(np.asarray(lat2, dtype=float))
    lon2_r = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Aggiunge feature derivate al DataFrame di transazioni.

    Tre famiglie di feature:

    Temporali:
        - hour, day_of_week, month, is_weekend, is_night
        - customer_age_at_tx (da `dob` parsato come data)

    Geografiche:
        - distance_km cliente <-> merchant (Haversine)
        - is_far (>500km) flag

    Trasformazioni dell'importo:
        - log_amt (log1p)

    Aggregati cliente (rolling expanding, NO leakage):
        - tx_count_so_far, mean_amt_so_far, std_amt_so_far per `cc_num`
        - amt_zscore_vs_history: quanto la transazione corrente devia
          dall'utente

    Tutte le aggregate sono calcolate con `expanding()` su dati ordinati
    cronologicamente PER CLIENTE, quindi il record t-esimo vede solo le
    statistiche calcolate sui record 0..t-1 dello stesso cliente.
    """

    DATETIME_COL = "trans_date_trans_time"
    DOB_COL = "dob"
    AMT_COL = "amt"
    LAT_COL = "lat"
    LON_COL = "long"
    MERCH_LAT_COL = "merch_lat"
    MERCH_LON_COL = "merch_long"
    CUSTOMER_COL = "cc_num"

    def __init__(
        self,
        add_temporal: bool = True,
        add_geo: bool = True,
        add_amount: bool = True,
        add_customer_aggregates: bool = True,
        drop_source_columns: bool = True,
    ) -> None:
        self.add_temporal = add_temporal
        self.add_geo = add_geo
        self.add_amount = add_amount
        self.add_customer_aggregates = add_customer_aggregates
        self.drop_source_columns = drop_source_columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureEngineer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FraudFeatureEngineer richiede un pd.DataFrame in input.")
        # Salva i nomi colonna di input (servono in inferenza).
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    # --- Sub-transform: ognuno restituisce un nuovo DataFrame, no mutation in-place ---

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ts = pd.to_datetime(out[self.DATETIME_COL], errors="coerce")
        out["hour"] = ts.dt.hour.fillna(0).astype(int)
        out["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
        out["month"] = ts.dt.month.fillna(1).astype(int)
        out["is_weekend"] = (ts.dt.dayofweek >= 5).fillna(False).astype(int)
        out["is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).fillna(False).astype(int)

        # Eta' del cliente alla data della transazione (anni).
        if self.DOB_COL in out.columns:
            dob = pd.to_datetime(out[self.DOB_COL], errors="coerce")
            age_days = (ts - dob).dt.days
            out["customer_age_years"] = (age_days / 365.25).fillna(40.0).clip(lower=0, upper=120)
        return out

    def _add_geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        needed = {self.LAT_COL, self.LON_COL, self.MERCH_LAT_COL, self.MERCH_LON_COL}
        if not needed.issubset(out.columns):
            return out
        out["distance_km"] = haversine_km(
            out[self.LAT_COL], out[self.LON_COL],
            out[self.MERCH_LAT_COL], out[self.MERCH_LON_COL],
        )
        # Flag "transazione lontana": >500km dal customer e' raro per uso normale.
        out["is_far_tx"] = (out["distance_km"] > 500).astype(int)
        return out

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.AMT_COL in out.columns:
            # log1p e' robusto a importi nulli e comprime la coda.
            out["log_amt"] = np.log1p(out[self.AMT_COL].clip(lower=0))
            # Indicatore "small amount" tipico delle frodi di test (<$1).
            out["is_small_amt"] = (out[self.AMT_COL] < 1.0).astype(int)
        return out

    def _add_customer_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregati expanding per cliente, calcolati senza leakage.

        Il record t-esimo vede solo le statistiche dei record 0..t-1 dello
        stesso cliente (shift(1) sull'expanding). I record iniziali (per
        cliente) hanno mean=NaN e ricevono fill 0/1 per coerenza:
        - count = 0 (il cliente e' "nuovo")
        - mean = importo corrente (nessun riferimento storico)
        - std = 0
        - zscore = 0
        """
        out = df.copy()
        if self.CUSTOMER_COL not in out.columns or self.AMT_COL not in out.columns:
            return out
        # Ordina cronologicamente per garantire che expanding sia sul passato.
        if self.DATETIME_COL in out.columns:
            out = out.sort_values([self.CUSTOMER_COL, self.DATETIME_COL]).reset_index(drop=True)

        grp = out.groupby(self.CUSTOMER_COL)[self.AMT_COL]

        # `shift(1)` BEFORE expanding garantisce che la riga corrente non
        # entri nella sua stessa statistica. Equivalente a `expanding(min_periods=1)
        # .mean().shift(1)` ma e' piu' diretto.
        prev_amt = grp.shift(1)

        # Per `expanding` su gruppi, usiamo apply via groupby per evitare
        # warning: la versione vettorizzata richiede 1 riga di accumulo
        # quindi ricostruiamo cumulativi a mano (piu' veloce).
        cum_count = grp.cumcount()  # 0,1,2,... PER cliente; 0 = prima tx
        cum_sum_prev = grp.shift(1).groupby(out[self.CUSTOMER_COL]).cumsum()
        cum_sumsq_prev = (grp.shift(1) ** 2).groupby(out[self.CUSTOMER_COL]).cumsum()
        # cum_count_prev = numero di transazioni pregresse (= cumcount nella tx attuale).
        cum_count_prev = cum_count

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_prev = cum_sum_prev / cum_count_prev
            var_prev = cum_sumsq_prev / cum_count_prev - mean_prev ** 2
            std_prev = np.sqrt(np.clip(var_prev, 0.0, None))

        # Fill per le prime transazioni (cum_count_prev == 0).
        first_tx_mask = (cum_count_prev == 0).values
        mean_prev = mean_prev.fillna(out[self.AMT_COL])
        std_prev = std_prev.fillna(0.0)

        out["customer_tx_count_so_far"] = cum_count_prev.astype(int).values
        out["customer_mean_amt_so_far"] = mean_prev.values
        out["customer_std_amt_so_far"] = std_prev.values

        # z-score vs history: quanto la transazione attuale e' anomala
        # rispetto al cliente. Usiamo std minimo = max(1.0, std_prev) per
        # stabilita': quando un cliente ha pochissime tx (std=0 o piccolo)
        # il rapporto esplode. Cap a $1 di std minima rende la feature
        # comparabile fra clienti con storia diversa.
        denom = np.maximum(std_prev.values, 1.0)
        zscore = (out[self.AMT_COL].values - mean_prev.values) / denom
        # Clip per evitare valori estremi (tipo 1e8) che dominerebbero il
        # modello. Range [-50, 50] copre tutti i casi realistici.
        zscore = np.clip(zscore, -50.0, 50.0)
        out["customer_amt_zscore"] = zscore
        # Per la prima tx, z-score = 0 (non ho storico).
        out.loc[first_tx_mask, "customer_amt_zscore"] = 0.0

        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FraudFeatureEngineer.transform richiede un pd.DataFrame.")
        out = X.copy()
        if self.add_temporal:
            out = self._add_temporal_features(out)
        if self.add_geo:
            out = self._add_geo_features(out)
        if self.add_amount:
            out = self._add_amount_features(out)
        if self.add_customer_aggregates:
            out = self._add_customer_aggregates(out)

        if self.drop_source_columns:
            # Le colonne sorgente non sono direttamente utili al modello
            # (sono testuali, ad alta cardinalita', o gia' compresse in
            # feature derivate). Le droppiamo per avere un X.dtypes pulito
            # in input al ColumnTransformer.
            to_drop = [
                self.DATETIME_COL, self.DOB_COL,
                self.LAT_COL, self.LON_COL,
                self.MERCH_LAT_COL, self.MERCH_LON_COL,
                "city", "state", "zip",
                # `cc_num` usato dagli aggregati ma droppato in output:
                # numero carta non e' una feature predittiva (rischio overfit
                # su singolo cliente).
                self.CUSTOMER_COL,
            ]
            out = out.drop(columns=[c for c in to_drop if c in out.columns])

        return out

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        # Necessario per sklearn>=1.0 quando il transformer e' usato in
        # ColumnTransformer/Pipeline e si vogliono i nomi di output.
        if input_features is None:
            return np.array([], dtype=object)
        return np.array(list(input_features), dtype=object)


__all__ = ["FraudFeatureEngineer", "haversine_km", "EARTH_RADIUS_KM"]
