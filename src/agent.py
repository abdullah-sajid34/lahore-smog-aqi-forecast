"""
agent.py -- ForecastAgent for the Lahore PM2.5 console system.

The agent is a classic perception-decision loop:

    while not env.is_done():
        percept = env.step()
        agent.perceive(percept)      # update internal history
        pm25_hat = agent.forecast()  # predict tomorrow
        advice   = agent.advise()    # map forecast to health advisory

Design decisions
----------------
Inference vs training features:
    build_model_features() in forecasters.py is designed for TRAINING: it
    also computes a TARGET column (pm25 at t+h) and drops any row where
    the target is NaN.  At the query date, target = pm25[t+1] is unknown
    (that is what we are predicting), so build_model_features would drop
    the query row entirely.

    Instead we use _build_inference_features(), which applies the SAME
    feature transformations (_calendar_shift, _calendar_rolling, cyclic
    encoding) but does NOT compute or require a target.  The resulting
    feature vector is identical to what training produced for any historical
    row -- this is verified by TestAgentForecast.test_forecast_matches_model.

7-day history buffer:
    The agent stores the last 8 Percepts (7+1) keyed by calendar date.
    _build_inference_features() uses a mini-DataFrame from these 8 rows so
    that lag6 (needs 6 prior days) and roll7 (needs 7 prior days) on the
    LAST (query) row are non-NaN.  Rows earlier in the window may have NaN
    lags -- that is fine because we only use the last row for prediction.

Feature completeness check:
    After building, we verify the query-date row has zero NaN in MODEL_FEATURES.
    If the buffer contains a calendar gap, some lags will be NaN and we return
    None with a clear message rather than silently using a stale value.
"""

import sys
from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    AGENT_MODEL_FILE,
    AQI_ADVISORY_DISCLAIMER,
    AQI_ADVISORY_TEXTS,
    AQI_BREAKPOINTS,
    MERGED_CSV,
    MODEL_FEATURES,
    OUTPUTS_MODELS,
)
from src.dataset import _calendar_shift, _calendar_rolling
from src.environment import Percept


# ---------------------------------------------------------------------------
# Advisory helper (standalone, importable by tests)
# ---------------------------------------------------------------------------

def pm25_to_category(pm25: float) -> str:
    """
    Map a PM2.5 concentration (ug/m3) to its US EPA AQI category label.
    Uses the same breakpoints as config.AQI_BREAKPOINTS.
    """
    for _, _, Cp_lo, Cp_hi, label, _ in AQI_BREAKPOINTS:
        if pm25 <= Cp_hi:
            return label
    return AQI_BREAKPOINTS[-1][4]   # Hazardous+ for anything above last breakpoint


def get_advisory(pm25: float) -> dict:
    """
    Return a dict with:
        category   : AQI category label
        advisory   : one-line action text from config.AQI_ADVISORY_TEXTS
        disclaimer : always-present legal note from config.AQI_ADVISORY_DISCLAIMER
    """
    category = pm25_to_category(pm25)
    advisory = AQI_ADVISORY_TEXTS.get(category, "No advisory available.")
    return {
        "category":   category,
        "advisory":   advisory,
        "disclaimer": AQI_ADVISORY_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Inference-time feature builder
# ---------------------------------------------------------------------------

def _build_inference_features(mini_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build MODEL_FEATURES for every row in mini_df WITHOUT a target column.

    This applies the same transformations as build_model_features() in
    forecasters.py -- _calendar_shift for lags, _calendar_rolling for the
    7-day mean, and sine/cosine cyclic encoding -- but does NOT require
    pm25[t+1] to exist.  At inference time the target is unknown by design.

    WHY not call build_model_features():
        That function shifts the target column backwards and drops rows
        where target is NaN.  On the query date (the last row), target =
        pm25[t+1] is exactly what we want to predict, so it is NaN and the
        row gets dropped.  We reproduce the same feature logic without that
        final step.

    Parameters
    ----------
    mini_df : DataFrame with DatetimeIndex containing at least 8 consecutive
              calendar days ending on the query date.  Must have pm25_ugm3,
              weather columns, and the lag1-3/roll7 columns from the CSV
              (or they will be re-computed from pm25_ugm3 if absent).

    Returns
    -------
    DataFrame with MODEL_FEATURES columns, same DatetimeIndex as mini_df.
    Rows with NaN in any MODEL_FEATURES column are NOT dropped here;
    the caller checks the last row.
    """
    out = mini_df.copy()

    # Lags 1-3 are in the CSV; lags 4-6 are not and must be computed here.
    # We re-compute all of them from pm25_ugm3 for consistency.
    for lag in [1, 2, 3, 4, 5, 6]:
        out[f"pm25_lag{lag}"] = _calendar_shift(out["pm25_ugm3"], lag)

    # 7-day rolling mean (calendar-day safe, current day excluded)
    out["pm25_roll7"] = _calendar_rolling(out["pm25_ugm3"], 7)

    # Cyclic calendar encoding
    out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12)
    out["doy_sin"]   = np.sin(2 * np.pi * out.index.day_of_year / 365)
    out["doy_cos"]   = np.cos(2 * np.pi * out.index.day_of_year / 365)

    return out


# ---------------------------------------------------------------------------
# ForecastAgent
# ---------------------------------------------------------------------------

class ForecastAgent:
    """
    A learning agent that uses a saved Forecaster to predict next-day PM2.5.

    Parameters
    ----------
    model_path : path to a joblib-serialised Forecaster (default: rf_h1.joblib,
                 the best model by validation RMSE).
    history_days : number of prior calendar days to keep in memory (default 7,
                   needed so that lag6 and roll7 on the query date are non-NaN).
    """

    def __init__(
        self,
        model_path: Path = AGENT_MODEL_FILE,
        history_days: int = 7,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run: python src/run_experiments.py"
            )
        self.forecaster        = joblib.load(model_path)
        self._history_days     = history_days
        # Store (pd.Timestamp, Percept) pairs; maxlen prevents unbounded growth
        # We need history_days prior + 1 current = history_days+1 total
        self._history: deque   = deque(maxlen=history_days + 1)
        self._last_forecast: Optional[float] = None

    # ------------------------------------------------------------------
    # perceive
    # ------------------------------------------------------------------

    def perceive(self, percept: Percept) -> None:
        """
        Update internal history with a new Percept.

        WHY store (Timestamp, Percept) pairs:
            We need to map calendar dates to PM2.5 values to build
            lag features correctly.  A plain list would require index
            arithmetic that is error-prone across gaps.
        """
        self._history.append((pd.Timestamp(percept.date), percept))

    # ------------------------------------------------------------------
    # forecast
    # ------------------------------------------------------------------

    def forecast(self) -> Optional[float]:
        """
        Predict PM2.5 for the NEXT calendar day given the current history.

        Returns the predicted PM2.5 (ug/m3) or None if the history buffer
        does not yet have enough entries to build all features.

        WHY use _build_inference_features instead of build_model_features:
            See module docstring.  build_model_features drops the query row
            because it requires pm25[t+1] as target.  We use
            _build_inference_features which builds the same feature vector
            without needing a target.
        """
        if len(self._history) < self._history_days + 1:
            return None   # not enough history yet

        # --- Build mini-DataFrame from history ---
        dates    = [ts for ts, _ in self._history]
        percepts = [p  for _, p  in self._history]

        rows = []
        for ts, p in zip(dates, percepts):
            rows.append({
                "date":                        ts,
                "pm25_ugm3":                   p.pm25,
                "temperature_2m_mean":         p.temperature,
                "relative_humidity_2m_mean":   p.humidity,
                "wind_speed_10m_max":          p.wind_speed,
                "precipitation_sum":           p.rainfall,
                "wind_direction_10m_dominant": 0.0,   # not in Percept/MODEL_FEATURES
            })

        mini_df = pd.DataFrame(rows).set_index("date").sort_index()
        mini_df.index.name = "date"

        try:
            feat_df = _build_inference_features(mini_df)
        except Exception:
            return None

        # Take the LAST row (query date = most recent percept)
        last_row = feat_df.iloc[[-1]]
        X        = last_row[MODEL_FEATURES]

        # Verify no NaN: a gap in history produces NaN lags
        if X.isnull().any().any():
            return None

        pred = self.forecaster.predict(X)
        self._last_forecast = float(pred[0])
        return self._last_forecast

    # ------------------------------------------------------------------
    # advise
    # ------------------------------------------------------------------

    def advise(self, pm25: Optional[float] = None) -> dict:
        """
        Map a PM2.5 forecast to an AQI category and advisory text.

        Parameters
        ----------
        pm25 : PM2.5 value (ug/m3).  If None, uses self._last_forecast.
        """
        value = pm25 if pm25 is not None else self._last_forecast
        if value is None:
            return {
                "category":   "Unknown",
                "advisory":   "No forecast available yet.",
                "disclaimer": AQI_ADVISORY_DISCLAIMER,
            }
        return get_advisory(value)

    # ------------------------------------------------------------------
    # forecast_for_date  (high-level helper used by main.py)
    # ------------------------------------------------------------------

    def forecast_for_date(
        self,
        query_date: pd.Timestamp,
        full_df: pd.DataFrame,
    ) -> dict:
        """
        Produce a next-day forecast for query_date using data from full_df.

        Steps
        -----
        1. Verify query_date is in the dataset.
        2. Verify all 7 prior calendar days are also present (no gaps).
        3. Build a mini-DataFrame (8 rows: 7 prior + query_date).
        4. Call _build_inference_features and extract the last row.
        5. Call self.forecaster.predict() on that row.

        Parameters
        ----------
        query_date : the "today" date; we forecast pm25[t+1].
        full_df    : full merged DataFrame from lahore_merged.csv.

        Returns
        -------
        dict with keys: query_date, forecast_date, pm25_forecast,
                        category, advisory, disclaimer.

        Raises ValueError with a human-readable message on any problem.
        """
        query_date = pd.Timestamp(query_date)
        present    = set(full_df.index)

        if query_date not in present:
            raise ValueError(
                f"{query_date.date()} is not in the cleaned dataset.  "
                "It may have been a data-gap day or outside the data range."
            )

        # Gather the 7 prior calendar days + query_date itself (8 days total)
        window_start = query_date - pd.Timedelta(days=self._history_days)
        window_dates = pd.date_range(window_start, query_date, freq="D")
        missing      = [d for d in window_dates if d not in present]

        if missing:
            raise ValueError(
                f"Missing calendar days in the {self._history_days}-day window "
                f"before {query_date.date()}: "
                f"{[str(d.date()) for d in missing]}.\n"
                f"Cannot build a valid feature vector without a complete history."
            )

        # Build mini-DataFrame and features
        mini_df = full_df.loc[list(window_dates)].copy()
        feat_df = _build_inference_features(mini_df)

        last_row = feat_df.loc[[query_date]]
        X        = last_row[MODEL_FEATURES]

        if X.isnull().any().any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(
                f"Feature vector for {query_date.date()} has NaN in: {nan_cols}.\n"
                "The 7-day history may contain calendar gaps."
            )

        pm25_hat = float(self.forecaster.predict(X)[0])
        self._last_forecast = pm25_hat

        advice        = self.advise(pm25_hat)
        forecast_date = query_date + pd.Timedelta(days=1)

        return {
            "query_date":    str(query_date.date()),
            "forecast_date": str(forecast_date.date()),
            "pm25_forecast": round(pm25_hat, 2),
            "category":      advice["category"],
            "advisory":      advice["advisory"],
            "disclaimer":    advice["disclaimer"],
        }