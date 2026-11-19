"""
forecasters.py -- Forecaster hierarchy for Lahore PM2.5 prediction.

All models predict PM2.5 (ug/m3) at t+horizon given information
available at the END of day t (i.e. the value at t itself is known,
but t+1 is the target).

Design principles
-----------------
* Abstract base class enforces the fit / predict interface so the
  evaluation loop in run_phase3.py treats all models identically.
* Baselines (Persistence, MovingAverage) require no fitting; they are
  included so every trained model can be compared against a trivial baseline.
* RidgeForecaster wraps scaler + Ridge in an sklearn Pipeline so the
  StandardScaler statistics are computed only on training rows -- fitting
  it separately on train+val and then applying to test would be correct too,
  but a Pipeline makes the boundary explicit and prevents accidents.
* All feature engineering that requires calendar-day awareness is centralised
  in build_model_features() so forecasters never re-implement it.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    AQI_BREAKPOINTS,
    HORIZON_1,
    MODEL_CYCLIC_COLS,
    MODEL_FEATURES,
    MODEL_LAG_COLS,
    MODEL_ROLL_COLS,
    MODEL_WEATHER_COLS,
    RANDOM_SEED,
)
from src.dataset import _calendar_shift, _calendar_rolling


# ---------------------------------------------------------------------------
# AQI category helper (ug/m3 -> category label)
# ---------------------------------------------------------------------------

def pm25_to_aqi_category(values: np.ndarray) -> list[str]:
    """
    Map PM2.5 ug/m3 values to US EPA AQI category labels.
    Used for the macro-F1 metric in evaluation.
    """
    cats = []
    for v in values:
        label = AQI_BREAKPOINTS[-1][4]          # default to last category
        for _, _, Cp_lo, Cp_hi, lbl, _ in AQI_BREAKPOINTS:
            if v <= Cp_hi:
                label = lbl
                break
        cats.append(label)
    return cats


# ---------------------------------------------------------------------------
# Feature engineering (calendar-day safe)
# ---------------------------------------------------------------------------

def build_model_features(df: pd.DataFrame, horizon: int = HORIZON_1) -> pd.DataFrame:
    """
    Build the full feature matrix and target vector for a given forecast horizon.

    Inputs
    ------
    df      : DataFrame with pm25_ugm3 and weather columns, DatetimeIndex.
    horizon : Number of calendar days ahead to forecast (1 or 3).

    Returns
    -------
    DataFrame with columns MODEL_FEATURES + ["target"], rows where all
    features and target are non-NaN.

    Calendar-day correctness
    ------------------------
    Lags 4, 5, 6 are not in the Phase 1 CSV; we build them here using
    _calendar_shift() which densifies to a full daily grid before shifting,
    so gaps propagate NaN rather than silently using stale row values.

    Leakage guard
    -------------
    The target at horizon h is pm25_ugm3 shifted by -h (forward in time).
    All features use only information available at time t or earlier
    (lags >= 1, rolling window excludes t via shift(1) inside _calendar_rolling).
    """
    out = df.copy()

    # -- Lags 4, 5, 6 (not stored in the CSV, build on the fly) --
    for lag in [4, 5, 6]:
        out[f"pm25_lag{lag}"] = _calendar_shift(out["pm25_ugm3"], lag)

    # pm25_roll7 is already in the CSV from Phase 1; keep it.
    # If for any reason it's absent, rebuild it.
    if "pm25_roll7" not in out.columns:
        out["pm25_roll7"] = _calendar_rolling(out["pm25_ugm3"], 7)

    # -- Cyclic calendar encoding --
    # WHY sine/cosine: wraps the circular calendar so models see that
    # month 12 and month 1 are adjacent, not 11 apart numerically.
    out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12)
    out["doy_sin"]   = np.sin(2 * np.pi * out.index.day_of_year / 365)
    out["doy_cos"]   = np.cos(2 * np.pi * out.index.day_of_year / 365)

    # -- Target: PM2.5 h calendar days into the future --
    # We use _calendar_shift with a negative lag (shift backwards = forward look).
    # WHY calendar shift: if there is a gap at t+h, the target is NaN and
    # that row will be dropped, which is correct -- we have no ground truth.
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="D")
    pm_dense = out["pm25_ugm3"].reindex(full_idx)
    target_dense = pm_dense.shift(-horizon)
    out["target"] = target_dense.reindex(out.index)

    # -- Select and drop NaN rows --
    cols_needed = MODEL_FEATURES + ["target"]
    # Warn if any MODEL_FEATURES column is entirely missing
    missing_cols = [c for c in MODEL_FEATURES if c not in out.columns]
    if missing_cols:
        raise ValueError(f"build_model_features: missing columns {missing_cols}")

    result = out[cols_needed].dropna()
    return result


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Forecaster(ABC):
    """
    Abstract base for all PM2.5 forecasters.

    WHY an ABC: enforces that every subclass implements fit() and predict()
    with consistent signatures, so the evaluation loop in run_phase3.py
    can iterate over a list of forecasters without knowing their internals.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        """Train on (X, y).  Must return self for chaining."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions aligned with X.index."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Baseline: Persistence
# ---------------------------------------------------------------------------

class PersistenceForecaster(Forecaster):
    """
    Naive persistence baseline: predict tomorrow's PM2.5 == today's PM2.5.

    WHY include this: it is the cheapest possible model and sets a floor
    for the skill score (skill = 0 means no improvement over persistence).
    Any model with skill < 0 is actively worse than doing nothing.

    For horizon h, predict pm25_ugm3[t] as the forecast for pm25_ugm3[t+h].
    The feature matrix X must contain the column "pm25_lag1" for h=1
    (which equals pm25[t]) or "pm25_lag{h}" for general h.
    We use pm25_lag1 always (value at t) regardless of horizon, because
    for multi-step persistence the forecast is still "last known value".
    """

    def __init__(self, horizon: int = HORIZON_1):
        super().__init__(name=f"Persistence_h{horizon}")
        self.horizon = horizon

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceForecaster":
        # No parameters to learn.
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # pm25_lag1 == pm25[t], the last observed value.
        # For h>1 this is still the standard persistence forecast.
        if "pm25_lag1" not in X.columns:
            raise ValueError("PersistenceForecaster needs 'pm25_lag1' in X.")
        return X["pm25_lag1"].values


# ---------------------------------------------------------------------------
# Baseline: Moving Average
# ---------------------------------------------------------------------------

class MovingAverageForecaster(Forecaster):
    """
    7-day moving average baseline: predict tomorrow == mean of last 7 days.

    WHY 7 days: matches the strongest autocorrelation window (EDA: lag-7 r=0.64).
    Uses pm25_roll7 which is already built calendar-day-correctly in the CSV.
    """

    def __init__(self, horizon: int = HORIZON_1):
        super().__init__(name=f"MovingAverage7_h{horizon}")
        self.horizon = horizon

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MovingAverageForecaster":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "pm25_roll7" not in X.columns:
            raise ValueError("MovingAverageForecaster needs 'pm25_roll7' in X.")
        return X["pm25_roll7"].values


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

class RidgeForecaster(Forecaster):
    """
    StandardScaler + Ridge regression in one sklearn Pipeline.

    WHY a Pipeline: ensures the scaler sees only training data.
    If we fit scaler and Ridge separately we risk accidentally calling
    scaler.fit() on validation or test rows.  The Pipeline makes the
    training boundary a hard constraint: fit() on train, transform() on all.

    WHY Ridge over OLS: PM2.5 features are correlated (lag1 ~ lag2 ~ roll7),
    so the OLS normal equations are ill-conditioned.  Ridge adds L2 penalty
    alpha * ||w||^2 which shrinks correlated coefficients towards each other
    rather than amplifying noise.
    """

    def __init__(self, alpha: float = 1.0, horizon: int = HORIZON_1):
        super().__init__(name=f"Ridge_a{alpha}_h{horizon}")
        self.alpha   = alpha
        self.horizon = horizon
        self._pipe: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecaster":
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=self.alpha, fit_intercept=True)),
        ])
        self._pipe.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("Call fit() before predict().")
        return self._pipe.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        """Coefficients in the scaled feature space (after StandardScaler)."""
        return self._pipe.named_steps["ridge"].coef_

    @property
    def feature_names(self) -> list[str]:
        return list(self._pipe.feature_names_in_) if hasattr(
            self._pipe, "feature_names_in_") else []


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestForecaster(Forecaster):
    """
    Random Forest regressor.

    WHY Random Forest:
      RF handles nonlinear interactions (e.g. high lag1 AND low wind
      speed -> very high pm25) that Ridge cannot capture.  It also provides
      free feature importances and is robust to the skewed target distribution.

    WHY NOT XGBoost / LightGBM here:
      Those are Phase 4 candidates.  RF establishes a solid nonlinear baseline
      without additional dependencies.

    min_samples_leaf > 1 prevents overfitting on the ~900 effective training
    rows (after NaN-drops); we tune this in tuning.py.
    """

    def __init__(
        self,
        n_estimators:     int = 200,
        max_depth:        int | None = None,
        min_samples_leaf: int = 5,
        random_state:     int = RANDOM_SEED,
        horizon:          int = HORIZON_1,
    ):
        super().__init__(name=f"RandomForest_h{horizon}")
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.horizon          = horizon
        self._model: RandomForestRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestForecaster":
        self._model = RandomForestRegressor(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            random_state     = self.random_state,
            n_jobs           = -1,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_