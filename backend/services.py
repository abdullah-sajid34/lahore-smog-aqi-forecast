"""
services.py -- All business logic for the API, loaded once at startup.

WHY load everything once:
  joblib models (RF = 3.8 MB) and the 229 KB kalman_series.json take
  measurable time to load.  Loading them in a FastAPI lifespan context
  means every request reads from in-memory objects -- sub-millisecond
  vs ~300 ms for cold load.

Missing-file strategy:
  At startup we scan for all required files.  Any that are absent are
  listed in AppState.missing_files.  Endpoints that need a missing file
  raise HTTPException(503) with a message telling the user to run
  python src/run_experiments.py.  This prevents silent 500 errors.

Feature-building:
  _build_inference_features() from src/agent.py is the single source of
  truth for inference-time features.  Services import and call it directly
  -- no duplicated feature logic.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException

from src.agent import _build_inference_features, get_advisory, pm25_to_category
from src.config import (
    AQI_ADVISORY_DISCLAIMER,
    MERGED_CSV,
    MODEL_FEATURES,
    OUTPUTS_MODELS,
    OUTPUTS_RESULTS,
)

import joblib


# ---------------------------------------------------------------------------
# File manifest -- every file the API depends on
# ---------------------------------------------------------------------------

REQUIRED_RESULTS = [
    "kpis.json",
    "eda_summary.json",
    "model_comparison.json",
    "predictions_test.json",
    "confusion_matrix.json",
    "feature_importance.json",
    "rf_seeds.json",
    "kalman_series.json",
    "kalman_params.json",
    "phase4_results.json",
]
REQUIRED_MODELS = ["rf_h1.joblib", "ridge_h1.joblib"]


# ---------------------------------------------------------------------------
# AppState  (populated once at startup)
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    missing_files:  list[str]        = field(default_factory=list)
    kpis:           dict             = field(default_factory=dict)
    eda:            dict             = field(default_factory=dict)
    model_comparison: list          = field(default_factory=list)
    predictions_test: list          = field(default_factory=list)
    confusion_matrix: dict          = field(default_factory=dict)
    feature_importance: dict        = field(default_factory=dict)
    rf_seeds:       dict             = field(default_factory=dict)
    kalman_series:  list            = field(default_factory=list)
    kalman_params:  dict             = field(default_factory=dict)
    phase4_results: dict             = field(default_factory=dict)
    df:             Optional[pd.DataFrame] = None   # lahore_merged.csv
    rf_model:       Any              = None
    ridge_model:    Any              = None
    # Pre-built index structures for fast date lookups
    df_index_set:   set              = field(default_factory=set)
    test_rmse:      float            = 68.4641   # persistence baseline, fallback


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_app_state() -> AppState:
    """
    Load all results files and models into memory.
    Records missing files but does not raise -- endpoints check individually.
    """
    state       = AppState()
    missing     = []

    # -- JSON results --
    for fname in REQUIRED_RESULTS:
        p = OUTPUTS_RESULTS / fname
        if not p.exists():
            missing.append(str(p))

    # -- CSV --
    if not MERGED_CSV.exists():
        missing.append(str(MERGED_CSV))

    # -- Models --
    for mname in REQUIRED_MODELS:
        p = OUTPUTS_MODELS / mname
        if not p.exists():
            missing.append(str(p))

    state.missing_files = missing

    # Load what we can even if some files are absent
    def _try_json(fname):
        p = OUTPUTS_RESULTS / fname
        return _load_json(p) if p.exists() else {}

    def _try_json_list(fname):
        p = OUTPUTS_RESULTS / fname
        return _load_json(p) if p.exists() else []

    state.kpis             = _try_json("kpis.json")
    state.eda              = _try_json("eda_summary.json")
    state.model_comparison = _try_json_list("model_comparison.json")
    state.predictions_test = _try_json_list("predictions_test.json")
    state.confusion_matrix = _try_json("confusion_matrix.json")
    state.feature_importance = _try_json("feature_importance.json")
    state.rf_seeds         = _try_json("rf_seeds.json")
    state.phase4_results   = _try_json("phase4_results.json")
    state.kalman_params    = _try_json("kalman_params.json")

    ks_full = _try_json("kalman_series.json")
    state.kalman_series    = ks_full.get("series", []) if isinstance(ks_full, dict) else []

    # Derive test RMSE for confidence intervals
    if state.kpis:
        state.test_rmse = state.kpis.get("persistence_rmse_test", 68.4641)

    # Load merged CSV
    if MERGED_CSV.exists():
        state.df          = pd.read_csv(
            MERGED_CSV, parse_dates=["date"], index_col="date"
        ).sort_index()
        state.df_index_set = set(state.df.index)

    # Load models
    rf_path    = OUTPUTS_MODELS / "rf_h1.joblib"
    ridge_path = OUTPUTS_MODELS / "ridge_h1.joblib"
    if rf_path.exists():
        state.rf_model    = joblib.load(rf_path)
    if ridge_path.exists():
        state.ridge_model = joblib.load(ridge_path)

    return state


# ---------------------------------------------------------------------------
# Shared guard helper
# ---------------------------------------------------------------------------

def _require(state: AppState, files: list[str]) -> None:
    """Raise 503 if any of the listed filenames are in state.missing_files."""
    blocked = [f for f in state.missing_files
               if any(f.endswith(req) for req in files)]
    if blocked:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Required files are missing: {blocked}. "
                "Run: python src/run_experiments.py"
            ),
        )


# ---------------------------------------------------------------------------
# Service functions (called by route handlers in main.py)
# ---------------------------------------------------------------------------

def svc_health(state: AppState) -> dict:
    return {
        "status":        "ok" if not state.missing_files else "degraded",
        "version":       "1.0.0",
        "missing_files": state.missing_files,
    }


def svc_kpis(state: AppState) -> dict:
    _require(state, ["kpis.json"])
    return state.kpis


def svc_today(state: AppState) -> dict:
    _require(state, [str(MERGED_CSV)])
    df  = state.df
    row = df.iloc[-1]
    return {
        "date":         str(df.index[-1].date()),
        "pm25_ugm3":    round(float(row["pm25_ugm3"]), 2),
        "aqi_category": pm25_to_category(float(row["pm25_ugm3"])),
        "temperature":  round(float(row["temperature_2m_mean"]), 1),
        "humidity":     round(float(row["relative_humidity_2m_mean"]), 1),
        "wind_speed":   round(float(row["wind_speed_10m_max"]), 1),
        "rainfall":     round(float(row["precipitation_sum"]), 2),
    }


def svc_series(
    state: AppState,
    start: Optional[str],
    end: Optional[str],
) -> dict:
    _require(state, [str(MERGED_CSV)])
    df = state.df

    # Date parsing with clear 400 on invalid input
    try:
        start_ts = pd.Timestamp(start) if start else df.index.min()
        end_ts   = pd.Timestamp(end)   if end   else df.index.max()
    except Exception:
        raise HTTPException(400, detail="Invalid date format. Use YYYY-MM-DD.")

    if start_ts > end_ts:
        raise HTTPException(400, detail="start must be <= end.")
    if start_ts > df.index.max():
        raise HTTPException(
            404,
            detail=f"start={start_ts.date()} is after the last available "
                   f"date ({df.index.max().date()}).",
        )
    if end_ts < df.index.min():
        raise HTTPException(
            404,
            detail=f"end={end_ts.date()} is before the first available "
                   f"date ({df.index.min().date()}).",
        )

    # Clip to available range
    start_ts = max(start_ts, df.index.min())
    end_ts   = min(end_ts,   df.index.max())

    subset = df.loc[start_ts:end_ts, ["pm25_ugm3"]]
    series = [
        {
            "date":         str(d.date()),
            "pm25_ugm3":    round(float(row["pm25_ugm3"]), 2),
            "aqi_category": pm25_to_category(float(row["pm25_ugm3"])),
        }
        for d, row in subset.iterrows()
    ]
    return {
        "start": str(start_ts.date()),
        "end":   str(end_ts.date()),
        "count": len(series),
        "series": series,
    }


def svc_forecast(
    state: AppState,
    date_str: str,
    model_name: str,
) -> dict:
    """
    Run the REAL trained model on the real features for `date_str`.

    WHY reuse _build_inference_features (from src/agent.py):
      That function is the single source of truth for inference-time features.
      It applies the identical _calendar_shift, _calendar_rolling, and cyclic
      encoding as training (build_model_features in forecasters.py) but does
      NOT require a target value, so the query-date row is never dropped.

    Confidence range:
      RF:    std of individual tree predictions (each tree gives one value,
             spread reflects model uncertainty on this input)
      Ridge: test-period RMSE used as one-sigma proxy -- the model is
             deterministic so no internal spread exists.
    """
    _require(state, [str(MERGED_CSV), "rf_h1.joblib", "ridge_h1.joblib"])

    # Validate model name
    if model_name not in ("ridge", "random_forest"):
        raise HTTPException(
            400,
            detail="model must be 'ridge' or 'random_forest'.",
        )

    # Parse date
    try:
        query_date = pd.Timestamp(date_str)
        if pd.isna(query_date):
            raise ValueError()
    except Exception:
        raise HTTPException(400, detail=f"Invalid date '{date_str}'. Use YYYY-MM-DD.")

    df      = state.df
    present = state.df_index_set

    if query_date not in present:
        raise HTTPException(
            404,
            detail=(
                f"Date {query_date.date()} not found in the cleaned dataset. "
                "It may be a data-gap day or outside the available range "
                f"({df.index.min().date()} to {df.index.max().date()})."
            ),
        )

    if query_date >= df.index.max():
        raise HTTPException(
            400,
            detail=(
                f"Date {query_date.date()} is the last available date. "
                "Cannot forecast beyond the last known observation."
            ),
        )

    # Check 7-day window
    window_start = query_date - pd.Timedelta(days=7)
    window_dates = pd.date_range(window_start, query_date, freq="D")
    missing_days = [d for d in window_dates if d not in present]
    if missing_days:
        raise HTTPException(
            400,
            detail=(
                f"Missing calendar days in the 7-day window before "
                f"{query_date.date()}: "
                f"{[str(d.date()) for d in missing_days]}. "
                "Cannot build a valid feature vector."
            ),
        )

    # Build features
    mini_df  = df.loc[list(window_dates)].copy()
    feat_df  = _build_inference_features(mini_df)
    last_row = feat_df.loc[[query_date]]
    X        = last_row[MODEL_FEATURES]

    if X.isnull().any().any():
        raise HTTPException(
            400,
            detail="Feature vector contains NaN. The 7-day window may have gaps.",
        )

    # Predict + confidence
    if model_name == "random_forest":
        model       = state.rf_model
        # Tree-spread confidence: collect per-tree predictions
        tree_preds  = np.array([
            tree.predict(X.values)[0]
            for tree in model._model.estimators_
        ])
        pm25_hat    = float(np.mean(tree_preds))
        spread      = float(np.std(tree_preds))
        conf_low    = round(max(0.0, pm25_hat - 1.645 * spread), 2)   # 90% CI
        conf_high   = round(pm25_hat + 1.645 * spread, 2)
        conf_method = "RF tree spread (90% CI: mean ± 1.645 × std of tree predictions)"
    else:
        model       = state.ridge_model
        pm25_hat    = float(model._pipe.predict(X)[0])
        # Ridge has no internal spread; use test-RMSE as proxy
        ridge_rmse  = 54.9426   # from phase3_results.json Ridge RMSE
        conf_low    = round(max(0.0, pm25_hat - 1.645 * ridge_rmse), 2)
        conf_high   = round(pm25_hat + 1.645 * ridge_rmse, 2)
        conf_method = "Ridge test-RMSE proxy (90% CI: forecast ± 1.645 × test RMSE)"

    pm25_hat = round(pm25_hat, 2)

    advice        = get_advisory(pm25_hat)
    forecast_date = query_date + pd.Timedelta(days=1)

    # Last 7 days history
    hist_start = query_date - pd.Timedelta(days=6)
    hist_df    = df.loc[hist_start:query_date, ["pm25_ugm3"]]
    history_7d = [
        {
            "date":         str(d.date()),
            "pm25_ugm3":    round(float(r["pm25_ugm3"]), 2),
            "aqi_category": pm25_to_category(float(r["pm25_ugm3"])),
        }
        for d, r in hist_df.iterrows()
    ]

    # Weather inputs used
    row_today = df.loc[query_date]
    weather   = {
        "temperature_2m_mean":       round(float(row_today["temperature_2m_mean"]),      1),
        "relative_humidity_2m_mean": round(float(row_today["relative_humidity_2m_mean"]),1),
        "wind_speed_10m_max":        round(float(row_today["wind_speed_10m_max"]),        1),
        "precipitation_sum":         round(float(row_today["precipitation_sum"]),         2),
    }

    return {
        "query_date":       str(query_date.date()),
        "forecast_date":    str(forecast_date.date()),
        "model":            model_name,
        "pm25_forecast":    pm25_hat,
        "aqi_category":     advice["category"],
        "advisory":         advice["advisory"],
        "disclaimer":       advice["disclaimer"],
        "confidence_low":   conf_low,
        "confidence_high":  conf_high,
        "confidence_method":conf_method,
        "history_7d":       history_7d,
        "weather_inputs":   weather,
    }


def svc_eda(state: AppState) -> dict:
    _require(state, ["eda_summary.json"])
    return state.eda


def svc_models(state: AppState) -> dict:
    _require(state, [
        "model_comparison.json",
        "predictions_test.json",
        "confusion_matrix.json",
        "feature_importance.json",
        "rf_seeds.json",
    ])
    return {
        "comparison":        state.model_comparison,
        "predictions_test":  state.predictions_test,
        "confusion_matrix":  state.confusion_matrix,
        "feature_importance":state.feature_importance,
        "seed_stability":    state.rf_seeds,
    }


def svc_kalman(state: AppState) -> dict:
    _require(state, ["kalman_series.json", "kalman_params.json", "phase4_results.json"])

    # Build comparison list from phase4_results
    p4 = state.phase4_results
    comparison = []
    for model_key, input_label in [
        ("ridge_raw",      "raw"),
        ("ridge_filtered", "kalman_filtered"),
        ("rf_raw",         "raw"),
        ("rf_filtered",    "kalman_filtered"),
    ]:
        entry = p4.get(model_key, {})
        if not entry:
            continue
        # rf entries have _mean/_std keys; ridge entries have direct values
        comparison.append({
            "model":    "Ridge"        if "ridge" in model_key else "RandomForest",
            "input":    input_label,
            "rmse":     entry.get("rmse",      entry.get("rmse_mean",   0.0)),
            "rmse_std": entry.get("rmse_std",  0.0),
            "mae":      entry.get("mae",       entry.get("mae_mean",    0.0)),
            "mae_std":  entry.get("mae_std",   0.0),
            "f1":       entry.get("f1",        entry.get("f1_mean",     0.0)),
            "f1_std":   entry.get("f1_std",    0.0),
            "skill":    entry.get("skill"),
        })

    return {
        "Q":          state.kalman_params.get("Q", 0.0),
        "R":          state.kalman_params.get("R", 0.0),
        "q_frac":     state.kalman_params.get("best_q_frac", 1.0),
        "series":     state.kalman_series,
        "comparison": comparison,
    }


def svc_about() -> dict:
    return {
        "title": "Lahore Smog Analysis and PM2.5 AQI Forecast System",
        "peas": {
            "performance_measure": (
                "RMSE (ug/m3) of next-day PM2.5 forecast; "
                "macro-F1 over US EPA AQI categories; "
                "skill score vs persistence baseline"
            ),
            "environment": (
                "Lahore, Punjab, Pakistan (31.52°N, 74.36°E). "
                "Daily time series 2019-05-09 to 2025-02-18. "
                "Non-stationary: strong seasonal smog cycle (Oct–Jan), "
                "calendar data gaps due to sensor outages."
            ),
            "actuators": (
                "REST API responses consumed by the web dashboard; "
                "advisory text presented to end users."
            ),
            "sensors": (
                "US Embassy Lahore PM2.5 monitor (via AQICN, US AQI scale). "
                "Open-Meteo ERA5 reanalysis: temperature, humidity, "
                "wind speed, precipitation."
            ),
        },
        "data_sources": [
            {
                "name":        "AQICN — US Embassy Lahore PM2.5",
                "url":         "https://aqicn.org/city/pakistan/lahore/us-embassy/",
                "description": "Daily US AQI readings from the US Consulate monitor in Lahore.",
                "attribution": "Data sourced from AQICN (aqicn.org). "
                               "Values are US EPA AQI converted to PM2.5 ug/m3 "
                               "via the inverse piecewise-linear EPA formula.",
            },
            {
                "name":        "Open-Meteo Historical Weather API",
                "url":         "https://open-meteo.com/",
                "description": "Free ERA5-based daily weather reanalysis. "
                               "No API key required.",
                "attribution": "Weather data from Open-Meteo (open-meteo.com), "
                               "ERA5 reanalysis © ECMWF.",
            },
            {
                "name":        "US EPA AQI Technical Assistance Document",
                "url":         "https://www.airnow.gov/sites/default/files/2020-05/"
                               "aqi-technical-assistance-document-sept2018.pdf",
                "description": "Official breakpoints used for AQI-to-PM2.5 conversion.",
                "attribution": "US Environmental Protection Agency.",
            },
        ],
        "team": [
            {"name": "Abdul Rehman", "role": "Lead Developer"},
        ],
        "disclaimer": AQI_ADVISORY_DISCLAIMER,
    }