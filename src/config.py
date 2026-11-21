"""
config.py -- central configuration for the Lahore Smog project.

All paths, constants, and feature lists live here so every other module
imports from one place.
"""

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_RESULTS = PROJECT_ROOT / "outputs" / "results"
OUTPUTS_MODELS  = PROJECT_ROOT / "outputs" / "models"

AQI_CSV    = DATA_RAW      / "lahore-us-embassy-air-quality.csv"
MERGED_CSV = DATA_PROCESSED / "lahore_merged.csv"
WEATHER_CSV = DATA_RAW     / "lahore_weather.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# US EPA PM2.5 AQI breakpoints
# (I_lo, I_hi, Cp_lo, Cp_hi, label, hex_color)
# ---------------------------------------------------------------------------
AQI_BREAKPOINTS = [
    (  0,  50,   0.0,  12.0, "Good",                   "#00E400"),
    ( 51, 100,  12.1,  35.4, "Moderate",                "#FFFF00"),
    (101, 150,  35.5,  55.4, "Unhealthy for Sensitive",  "#FF7E00"),
    (151, 200,  55.5, 150.4, "Unhealthy",               "#FF0000"),
    (201, 300, 150.5, 250.4, "Very Unhealthy",          "#8F3F97"),
    (301, 400, 250.5, 350.4, "Hazardous",               "#7E0023"),
    (401, 500, 350.5, 500.4, "Hazardous+",              "#7E0023"),
]

AQI_CATEGORY_COLORS = {bp[4]: bp[5] for bp in AQI_BREAKPOINTS}
AQI_CATEGORY_LABELS = [bp[4] for bp in AQI_BREAKPOINTS]

# ---------------------------------------------------------------------------
# Feature lists (Phase 1 / EDA)
# ---------------------------------------------------------------------------
WEATHER_FEATURES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",
]

LAG_FEATURES = [
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
    "pm25_lag7",
]

ROLLING_FEATURES = [
    "pm25_roll3",
    "pm25_roll7",
]

CALENDAR_FEATURES = [
    "month",
    "day_of_year",
    "day_of_week",
    "is_smog_season",
]

ALL_FEATURES = WEATHER_FEATURES + LAG_FEATURES + ROLLING_FEATURES + CALENDAR_FEATURES

# ---------------------------------------------------------------------------
# Phase 3 model feature set
#
# WHY month_sin / month_cos instead of raw month integer:
#   Month 12 and month 1 are neighbours on the calendar but are 11 apart
#   numerically, so a linear model would treat them as far apart.  Sine /
#   cosine encoding wraps the circle so December and January are close.
#   The same logic applies to day_of_year for capturing intra-year seasonality.
#
# WHY include pm25_lag1..lag6 (not just lag1,2,3,7):
#   The EDA showed autocorrelation stays above 0.7 through lag 6.
#   Lags 4-6 add information beyond lag3 without introducing lag7 at the
#   cost of a large NaN footprint (~6% of rows lose lag7 to calendar gaps).
# ---------------------------------------------------------------------------
MODEL_LAG_COLS = [
    "pm25_lag1", "pm25_lag2", "pm25_lag3",
    "pm25_lag4", "pm25_lag5", "pm25_lag6",   # added for Phase 3
]
MODEL_ROLL_COLS = ["pm25_roll7"]              # 7-day rolling mean (calendar-safe)
MODEL_WEATHER_COLS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
]

# Engineered cyclic features built in forecasters.build_model_features()
MODEL_CYCLIC_COLS = [
    "month_sin", "month_cos",
    "doy_sin",   "doy_cos",
]

MODEL_FEATURES = (
    MODEL_LAG_COLS
    + MODEL_ROLL_COLS
    + MODEL_WEATHER_COLS
    + MODEL_CYCLIC_COLS
)

# ---------------------------------------------------------------------------
# Forecast horizons
# ---------------------------------------------------------------------------
HORIZON_1 = 1   # t+1 (primary)
HORIZON_3 = 3   # t+3 (bonus)

# ---------------------------------------------------------------------------
# Split fractions
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# ---------------------------------------------------------------------------
# Outlier / cleaning
# ---------------------------------------------------------------------------
OUTLIER_IQR_K     = 3.0
MAX_INTERP_GAP_DAYS = 3

# ---------------------------------------------------------------------------
# Tuning grids (small, runs in a few minutes)
# WHY these ranges:
#   Ridge alpha: log-spaced 0.01-100 covers underfitting to heavy regularisation.
#   RF n_estimators 100-300: enough for stable OOB estimates; >300 adds runtime
#     without meaningful accuracy gain on ~1 k training rows.
#   RF max_depth None/8/12: None grows full trees (high variance);
#     8-12 is a typical sweet-spot for tabular daily data.
#   RF min_samples_leaf 1/5/10: controls minimum node size; 5-10 reduces
#     overfitting on the relatively small dataset.
# ---------------------------------------------------------------------------
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

RF_PARAM_GRID = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [None, 8, 12],
    "min_samples_leaf":[1, 5, 10],
}

# Walk-forward CV splits for tuning (TimeSeriesSplit)
N_CV_SPLITS = 5
# ---------------------------------------------------------------------------
# Phase 6 — Advisory texts for each AQI category
#
# WHY these texts:
#   Each advisory is a one-line, plain-English action recommendation.
#   They mirror the guidance in the US AQI communication guide and
#   Pakistan EPA alerts, calibrated for Lahore's population (a large fraction
#   of outdoor workers and children).
#
# DISCLAIMER: added automatically by advise(); never omit it.
# ---------------------------------------------------------------------------
AQI_ADVISORY_DISCLAIMER = (
    "Forecasts are informational only and not an official health advisory."
)

AQI_ADVISORY_TEXTS = {
    "Good": (
        "Air quality is satisfactory; enjoy outdoor activities."
    ),
    "Moderate": (
        "Acceptable air quality; unusually sensitive individuals should limit "
        "prolonged outdoor exertion."
    ),
    "Unhealthy for Sensitive": (
        "Sensitive groups (children, elderly, respiratory patients) should "
        "limit outdoor activity."
    ),
    "Unhealthy": (
        "Limit outdoor activity; sensitive groups should stay indoors."
    ),
    "Very Unhealthy": (
        "Avoid outdoor activity; everyone should reduce time spent outside."
    ),
    "Hazardous": (
        "Health emergency — stay indoors, keep windows closed, use air "
        "purifiers if available."
    ),
    "Hazardous+": (
        "Extreme health emergency — do not go outside; seek clean indoor air "
        "immediately."
    ),
}

# Model file used by the agent (best model by validation RMSE)
AGENT_MODEL_FILE = OUTPUTS_MODELS / "rf_h1.joblib"