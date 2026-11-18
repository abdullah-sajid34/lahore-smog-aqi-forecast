"""
config.py -- central configuration for the Lahore Smog project.

All paths, constants, and feature lists live here so every other module
imports from one place. That prevents magic strings from scattering across
files and makes it trivial to change a path or threshold in one shot.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # lahore-smog/
DATA_RAW      = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_RESULTS = PROJECT_ROOT / "outputs" / "results"
OUTPUTS_MODELS  = PROJECT_ROOT / "outputs" / "models"

# Raw CSV from AQICN (US Embassy Lahore monitor)
AQI_CSV = DATA_RAW / "lahore-us-embassy-air-quality.csv"

# Processed / merged dataset written by dataset.py
MERGED_CSV = DATA_PROCESSED / "lahore_merged.csv"

# Weather file downloaded by download_weather.py
WEATHER_CSV = DATA_RAW / "lahore_weather.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# US EPA PM2.5 AQI breakpoints
# Reference: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
#
# Each tuple: (I_lo, I_hi, Cp_lo, Cp_hi, label, hex_color)
#   I_lo/I_hi   - AQI integer range (inclusive)
#   Cp_lo/Cp_hi - corresponding PM2.5 concentration range (ug/m3, 24-hr avg)
#
# WHY store breakpoints here: the inverse formula (AQI -> ug/m3) and the
# forward formula (ug/m3 -> AQI) are both needed; keeping the table in one
# place avoids duplication and drift between modules.
# ---------------------------------------------------------------------------
AQI_BREAKPOINTS = [
    (  0,  50,   0.0,  12.0, "Good",                  "#00E400"),
    ( 51, 100,  12.1,  35.4, "Moderate",               "#FFFF00"),
    (101, 150,  35.5,  55.4, "Unhealthy for Sensitive", "#FF7E00"),
    (151, 200,  55.5, 150.4, "Unhealthy",              "#FF0000"),
    (201, 300, 150.5, 250.4, "Very Unhealthy",         "#8F3F97"),
    (301, 400, 250.5, 350.4, "Hazardous",              "#7E0023"),
    (401, 500, 350.5, 500.4, "Hazardous+",             "#7E0023"),
]

AQI_CATEGORY_COLORS = {bp[4]: bp[5] for bp in AQI_BREAKPOINTS}

# ---------------------------------------------------------------------------
# Feature list
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
# Train / val / test split fractions (time-based, never shuffled)
# WHY: air quality is a time series; random splits leak future information.
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# ---------------------------------------------------------------------------
# Outlier rule
# WHY IQR-based: AQI distributions are right-skewed.  IQR with k=3 is
# conservative -- only flags extreme statistical outliers, not real high-smog.
# ---------------------------------------------------------------------------
OUTLIER_IQR_K = 3.0

# Max consecutive missing days to interpolate; drop runs longer than this.
MAX_INTERP_GAP_DAYS = 3