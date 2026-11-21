"""
schemas.py -- Pydantic response models for every API endpoint.

WHY Pydantic:
  - Validates outgoing data at the boundary, so a NaN or wrong type in the
    underlying result files becomes a clear 500 (or is handled upstream)
    rather than silent corruption in the client.
  - Auto-generates OpenAPI docs at /docs and /redoc for free.
  - model_config = ConfigDict(from_attributes=True) lets us build models
    directly from plain dicts with .model_validate().
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str           # "ok" | "degraded"
    version: str          # e.g. "1.0.0"
    missing_files: list[str]   # empty when status == "ok"


# ---------------------------------------------------------------------------
# /api/kpis
# ---------------------------------------------------------------------------

class KpisResponse(BaseModel):
    annual_mean_pm25_ugm3: dict[str, float]
    peak_pm25_ugm3: float
    peak_date: str
    best_model_by_val: str
    best_model_rmse_test: float
    best_model_skill_test: float
    persistence_rmse_test: float
    data_date_range_start: str
    data_date_range_end: str
    total_rows: int
    train_rows: int
    val_rows: int
    test_rows: int


# ---------------------------------------------------------------------------
# /api/today
# ---------------------------------------------------------------------------

class TodayResponse(BaseModel):
    date: str
    pm25_ugm3: float
    aqi_category: str
    temperature: float
    humidity: float
    wind_speed: float
    rainfall: float


# ---------------------------------------------------------------------------
# /api/series
# ---------------------------------------------------------------------------

class SeriesPoint(BaseModel):
    date: str
    pm25_ugm3: float
    aqi_category: str


class SeriesResponse(BaseModel):
    start: str
    end: str
    count: int
    series: list[SeriesPoint]


# ---------------------------------------------------------------------------
# /api/forecast
# ---------------------------------------------------------------------------

class WeatherInputs(BaseModel):
    temperature_2m_mean: float
    relative_humidity_2m_mean: float
    wind_speed_10m_max: float
    precipitation_sum: float


class HistoryPoint(BaseModel):
    date: str
    pm25_ugm3: float
    aqi_category: str


class ForecastResponse(BaseModel):
    query_date: str
    forecast_date: str
    model: str
    pm25_forecast: float
    aqi_category: str
    advisory: str
    disclaimer: str
    # Confidence range
    # WHY two sources:
    #   RF: spread of per-tree predictions (std across individual tree outputs)
    #   Ridge: test-period RMSE used as a proxy for one-sigma uncertainty
    confidence_low: float
    confidence_high: float
    confidence_method: str
    # Context
    history_7d: list[HistoryPoint]
    weather_inputs: WeatherInputs


# ---------------------------------------------------------------------------
# /api/eda
# ---------------------------------------------------------------------------

class EdaResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    monthly_mean_by_month: dict[str, float]
    peak_month: int
    peak_month_mean_ugm3: float
    weekday_means_ugm3: dict[str, float]
    max_weekday: str
    min_weekday: str
    weekday_range_ugm3: float
    smog_season_mean_ugm3: float
    off_season_mean_ugm3: float
    smog_vs_offseason_ratio: float
    smog_category_pct: dict[str, float]
    offseason_category_pct: dict[str, float]
    weather_correlations: dict[str, float]
    strongest_weather_corr_var: str
    strongest_weather_corr_value: float
    autocorrelations: dict[str, float]
    autocorr_lag1: float
    autocorr_lag7: float


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------

class ModelMetrics(BaseModel):
    model: str
    rmse: float
    rmse_std: float
    mae: float
    mae_std: float
    f1: float
    f1_std: float
    skill: Optional[float]


class PredictionPoint(BaseModel):
    date: str
    actual: float
    persistence: float
    moving_avg7: float
    ridge: float
    rf: float
    rf_kalman: Optional[float]


class ConfusionMatrixData(BaseModel):
    model: str
    labels: list[str]
    matrix: list[list[int]]


class FeatureImportanceData(BaseModel):
    model: str
    importances: dict[str, float]


class SeedStability(BaseModel):
    model: str
    seeds: list[int]
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    f1_mean: float
    f1_std: float
    per_seed: list[dict[str, Any]]


class ModelsResponse(BaseModel):
    comparison: list[ModelMetrics]
    predictions_test: list[PredictionPoint]
    confusion_matrix: ConfusionMatrixData
    feature_importance: FeatureImportanceData
    seed_stability: SeedStability


# ---------------------------------------------------------------------------
# /api/kalman
# ---------------------------------------------------------------------------

class KalmanPoint(BaseModel):
    date: str
    raw: float
    filtered: float
    variance: float


class KalmanRawVsFiltered(BaseModel):
    model: str
    input: str
    rmse: float
    rmse_std: float
    mae: float
    mae_std: float
    f1: float
    f1_std: float
    skill: Optional[float]


class KalmanResponse(BaseModel):
    Q: float
    R: float
    q_frac: float
    series: list[KalmanPoint]      # one point per day
    comparison: list[KalmanRawVsFiltered]


# ---------------------------------------------------------------------------
# /api/about
# ---------------------------------------------------------------------------

class DataSource(BaseModel):
    name: str
    url: str
    description: str
    attribution: str


class TeamMember(BaseModel):
    name: str
    role: str


class PEASDescription(BaseModel):
    performance_measure: str
    environment: str
    actuators: str
    sensors: str


class AboutResponse(BaseModel):
    title: str
    peas: PEASDescription
    data_sources: list[DataSource]
    team: list[TeamMember]
    disclaimer: str


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    detail: str