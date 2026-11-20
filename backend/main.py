from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AboutResponse, EdaResponse, ForecastResponse, HealthResponse,
    KalmanResponse, KpisResponse, ModelsResponse, SeriesResponse, TodayResponse,
)
from backend.services import (
    AppState, build_app_state,
    svc_about, svc_eda, svc_forecast, svc_health, svc_kalman,
    svc_kpis, svc_models, svc_series, svc_today,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and result files once at startup."""
    app.state.ctx = build_app_state()
    yield


app = FastAPI(
    title="Lahore PM2.5 AQI Forecast API",
    description="REST API for the Lahore Smog Analysis and AQI Forecast project.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _state() -> AppState:
    return app.state.ctx


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check. Returns ok when all required files are present."""
    return svc_health(_state())


@app.get("/api/kpis", response_model=KpisResponse, tags=["Data"])
def kpis():
    """Annual means, peak date, best model RMSE, split sizes."""
    return svc_kpis(_state())


@app.get("/api/today", response_model=TodayResponse, tags=["Data"])
def today():
    """Latest available observation."""
    return svc_today(_state())


@app.get("/api/series", response_model=SeriesResponse, tags=["Data"])
def series(
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end:   Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Daily PM2.5 series with AQI category."""
    return svc_series(_state(), start, end)


@app.get("/api/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast(
    date:  str = Query(..., description="Query date YYYY-MM-DD"),
    model: str = Query("random_forest", description="ridge or random_forest"),
):
    """Next-day forecast with confidence range, history, and weather inputs."""
    return svc_forecast(_state(), date, model)


@app.get("/api/eda", response_model=EdaResponse, tags=["Analytics"])
def eda():
    """EDA statistics from the training split."""
    return svc_eda(_state())


@app.get("/api/models", response_model=ModelsResponse, tags=["Analytics"])
def models():
    """Model comparison, predictions, confusion matrix, feature importances, seed stability."""
    return svc_models(_state())


@app.get("/api/kalman", response_model=KalmanResponse, tags=["Analytics"])
def kalman():
    """Kalman filter Q/R params, full series, raw-vs-filtered comparison."""
    return svc_kalman(_state())


@app.get("/api/about", response_model=AboutResponse, tags=["System"])
def about():
    """PEAS, data sources, team, disclaimer."""
    return svc_about()
