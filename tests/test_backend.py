"""
test_backend.py -- FastAPI TestClient tests for all API endpoints.

Every test:
  - Uses FastAPI TestClient (synchronous, no real server needed).
  - Verifies HTTP 200 and that the response parses as valid JSON.
  - Key fields are spot-checked against known values from the saved outputs.

The forecast test also verifies that the API forecast for 2024-11-01
equals the console-menu forecast for the same date (the same model,
the same feature builder, the same saved model file).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from backend.main import app

# Use context-manager form so the lifespan (startup/shutdown) fires,
# which populates app.state.ctx with models and result files.
# Without this, app.state.ctx is never set and every endpoint fails
# with AttributeError.
_cm = TestClient(app)
_cm.__enter__()
client = _cm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def check_outputs():
    """Skip all backend tests if the pipeline has not been run."""
    from src.config import OUTPUTS_RESULTS, OUTPUTS_MODELS, MERGED_CSV
    missing = []
    for f in ["kpis.json","model_comparison.json","kalman_series.json"]:
        if not (OUTPUTS_RESULTS / f).exists():
            missing.append(f)
    if not MERGED_CSV.exists():
        missing.append("lahore_merged.csv")
    if not (OUTPUTS_MODELS / "rf_h1.joblib").exists():
        missing.append("rf_h1.joblib")
    if missing:
        pytest.skip(f"Required files missing ({missing}). Run src/run_experiments.py")


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

def test_health_200():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] in ("ok", "degraded")
    assert "missing_files" in data
    assert "version" in data


def test_health_ok_when_all_files_present():
    r = client.get("/api/health")
    data = r.json()
    # With all files present the pipeline has been run
    assert data["status"] == "ok", (
        f"Status is degraded, missing: {data['missing_files']}"
    )


# ---------------------------------------------------------------------------
# /api/kpis
# ---------------------------------------------------------------------------

def test_kpis_200():
    r = client.get("/api/kpis")
    assert r.status_code == 200


def test_kpis_has_required_fields():
    r = client.get("/api/kpis")
    data = r.json()
    for key in ["peak_pm25_ugm3","peak_date","best_model_by_val",
                "data_date_range_start","data_date_range_end","total_rows"]:
        assert key in data, f"kpis missing key: {key}"


def test_kpis_best_model():
    r = client.get("/api/kpis")
    data = r.json()
    assert data["best_model_by_val"] == "RandomForest"


def test_kpis_peak_date():
    r = client.get("/api/kpis")
    data = r.json()
    # Known from outputs
    assert data["peak_date"] == "2025-01-03"
    assert data["peak_pm25_ugm3"] == pytest.approx(545.82, abs=0.1)


# ---------------------------------------------------------------------------
# /api/today
# ---------------------------------------------------------------------------

def test_today_200():
    r = client.get("/api/today")
    assert r.status_code == 200


def test_today_has_required_fields():
    r = client.get("/api/today")
    data = r.json()
    for key in ["date","pm25_ugm3","aqi_category","temperature","humidity",
                "wind_speed","rainfall"]:
        assert key in data

def test_today_pm25_positive():
    r = client.get("/api/today")
    assert r.json()["pm25_ugm3"] > 0


# ---------------------------------------------------------------------------
# /api/series
# ---------------------------------------------------------------------------

def test_series_full_range():
    r = client.get("/api/series")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] > 1800


def test_series_narrow_range():
    r = client.get("/api/series?start=2024-11-01&end=2024-11-07")
    assert r.status_code == 200
    data = r.json()
    assert 1 <= data["count"] <= 7


def test_series_each_point_has_category():
    r = client.get("/api/series?start=2024-11-01&end=2024-11-07")
    data = r.json()
    from src.config import AQI_CATEGORY_LABELS
    for pt in data["series"]:
        assert pt["aqi_category"] in AQI_CATEGORY_LABELS


def test_series_bad_date_400():
    r = client.get("/api/series?start=not-a-date")
    assert r.status_code == 400


def test_series_start_after_end_400():
    r = client.get("/api/series?start=2025-01-01&end=2024-01-01")
    assert r.status_code == 400


def test_series_future_start_4xx():
    # A start date beyond the last available date is a client error.
    # svc_series returns 404 when start > df.max, 400 when start > end after clip.
    # Both are correct 4xx responses.
    r = client.get("/api/series?start=2099-01-01")
    assert r.status_code in (400, 404)


# ---------------------------------------------------------------------------
# /api/forecast
# ---------------------------------------------------------------------------

def test_forecast_rf_200():
    r = client.get("/api/forecast?date=2024-11-01&model=random_forest")
    assert r.status_code == 200


def test_forecast_ridge_200():
    r = client.get("/api/forecast?date=2024-11-01&model=ridge")
    assert r.status_code == 200


def test_forecast_has_required_fields():
    r = client.get("/api/forecast?date=2024-11-01&model=random_forest")
    data = r.json()
    for key in ["query_date","forecast_date","model","pm25_forecast",
                "aqi_category","advisory","disclaimer",
                "confidence_low","confidence_high","confidence_method",
                "history_7d","weather_inputs"]:
        assert key in data, f"forecast missing key: {key}"


def test_forecast_dates_correct():
    r = client.get("/api/forecast?date=2024-11-01&model=random_forest")
    data = r.json()
    assert data["query_date"]    == "2024-11-01"
    assert data["forecast_date"] == "2024-11-02"


def test_forecast_confidence_range_valid():
    r = client.get("/api/forecast?date=2024-11-01&model=random_forest")
    data = r.json()
    assert data["confidence_low"] <= data["pm25_forecast"]
    assert data["pm25_forecast"]  <= data["confidence_high"]


def test_forecast_history_7d():
    r = client.get("/api/forecast?date=2024-11-01&model=random_forest")
    data = r.json()
    assert 1 <= len(data["history_7d"]) <= 7


def test_forecast_disclaimer_present():
    r = client.get("/api/forecast?date=2024-11-01&model=random_forest")
    data = r.json()
    assert "informational" in data["disclaimer"].lower()


def test_forecast_bad_date_400():
    r = client.get("/api/forecast?date=not-a-date&model=random_forest")
    assert r.status_code == 400


def test_forecast_future_date_404():
    r = client.get("/api/forecast?date=2099-01-01&model=random_forest")
    assert r.status_code == 404


def test_forecast_bad_model_400():
    r = client.get("/api/forecast?date=2024-11-01&model=xgboost")
    assert r.status_code == 400


def test_forecast_matches_console_menu():
    """
    API forecast for 2024-11-01 must match src/agent.py ForecastAgent
    for the same date.  Both use the same model file and
    _build_inference_features -- this test proves no logic was duplicated.
    """
    # API result
    r = client.get("/api/forecast?date=2024-11-01&model=random_forest")
    assert r.status_code == 200
    api_pred = r.json()["pm25_forecast"]

    # Console agent result
    import pandas as pd
    from src.agent import ForecastAgent
    from src.config import MERGED_CSV, AGENT_MODEL_FILE
    if not MERGED_CSV.exists() or not AGENT_MODEL_FILE.exists():
        pytest.skip("Required files missing")

    df     = pd.read_csv(MERGED_CSV, parse_dates=["date"], index_col="date").sort_index()
    agent  = ForecastAgent()
    result = agent.forecast_for_date(pd.Timestamp("2024-11-01"), df)
    console_pred = result["pm25_forecast"]

    assert api_pred == pytest.approx(console_pred, rel=1e-4), (
        f"API forecast {api_pred:.4f} != console forecast {console_pred:.4f}"
    )


# ---------------------------------------------------------------------------
# /api/eda
# ---------------------------------------------------------------------------

def test_eda_200():
    r = client.get("/api/eda")
    assert r.status_code == 200


def test_eda_has_peak_month():
    r = client.get("/api/eda")
    data = r.json()
    assert data["peak_month"] == 12   # December
    assert data["smog_season_mean_ugm3"] == pytest.approx(199.74, abs=0.1)


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------

def test_models_200():
    r = client.get("/api/models")
    assert r.status_code == 200


def test_models_comparison_has_five():
    r = client.get("/api/models")
    data = r.json()
    assert len(data["comparison"]) == 5


def test_models_has_confusion_matrix():
    r = client.get("/api/models")
    data = r.json()
    cm = data["confusion_matrix"]
    assert "matrix" in cm
    assert "labels" in cm
    import numpy as np
    mat = np.array(cm["matrix"])
    assert mat.shape[0] == mat.shape[1]


def test_models_predictions_test_has_dates():
    r = client.get("/api/models")
    data = r.json()
    pts = data["predictions_test"]
    assert len(pts) > 0
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", pts[0]["date"])


# ---------------------------------------------------------------------------
# /api/kalman
# ---------------------------------------------------------------------------

def test_kalman_200():
    r = client.get("/api/kalman")
    assert r.status_code == 200


def test_kalman_has_qr():
    r = client.get("/api/kalman")
    data = r.json()
    assert "Q" in data and "R" in data
    assert data["Q"] > 0 and data["R"] > 0


def test_kalman_series_length():
    r = client.get("/api/kalman")
    data = r.json()
    assert len(data["series"]) == 1882


def test_kalman_comparison_has_entries():
    r = client.get("/api/kalman")
    data = r.json()
    assert len(data["comparison"]) >= 2


# ---------------------------------------------------------------------------
# /api/about
# ---------------------------------------------------------------------------

def test_about_200():
    r = client.get("/api/about")
    assert r.status_code == 200


def test_about_has_peas():
    r = client.get("/api/about")
    data = r.json()
    assert "peas" in data
    peas = data["peas"]
    for key in ["performance_measure","environment","actuators","sensors"]:
        assert key in peas


def test_about_has_data_sources():
    r = client.get("/api/about")
    data = r.json()
    assert len(data["data_sources"]) >= 2
    for ds in data["data_sources"]:
        assert "name" in ds and "url" in ds and "attribution" in ds


def test_about_disclaimer():
    r = client.get("/api/about")
    data = r.json()
    assert "informational" in data["disclaimer"].lower()
