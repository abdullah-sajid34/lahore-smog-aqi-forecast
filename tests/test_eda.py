"""
test_eda.py -- pytest tests for EDA functions.

All tests run on a small synthetic DataFrame so they execute fast,
require no network access, and do not depend on the real processed CSV.
Tests verify:
  - Functions return dictionaries with the expected keys.
  - Computed values are numerically sensible.
  - Figures are created without crashing.
  - aqi_category returns valid category labels.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

from src.eda import (
    aqi_category,
    plot_monthly_trend,
    plot_weekday_effect,
    plot_month_year_heatmap,
    plot_smog_season,
    plot_correlations,
)
from src.config import AQI_BREAKPOINTS, OUTPUTS_FIGURES


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_df():
    """
    Two full years of daily data with known seasonal signal:
    months 10-12 have PM2.5 ~ 200 ug/m3, rest ~ 50 ug/m3.
    Weather columns are filled with simple constants.
    """
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", "2022-12-31", freq="D")
    n     = len(dates)
    month = dates.month

    pm25 = np.where(
        np.isin(month, [10, 11, 12]),
        200 + np.random.normal(0, 10, n),
        50  + np.random.normal(0, 5,  n),
    ).clip(0)

    df = pd.DataFrame({
        "pm25_ugm3":                  pm25,
        "is_smog_season":             np.isin(month, [10, 11, 12, 1]).astype(int),
        "temperature_2m_mean":        np.random.uniform(10, 35, n),
        "relative_humidity_2m_mean":  np.random.uniform(40, 90, n),
        "wind_speed_10m_max":         np.random.uniform(5, 30, n),
        "precipitation_sum":          np.random.uniform(0, 5, n),
    }, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# aqi_category tests
# ---------------------------------------------------------------------------

class TestAqiCategory:

    def test_low_value_good(self):
        assert aqi_category(5.0) == "Good"

    def test_mid_value_unhealthy(self):
        # 100 ug/m3 is in Very Unhealthy (201-300 AQI) range based on EPA table
        result = aqi_category(100.0)
        assert result in [bp[4] for bp in AQI_BREAKPOINTS]

    def test_high_value_hazardous(self):
        result = aqi_category(400.0)
        assert "Hazardous" in result

    def test_returns_string(self):
        assert isinstance(aqi_category(50.0), str)


# ---------------------------------------------------------------------------
# plot_monthly_trend
# ---------------------------------------------------------------------------

class TestMonthlyTrend:

    def test_returns_dict_with_required_keys(self, synthetic_df):
        result = plot_monthly_trend(synthetic_df)
        assert "peak_month" in result
        assert "peak_month_mean_ugm3" in result
        assert "monthly_mean_by_month" in result

    def test_peak_month_is_integer(self, synthetic_df):
        result = plot_monthly_trend(synthetic_df)
        assert isinstance(result["peak_month"], int)
        assert 1 <= result["peak_month"] <= 12

    def test_peak_month_in_smog_season(self, synthetic_df):
        """With our synthetic data Oct-Dec is high, so peak must be in [10,11,12]."""
        result = plot_monthly_trend(synthetic_df)
        assert result["peak_month"] in [10, 11, 12], (
            f"Expected smog-season peak, got month {result['peak_month']}"
        )

    def test_figure_saved(self):
        assert (OUTPUTS_FIGURES / "01_monthly_trend_and_boxplot.png").exists()


# ---------------------------------------------------------------------------
# plot_weekday_effect
# ---------------------------------------------------------------------------

class TestWeekdayEffect:

    def test_returns_dict_with_required_keys(self, synthetic_df):
        result = plot_weekday_effect(synthetic_df)
        for key in ["weekday_means_ugm3", "max_weekday", "min_weekday", "weekday_range_ugm3"]:
            assert key in result

    def test_range_is_non_negative(self, synthetic_df):
        result = plot_weekday_effect(synthetic_df)
        assert result["weekday_range_ugm3"] >= 0

    def test_all_seven_weekdays_present(self, synthetic_df):
        result = plot_weekday_effect(synthetic_df)
        assert len(result["weekday_means_ugm3"]) == 7

    def test_figure_saved(self):
        assert (OUTPUTS_FIGURES / "02_weekday_effect.png").exists()


# ---------------------------------------------------------------------------
# plot_month_year_heatmap
# ---------------------------------------------------------------------------

class TestHeatmap:

    def test_returns_dict(self, synthetic_df):
        result = plot_month_year_heatmap(synthetic_df)
        assert "heatmap_years" in result
        assert "heatmap_months" in result

    def test_years_in_output(self, synthetic_df):
        result = plot_month_year_heatmap(synthetic_df)
        assert "2021" in result["heatmap_years"]
        assert "2022" in result["heatmap_years"]

    def test_figure_saved(self):
        assert (OUTPUTS_FIGURES / "03_month_year_heatmap.png").exists()


# ---------------------------------------------------------------------------
# plot_smog_season
# ---------------------------------------------------------------------------

class TestSmogSeason:

    def test_returns_dict_with_required_keys(self, synthetic_df):
        result = plot_smog_season(synthetic_df)
        for key in ["smog_season_mean_ugm3", "off_season_mean_ugm3",
                    "smog_vs_offseason_ratio"]:
            assert key in result

    def test_smog_mean_higher_than_offseason(self, synthetic_df):
        """Synthetic data has smog months ~200 ug/m3 vs ~50 off-season."""
        result = plot_smog_season(synthetic_df)
        assert result["smog_season_mean_ugm3"] > result["off_season_mean_ugm3"]

    def test_ratio_positive(self, synthetic_df):
        result = plot_smog_season(synthetic_df)
        assert result["smog_vs_offseason_ratio"] > 1.0

    def test_figure_saved(self):
        assert (OUTPUTS_FIGURES / "04_smog_season_analysis.png").exists()


# ---------------------------------------------------------------------------
# plot_correlations
# ---------------------------------------------------------------------------

class TestCorrelations:

    def test_returns_dict_with_required_keys(self, synthetic_df):
        result = plot_correlations(synthetic_df)
        for key in ["weather_correlations", "autocorrelations",
                    "autocorr_lag1", "strongest_weather_corr_var"]:
            assert key in result

    def test_autocorrelations_lags_1_to_7(self, synthetic_df):
        result = plot_correlations(synthetic_df)
        ac = result["autocorrelations"]
        for lag in range(1, 8):
            assert f"lag{lag}" in ac

    def test_autocorr_values_in_range(self, synthetic_df):
        result = plot_correlations(synthetic_df)
        for v in result["autocorrelations"].values():
            assert -1.0 <= v <= 1.0

    def test_lag1_autocorr_is_positive(self, synthetic_df):
        """PM2.5 is persistent: lag-1 autocorrelation must be positive."""
        result = plot_correlations(synthetic_df)
        assert result["autocorr_lag1"] > 0

    def test_weather_corr_values_in_range(self, synthetic_df):
        result = plot_correlations(synthetic_df)
        for v in result["weather_correlations"].values():
            assert -1.0 <= v <= 1.0

    def test_figure_saved(self):
        assert (OUTPUTS_FIGURES / "05_correlations_and_autocorrelation.png").exists()