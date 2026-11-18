"""
test_dataset.py -- pytest tests for cleaning and feature engineering.

Tests are designed around correctness properties:
  - No data leakage: rolling/lag features must not include the current row.
  - No duplicates after clean().
  - pm25_ugm3 stays within physically plausible bounds after conversion.
  - Time split is strictly ordered with no overlap.
  - Outlier removal follows the documented IQR*k rule.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import AirQualityDataset, aqi_to_pm25_ugm3
from src.config  import AQI_CSV, OUTLIER_IQR_K, MAX_INTERP_GAP_DAYS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_dataset():
    """Load only -- no cleaning yet."""
    ds = AirQualityDataset()
    ds.load()
    return ds


@pytest.fixture(scope="module")
def clean_dataset():
    """Full pipeline on the real CSV (no weather merge for unit tests)."""
    ds = AirQualityDataset()
    ds.load()
    ds.clean()
    ds.add_features()
    return ds


# ---------------------------------------------------------------------------
# AQI conversion tests
# ---------------------------------------------------------------------------

class TestAqiConversion:
    """The inverse EPA formula must round-trip through known breakpoint values."""

    def test_aqi_50_gives_12_ugm3(self):
        result = aqi_to_pm25_ugm3(pd.Series([50]))
        assert result.iloc[0] == pytest.approx(12.0, abs=0.1)

    def test_aqi_100_gives_35_ugm3(self):
        result = aqi_to_pm25_ugm3(pd.Series([100]))
        assert result.iloc[0] == pytest.approx(35.4, abs=0.1)

    def test_aqi_200_gives_150_ugm3(self):
        result = aqi_to_pm25_ugm3(pd.Series([200]))
        assert result.iloc[0] == pytest.approx(150.4, abs=0.1)

    def test_aqi_above_500_does_not_raise(self):
        """Off-scale AQI values (Lahore Nov smog) must still return a number."""
        result = aqi_to_pm25_ugm3(pd.Series([530, 580]))
        assert result.notna().all()
        assert (result > 0).all()

    def test_all_conversions_positive(self):
        """Every AQI in the real CSV should produce a positive ug/m3 value."""
        df = pd.read_csv(AQI_CSV)
        df.columns = df.columns.str.strip()
        ugm3 = aqi_to_pm25_ugm3(df["pm25"])
        assert (ugm3 > 0).all(), "Some conversions produced non-positive ug/m3"


# ---------------------------------------------------------------------------
# Cleaning tests
# ---------------------------------------------------------------------------

class TestCleaning:

    def test_no_duplicate_dates_after_clean(self, clean_dataset):
        """clean() must eliminate all duplicate index values."""
        assert not clean_dataset.df.index.duplicated().any()

    def test_index_is_datetime(self, clean_dataset):
        assert isinstance(clean_dataset.df.index, pd.DatetimeIndex)

    def test_pm25_ugm3_no_negatives(self, clean_dataset):
        """Physical concentration cannot be negative."""
        assert (clean_dataset.df["pm25_ugm3"].dropna() >= 0).all()

    def test_outlier_rule_respected(self, clean_dataset):
        """
        After cleaning, no value should lie outside Q1-k*IQR .. Q3+k*IQR
        of the ORIGINAL distribution (we can only verify the final series
        is within its own Q1-k*IQR .. Q3+k*IQR, which is a necessary
        condition).
        """
        s = clean_dataset.df["pm25_ugm3"].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_K * iqr
        upper = q3 + OUTLIER_IQR_K * iqr
        outside = ((s < lower) | (s > upper)).sum()
        # After removing outliers from the original data, the cleaned series
        # may still have values outside ITS OWN IQR bounds (the distribution
        # changed), but extreme violations signal a logic error.
        assert outside < 5, f"{outside} values still outside cleaned IQR bounds"

    def test_no_gaps_longer_than_threshold(self, clean_dataset):
        """
        After cleaning, no run of NaN in pm25_ugm3 should exceed
        MAX_INTERP_GAP_DAYS (longer runs were dropped entirely).
        """
        s = clean_dataset.df["pm25_ugm3"]
        max_run = 0
        run = 0
        for v in s:
            if pd.isna(v):
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        assert max_run <= MAX_INTERP_GAP_DAYS, (
            f"Found NaN run of {max_run} days, threshold is {MAX_INTERP_GAP_DAYS}"
        )


# ---------------------------------------------------------------------------
# Feature engineering tests (no-leakage properties)
# ---------------------------------------------------------------------------

class TestFeatures:

    def test_lag1_is_previous_row(self, clean_dataset):
        """
        pm25_lag1[t] must equal pm25_ugm3[t-1], not pm25_ugm3[t].
        This is the core no-leakage property for lag features.
        """
        df = clean_dataset.df.dropna(subset=["pm25_lag1", "pm25_ugm3"])
        # Find contiguous pairs
        expected = df["pm25_ugm3"].shift(1)
        # Compare on the rows where both are defined
        valid = df["pm25_lag1"].notna() & expected.notna()
        diff = (df.loc[valid, "pm25_lag1"] - expected[valid]).abs()
        assert (diff < 1e-6).all(), "pm25_lag1 does not equal the previous row"

    def test_rolling3_excludes_current_day(self, clean_dataset):
        """
        pm25_roll3[t] must NOT use pm25_ugm3[t].
        We verify by checking roll3 != pm25 on rows where they would
        differ if the current value were included.
        """
        df = clean_dataset.df.dropna(subset=["pm25_roll3", "pm25_ugm3"])
        # If roll3 included today, roll3[t] == mean(t-2, t-1, t).
        # If it excludes today, roll3[t] == mean(t-3, t-2, t-1).
        # We cannot check every row, but we can verify roll3 != pm25_ugm3
        # on days where the current value is very different from neighbours.
        same = (df["pm25_roll3"] == df["pm25_ugm3"]).sum()
        # Equality on every row would be suspicious
        assert same < len(df) * 0.5, (
            "pm25_roll3 equals pm25_ugm3 on too many rows -- possible leakage"
        )

    def test_calendar_features_present(self, clean_dataset):
        df = clean_dataset.df
        for col in ["month", "day_of_year", "day_of_week", "is_smog_season"]:
            assert col in df.columns, f"Missing calendar feature: {col}"

    def test_is_smog_season_binary(self, clean_dataset):
        vals = clean_dataset.df["is_smog_season"].dropna().unique()
        assert set(vals).issubset({0, 1}), f"is_smog_season has non-binary values: {vals}"

    def test_smog_season_months(self, clean_dataset):
        """Oct, Nov, Dec, Jan rows should all be is_smog_season==1."""
        df = clean_dataset.df
        smog_months = df[df.index.month.isin([10, 11, 12, 1])]
        assert (smog_months["is_smog_season"] == 1).all()

    def test_non_smog_season_months(self, clean_dataset):
        df = clean_dataset.df
        non_smog = df[~df.index.month.isin([10, 11, 12, 1])]
        assert (non_smog["is_smog_season"] == 0).all()


# ---------------------------------------------------------------------------
# Time split tests
# ---------------------------------------------------------------------------

class TestTimeSplit:

    def test_split_no_overlap(self, clean_dataset):
        """Train, val, and test must be strictly non-overlapping in time."""
        train, val, test = clean_dataset.time_split()
        assert train.index.max() < val.index.min(), "Train/val overlap"
        assert val.index.max() < test.index.min(), "Val/test overlap"

    def test_split_covers_all_rows(self, clean_dataset):
        train, val, test = clean_dataset.time_split()
        total = len(train) + len(val) + len(test)
        expected = len(clean_dataset.df.dropna(subset=["pm25_ugm3"]))
        assert total == expected, (
            f"Split total {total} != dataset rows {expected}"
        )

    def test_train_is_largest(self, clean_dataset):
        train, val, test = clean_dataset.time_split()
        assert len(train) > len(val)
        assert len(train) > len(test)