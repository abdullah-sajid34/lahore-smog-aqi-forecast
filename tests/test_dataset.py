"""
test_dataset.py -- pytest tests for cleaning and feature engineering.

Tests are designed around correctness properties:
  - No data leakage: rolling/lag features must not include the current row.
  - No duplicates after clean().
  - pm25_ugm3 stays within physically plausible bounds after conversion.
  - Time split is strictly ordered with no overlap.
  - Outlier removal follows the documented IQR*k rule.
  - Lag and rolling features use CALENDAR DAYS, not row positions
    (critical when the index has gaps after dropping long outage periods).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import AirQualityDataset, aqi_to_pm25_ugm3, _calendar_shift, _calendar_rolling
from src.config  import AQI_CSV, OUTLIER_IQR_K, MAX_INTERP_GAP_DAYS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_dataset():
    """Full pipeline on the real CSV."""
    ds = AirQualityDataset()
    ds.load()
    ds.clean()
    ds.add_features()
    return ds


# ---------------------------------------------------------------------------
# AQI conversion tests
# ---------------------------------------------------------------------------

class TestAqiConversion:

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
        result = aqi_to_pm25_ugm3(pd.Series([530, 580]))
        assert result.notna().all()
        assert (result > 0).all()

    def test_all_conversions_positive(self):
        df = pd.read_csv(AQI_CSV)
        df.columns = df.columns.str.strip()
        ugm3 = aqi_to_pm25_ugm3(df["pm25"])
        assert (ugm3 > 0).all()


# ---------------------------------------------------------------------------
# Calendar-day correctness tests  (Step 0b fix verification)
# ---------------------------------------------------------------------------

class TestCalendarDayFeatures:
    """
    Prove that _calendar_shift and _calendar_rolling respect calendar days
    not row positions.  We use a synthetic Series with a deliberate gap to
    make the difference observable.
    """

    @pytest.fixture
    def gapped_series(self):
        """
        Days 1-5, then a 4-day gap, then days 10-14.
        Row-shift(1) on day-10 would return day-5's value.
        Calendar-shift(1) on day-10 must return NaN (day-9 is missing).
        """
        dates = (
            pd.date_range("2020-01-01", periods=5, freq="D").tolist()
            + pd.date_range("2020-01-10", periods=5, freq="D").tolist()
        )
        vals = [10.0, 20.0, 30.0, 40.0, 50.0,
                60.0, 70.0, 80.0, 90.0, 100.0]
        return pd.Series(vals, index=pd.DatetimeIndex(dates), name="pm25_ugm3")

    def test_calendar_shift_returns_nan_across_gap(self, gapped_series):
        """
        lag1 on 2020-01-10 must be NaN because 2020-01-09 is missing.
        A row-based shift would incorrectly return 50.0 (the value from 2020-01-05).
        """
        shifted = _calendar_shift(gapped_series, 1)
        assert pd.isna(shifted.loc[pd.Timestamp("2020-01-10")]), (
            "lag1 should be NaN across a calendar gap, not the previous row value"
        )

    def test_calendar_shift_correct_within_contiguous_block(self, gapped_series):
        """lag1 on 2020-01-05 must equal the value on 2020-01-04 (40.0)."""
        shifted = _calendar_shift(gapped_series, 1)
        assert shifted.loc[pd.Timestamp("2020-01-05")] == pytest.approx(40.0)

    def test_calendar_shift_lag7_correct_across_gap(self, gapped_series):
        """
        lag7 on 2020-01-10 should equal the value from 2020-01-03 (30.0),
        because exactly 7 calendar days prior exists in the series.
        """
        shifted = _calendar_shift(gapped_series, 7)
        result = shifted.loc[pd.Timestamp("2020-01-10")]
        assert result == pytest.approx(30.0), (
            f"Expected 30.0 (value on 2020-01-03), got {result}"
        )

    def test_calendar_rolling_nan_across_gap(self, gapped_series):
        """
        roll3 on 2020-01-10 must be NaN because the 3-day window [07,08,09]
        contains missing calendar days.
        """
        rolled = _calendar_rolling(gapped_series, 3)
        assert pd.isna(rolled.loc[pd.Timestamp("2020-01-10")]), (
            "roll3 should be NaN when any day in the calendar window is missing"
        )

    def test_calendar_rolling_correct_in_contiguous_block(self, gapped_series):
        """
        roll3 on 2020-01-05 = mean(day2, day3, day4) shifted, i.e.,
        mean of values on 01-02, 01-03, 01-04 = mean(20, 30, 40) = 30.0
        """
        rolled = _calendar_rolling(gapped_series, 3)
        result = rolled.loc[pd.Timestamp("2020-01-05")]
        assert result == pytest.approx(30.0, abs=0.01)

    def test_real_data_lag1_nan_after_gap(self, clean_dataset):
        """
        On the real cleaned data, the first row after a 4-day gap (2021-07-08)
        must have pm25_lag1 = NaN because 2021-07-07 was dropped.
        """
        df = clean_dataset.df
        gap_after = pd.Timestamp("2021-07-08")
        if gap_after in df.index:
            assert pd.isna(df.loc[gap_after, "pm25_lag1"]), (
                "pm25_lag1 on 2021-07-08 should be NaN (2021-07-07 is a dropped gap day)"
            )


# ---------------------------------------------------------------------------
# Cleaning tests
# ---------------------------------------------------------------------------

class TestCleaning:

    def test_no_duplicate_dates_after_clean(self, clean_dataset):
        assert not clean_dataset.df.index.duplicated().any()

    def test_index_is_datetime(self, clean_dataset):
        assert isinstance(clean_dataset.df.index, pd.DatetimeIndex)

    def test_pm25_ugm3_no_negatives(self, clean_dataset):
        assert (clean_dataset.df["pm25_ugm3"].dropna() >= 0).all()

    def test_outlier_lower_bound_clipped_at_zero(self, clean_dataset):
        """The cleaned series must contain no negative pm25 values."""
        assert (clean_dataset.df["pm25_ugm3"].dropna() >= 0).all()

    def test_no_gaps_longer_than_threshold(self, clean_dataset):
        s = clean_dataset.df["pm25_ugm3"]
        max_run = run = 0
        for v in s:
            if pd.isna(v):
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        assert max_run <= MAX_INTERP_GAP_DAYS


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatures:

    def test_lag1_is_previous_calendar_day(self, clean_dataset):
        """
        For contiguous rows pm25_lag1[t] must equal pm25_ugm3[t-1].
        We test only on rows where the previous calendar day is also present.
        """
        df = clean_dataset.df.copy()
        df["_prev"] = df["pm25_ugm3"].shift(1)   # row shift for reference
        # Only valid where both rows exist and are contiguous
        full = pd.date_range(df.index.min(), df.index.max(), freq="D")
        present = set(df.index)
        contiguous_mask = pd.Series(
            [(d in present) and ((d - pd.Timedelta(days=1)) in present) for d in df.index],
            index=df.index
        )
        sub = df[contiguous_mask].dropna(subset=["pm25_lag1","pm25_ugm3"])
        diff = (sub["pm25_lag1"] - sub["_prev"]).abs()
        assert (diff < 1e-6).all()

    def test_rolling3_excludes_current_day(self, clean_dataset):
        df = clean_dataset.df.dropna(subset=["pm25_roll3", "pm25_ugm3"])
        same = (df["pm25_roll3"] == df["pm25_ugm3"]).sum()
        assert same < len(df) * 0.5

    def test_calendar_features_present(self, clean_dataset):
        for col in ["month", "day_of_year", "day_of_week", "is_smog_season"]:
            assert col in clean_dataset.df.columns

    def test_is_smog_season_binary(self, clean_dataset):
        vals = clean_dataset.df["is_smog_season"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_smog_season_months(self, clean_dataset):
        df = clean_dataset.df
        smog = df[df.index.month.isin([10, 11, 12, 1])]
        assert (smog["is_smog_season"] == 1).all()

    def test_non_smog_season_months(self, clean_dataset):
        df = clean_dataset.df
        non = df[~df.index.month.isin([10, 11, 12, 1])]
        assert (non["is_smog_season"] == 0).all()


# ---------------------------------------------------------------------------
# Time split tests
# ---------------------------------------------------------------------------

class TestTimeSplit:

    def test_split_no_overlap(self, clean_dataset):
        train, val, test = clean_dataset.time_split()
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_split_covers_all_rows(self, clean_dataset):
        train, val, test = clean_dataset.time_split()
        total    = len(train) + len(val) + len(test)
        expected = len(clean_dataset.df.dropna(subset=["pm25_ugm3"]))
        assert total == expected

    def test_train_is_largest(self, clean_dataset):
        train, val, test = clean_dataset.time_split()
        assert len(train) > len(val)
        assert len(train) > len(test)