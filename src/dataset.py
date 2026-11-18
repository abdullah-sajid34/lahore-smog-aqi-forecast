"""
dataset.py -- AirQualityDataset: load, clean, feature-engineer, split.

The class is the single source of truth for data preparation.  All other
modules (EDA, training, evaluation) call these methods instead of repeating
data-wrangling logic.

Data provenance
---------------
pm25 column in the raw CSV is US AQI (reported by AQICN / US Embassy Lahore).
Values such as 530 or 580 are only possible on the AQI scale -- the US EPA
AQI ceiling for PM2.5 is 500, but AQICN continues the same piecewise-linear
scale above 500 for extreme events.

WHY convert AQI -> ug/m3:
  ML models should predict a physical quantity (concentration), not an
  index whose breakpoints introduce discontinuities.  The inverse US EPA
  piecewise-linear formula is the correct inverse of the official AQI
  formula and is fully documented in config.AQI_BREAKPOINTS.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    AQI_BREAKPOINTS,
    AQI_CSV,
    MAX_INTERP_GAP_DAYS,
    MERGED_CSV,
    OUTLIER_IQR_K,
    TRAIN_FRAC,
    VAL_FRAC,
    WEATHER_CSV,
)
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Standalone helper: inverse US EPA AQI formula
# ---------------------------------------------------------------------------

def aqi_to_pm25_ugm3(aqi_series: pd.Series) -> pd.Series:
    """
    Convert a Series of US AQI values to PM2.5 concentrations (ug/m3).

    Formula (US EPA, Appendix A of AQI Technical Assistance Document):
        Cp = (Cp_hi - Cp_lo) / (I_hi - I_lo) * (I_p - I_lo) + Cp_lo

    For AQI > 500 (off-scale but observed in Lahore November smog), the last
    segment is extended linearly.
    """
    result = pd.Series(index=aqi_series.index, dtype=float, name="pm25_ugm3")

    for I_lo, I_hi, Cp_lo, Cp_hi, *_ in AQI_BREAKPOINTS:
        mask = (aqi_series >= I_lo) & (aqi_series <= I_hi)
        result[mask] = (
            Cp_lo
            + (aqi_series[mask] - I_lo) / (I_hi - I_lo) * (Cp_hi - Cp_lo)
        )

    # Off-scale (AQI > 500): extend last segment linearly
    last = AQI_BREAKPOINTS[-1]
    I_lo, I_hi, Cp_lo, Cp_hi = last[0], last[1], last[2], last[3]
    off_scale = aqi_series > 500
    if off_scale.any():
        result[off_scale] = (
            Cp_lo
            + (aqi_series[off_scale] - I_lo) / (I_hi - I_lo) * (Cp_hi - Cp_lo)
        )

    return result.round(2)


def _calendar_shift(series: pd.Series, days: int) -> pd.Series:
    """
    Shift a daily Series by `days` CALENDAR days, not row positions.

    WHY this matters: after clean() drops long-gap rows the DatetimeIndex is
    non-contiguous.  pandas .shift(n) moves by row number, so across a gap
    shift(1) on 2021-07-08 returns the value from 2021-07-03 instead of NaN
    (2021-07-07 is missing).  That silently smuggles a stale value into the
    lag feature -- a form of data leakage.

    The fix: reindex to a full daily calendar (introducing NaN for missing
    dates), shift by row (now equivalent to calendar shift), then reindex
    back to the original sparse index.  Missing calendar neighbours
    propagate as NaN, which correctly signals "this lag is unavailable".
    """
    full_idx  = pd.date_range(series.index.min(), series.index.max(), freq="D")
    s_full    = series.reindex(full_idx)          # sparse -> dense, NaN for gaps
    s_shifted = s_full.shift(days)                # row shift == calendar shift on dense index
    return s_shifted.reindex(series.index)        # dense -> back to sparse


def _calendar_rolling(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling mean over a `window`-day CALENDAR window, excluding current day.

    Same gap problem as _calendar_shift: .rolling() on a sparse index counts
    rows, not days, so a 7-row window after a 10-day gap covers 17 calendar
    days, mixing old and recent values.

    Fix: densify, shift(1) to exclude current day, roll on the dense series,
    then reindex back.  Any window that spans a gap will have >= 1 NaN input
    and will correctly return NaN for the whole window (min_periods=window).
    """
    full_idx  = pd.date_range(series.index.min(), series.index.max(), freq="D")
    s_full    = series.reindex(full_idx)
    rolled    = s_full.shift(1).rolling(window, min_periods=window).mean()
    return rolled.reindex(series.index)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AirQualityDataset:
    """
    Encapsulates loading, cleaning, feature engineering, and splitting
    of the Lahore air-quality dataset.

    Typical usage:
        ds = AirQualityDataset()
        ds.load()
        ds.clean()
        ds.add_features()
        train, val, test = ds.time_split()
    """

    def __init__(self, aqi_path: Path = AQI_CSV, weather_path: Path = WEATHER_CSV):
        set_seed()
        self.aqi_path     = aqi_path
        self.weather_path = weather_path
        self.df: pd.DataFrame | None = None
        self._cleaning_log: list[str] = []

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------

    def load(self) -> "AirQualityDataset":
        """
        Read the raw AQI CSV, strip whitespace from column names, parse dates,
        convert AQI -> ug/m3, and merge with weather data.
        """
        aqi = pd.read_csv(self.aqi_path)
        aqi.columns = aqi.columns.str.strip()
        aqi["date"]  = pd.to_datetime(aqi["date"])
        aqi = aqi.set_index("date").sort_index()
        aqi = aqi.rename(columns={"pm25": "pm25_aqi"})
        aqi["pm25_ugm3"] = aqi_to_pm25_ugm3(aqi["pm25_aqi"])

        self._cleaning_log.append(
            f"Loaded AQI CSV: {len(aqi)} rows, "
            f"{aqi.index.min().date()} to {aqi.index.max().date()}"
        )

        if self.weather_path.exists():
            weather = pd.read_csv(self.weather_path, parse_dates=["date"], index_col="date")
            weather.index = pd.to_datetime(weather.index)
            self.df = aqi.join(weather, how="inner")
            self._cleaning_log.append(f"After merge with weather: {len(self.df)} rows")
        else:
            self.df = aqi.copy()
            self._cleaning_log.append("Weather file not found -- AQI data only.")

        return self

    # ------------------------------------------------------------------
    # 2. CLEAN
    # ------------------------------------------------------------------

    def clean(self) -> "AirQualityDataset":
        """
        1. Remove duplicate dates.
        2. Remove statistical outliers (IQR * k, lower bound clipped at 0).
        3. Reindex to full daily calendar; interpolate gaps <= MAX_INTERP_GAP_DAYS;
           drop rows in longer gaps.
        """
        df = self.df.copy()
        n0 = len(df)

        # (a) Duplicates
        dupes = df.index.duplicated(keep="first")
        if dupes.sum():
            df = df[~dupes]
            self._cleaning_log.append(f"Removed {dupes.sum()} duplicate dates.")

        # (b) Outlier removal -- IQR * k on pm25_ugm3
        # WHY IQR with k=3: conservative, only removes extreme statistical
        # outliers, not genuine high-smog days.
        # WHY clip lower at 0: PM2.5 concentration cannot be negative.
        # The raw lower bound (-342 ug/m3) is a mathematical artefact of the
        # symmetric IQR formula applied to a right-skewed distribution.
        q1  = df["pm25_ugm3"].quantile(0.25)
        q3  = df["pm25_ugm3"].quantile(0.75)
        iqr = q3 - q1
        lower = max(0.0, q1 - OUTLIER_IQR_K * iqr)   # clipped at 0
        upper = q3 + OUTLIER_IQR_K * iqr
        outlier_mask = (df["pm25_ugm3"] < lower) | (df["pm25_ugm3"] > upper)
        n_outliers = outlier_mask.sum()
        if n_outliers:
            df = df[~outlier_mask]
            self._cleaning_log.append(
                f"Removed {n_outliers} outlier rows "
                f"(pm25_ugm3 outside [{lower:.1f}, {upper:.1f}] ug/m3, "
                f"IQR*{OUTLIER_IQR_K}, lower clipped at 0)."
            )

        # (c) Gap handling
        full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
        df = df.reindex(full_index)
        missing_before = df["pm25_ugm3"].isna()
        gap_lengths    = self._run_lengths(missing_before)
        too_long       = missing_before & (gap_lengths > MAX_INTERP_GAP_DAYS)
        n_drop         = too_long.sum()

        df["pm25_ugm3"] = df["pm25_ugm3"].interpolate(
            method="time", limit=MAX_INTERP_GAP_DAYS
        )
        if "pm25_aqi" in df.columns:
            df["pm25_aqi"] = df["pm25_aqi"].interpolate(
                method="time", limit=MAX_INTERP_GAP_DAYS
            )

        df = df[~too_long]
        df.index.name = "date"

        self._cleaning_log.append(
            f"Gap interpolation: {missing_before.sum()} missing days; "
            f"interpolated <= {MAX_INTERP_GAP_DAYS} days; dropped {n_drop} longer-gap rows."
        )

        self.df = df

        full_range_all = pd.date_range(df.index.min(), df.index.max(), freq="D")
        n_missing = len(full_range_all) - len(df)
        print("=" * 60)
        print("DATASET SUMMARY AFTER CLEANING")
        print("=" * 60)
        print(f"  Rows before cleaning  : {n0}")
        print(f"  Rows after  cleaning  : {len(df)}")
        print(f"  Date range            : {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Long-gap rows dropped : {n_drop}")
        print(f"  Remaining NaN (pm25)  : {df['pm25_ugm3'].isna().sum()}")
        print()
        for msg in self._cleaning_log:
            print(f"  LOG | {msg}")
        print("=" * 60)

        return self

    @staticmethod
    def _run_lengths(mask: pd.Series) -> pd.Series:
        """Return the run-length of the consecutive-True run each True element belongs to."""
        lengths     = pd.Series(0, index=mask.index, dtype=int)
        run         = 0
        run_indices = []
        for idx, val in mask.items():
            if val:
                run += 1
                run_indices.append(idx)
            else:
                for i in run_indices:
                    lengths[i] = run
                run         = 0
                run_indices = []
        for i in run_indices:
            lengths[i] = run
        return lengths

    # ------------------------------------------------------------------
    # 3. ADD FEATURES  (calendar-day safe)
    # ------------------------------------------------------------------

    def add_features(self) -> "AirQualityDataset":
        """
        Engineer lag, rolling, and calendar features.

        CALENDAR-DAY CORRECTNESS:
          After clean() drops long-gap rows the index is non-contiguous.
          Using pandas .shift(n) or .rolling(w) directly on a sparse index
          operates on ROW POSITIONS, not calendar days.  Across a gap this
          silently assigns a stale value to a lag feature.

          We use _calendar_shift() and _calendar_rolling() which densify
          the index to a full daily calendar, perform the operation, then
          reindex back.  Any lag or window that spans a missing calendar day
          correctly returns NaN.

        NO LEAKAGE:
          - pm25_lag{n}  : value at t-n calendar days (never t).
          - pm25_roll{w} : mean of [t-w-1 .. t-1] calendar days.
            _calendar_rolling already calls shift(1) before rolling, so
            the current day is always excluded.
        """
        df = self.df.copy()
        pm = df["pm25_ugm3"]

        # Lag features
        for lag in [1, 2, 3, 7]:
            df[f"pm25_lag{lag}"] = _calendar_shift(pm, lag)

        # Rolling means (calendar-aware, current day excluded)
        df["pm25_roll3"] = _calendar_rolling(pm, 3)
        df["pm25_roll7"] = _calendar_rolling(pm, 7)

        # Calendar features
        df["month"]          = df.index.month
        df["day_of_year"]    = df.index.day_of_year
        df["day_of_week"]    = df.index.day_of_week
        df["is_smog_season"] = df.index.month.isin([10, 11, 12, 1]).astype(int)

        self.df = df
        return self

    # ------------------------------------------------------------------
    # 4. TIME SPLIT
    # ------------------------------------------------------------------

    def time_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split into train / val / test by time order (never shuffled).

        WHY time-based: random splits leak future lag/rolling values into
        training data.  We always train on the past, evaluate on the future.
        """
        df = self.df.dropna(subset=["pm25_ugm3"])
        n  = len(df)

        n_train = int(n * TRAIN_FRAC)
        n_val   = int(n * VAL_FRAC)

        train = df.iloc[:n_train]
        val   = df.iloc[n_train : n_train + n_val]
        test  = df.iloc[n_train + n_val :]

        print("\nTIME SPLIT")
        print(f"  Train : {len(train):4d} rows  "
              f"({train.index.min().date()} to {train.index.max().date()})")
        print(f"  Val   : {len(val):4d} rows  "
              f"({val.index.min().date()} to {val.index.max().date()})")
        print(f"  Test  : {len(test):4d} rows  "
              f"({test.index.min().date()} to {test.index.max().date()})")

        return train, val, test

    # ------------------------------------------------------------------
    # 5. SAVE
    # ------------------------------------------------------------------

    def save(self, path: Path = MERGED_CSV) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path)
        print(f"\nSaved merged dataset to {path}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = AirQualityDataset()
    ds.load()
    ds.clean()
    ds.add_features()
    ds.time_split()
    ds.save()