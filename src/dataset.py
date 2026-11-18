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

# Allow running as a script from the project root
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

    where I_p is the AQI value and [I_lo, I_hi], [Cp_lo, Cp_hi] are the
    matching breakpoint pair from AQI_BREAKPOINTS.

    For AQI > 500 (off-scale but observed in Lahore November smog), the last
    segment is extended linearly.  This is the best available approximation
    because the EPA formula was not defined above 500; we document this in
    the returned Series name.
    """
    result = pd.Series(index=aqi_series.index, dtype=float, name="pm25_ugm3")

    for I_lo, I_hi, Cp_lo, Cp_hi, *_ in AQI_BREAKPOINTS:
        mask = (aqi_series >= I_lo) & (aqi_series <= I_hi)
        result[mask] = (
            Cp_lo
            + (aqi_series[mask] - I_lo) / (I_hi - I_lo) * (Cp_hi - Cp_lo)
        )

    # Off-scale (AQI > 500): extend last segment
    last = AQI_BREAKPOINTS[-1]
    I_lo, I_hi, Cp_lo, Cp_hi = last[0], last[1], last[2], last[3]
    off_scale = aqi_series > 500
    if off_scale.any():
        result[off_scale] = (
            Cp_lo
            + (aqi_series[off_scale] - I_lo) / (I_hi - I_lo) * (Cp_hi - Cp_lo)
        )

    return result.round(2)


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
        self.df: pd.DataFrame | None = None   # populated by load()
        self._cleaning_log: list[str] = []    # human-readable audit trail

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------

    def load(self) -> "AirQualityDataset":
        """
        Read the raw AQI CSV, strip whitespace from column names, parse dates,
        convert AQI -> ug/m3, and merge with weather data.

        WHY strip whitespace: the raw CSV has a leading space in ' pm25'.
        WHY parse dates here: downstream methods rely on a DatetimeIndex.
        """
        # --- AQI data ---
        aqi = pd.read_csv(self.aqi_path)
        aqi.columns = aqi.columns.str.strip()          # ' pm25' -> 'pm25'
        aqi["date"]  = pd.to_datetime(aqi["date"])     # 'YYYY/M/D' -> Timestamp
        aqi = aqi.set_index("date").sort_index()

        # Rename raw column to make its units explicit
        aqi = aqi.rename(columns={"pm25": "pm25_aqi"})

        # Convert AQI -> physical concentration
        aqi["pm25_ugm3"] = aqi_to_pm25_ugm3(aqi["pm25_aqi"])

        self._cleaning_log.append(
            f"Loaded AQI CSV: {len(aqi)} rows, "
            f"{aqi.index.min().date()} to {aqi.index.max().date()}"
        )

        # --- Weather data ---
        if self.weather_path.exists():
            weather = pd.read_csv(self.weather_path, parse_dates=["date"], index_col="date")
            weather.index = pd.to_datetime(weather.index)

            # Merge on date index; inner join keeps only dates present in both.
            # WHY inner: if weather is missing for an AQI date, we cannot
            # compute weather features for that row anyway.
            self.df = aqi.join(weather, how="inner")
            self._cleaning_log.append(
                f"After merge with weather: {len(self.df)} rows"
            )
        else:
            # Weather file not yet downloaded; proceed with AQI only
            self.df = aqi.copy()
            self._cleaning_log.append(
                "Weather file not found -- proceeding with AQI data only."
            )

        return self

    # ------------------------------------------------------------------
    # 2. CLEAN
    # ------------------------------------------------------------------

    def clean(self) -> "AirQualityDataset":
        """
        1. Remove duplicate dates (keep first occurrence).
        2. Flag and remove statistical outliers in pm25_ugm3.
        3. Reindex to a full daily calendar; interpolate gaps <= 3 days;
           drop rows where the gap exceeds MAX_INTERP_GAP_DAYS.
        4. Report row count, date range, and missing-day count.
        """
        df = self.df.copy()
        n0 = len(df)

        # -- (a) Duplicate dates --
        # WHY: duplicates would create multiple rows for the same date,
        # breaking time-series operations and inflating some lag values.
        dupes = df.index.duplicated(keep="first")
        if dupes.sum():
            df = df[~dupes]
            self._cleaning_log.append(f"Removed {dupes.sum()} duplicate dates.")

        # -- (b) Outlier removal (IQR method on pm25_ugm3) --
        # WHY IQR with k=3: the distribution is right-skewed; a fixed cap
        # (e.g., >500 ug/m3) would miss extreme anomalies in summer.
        # k=3 is conservative: only flags values that are extremely far
        # from the interquartile range, minimising false positives on
        # genuine high-smog days.
        q1 = df["pm25_ugm3"].quantile(0.25)
        q3 = df["pm25_ugm3"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_K * iqr
        upper = q3 + OUTLIER_IQR_K * iqr
        outlier_mask = (df["pm25_ugm3"] < lower) | (df["pm25_ugm3"] > upper)
        n_outliers = outlier_mask.sum()
        if n_outliers:
            df = df[~outlier_mask]
            self._cleaning_log.append(
                f"Removed {n_outliers} outlier rows "
                f"(pm25_ugm3 outside [{lower:.1f}, {upper:.1f}] ug/m3, "
                f"IQR*{OUTLIER_IQR_K})."
            )

        # -- (c) Gap handling --
        # Reindex to a contiguous daily calendar so gaps become explicit NaTs.
        full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
        df = df.reindex(full_index)

        # Mark which rows are genuinely missing (not just interpolation targets)
        missing_before = df["pm25_ugm3"].isna()

        # Identify run lengths of consecutive NaNs
        # WHY max 3 days: gaps of 1-3 days are plausibly sensor downtime and
        # can be recovered by linear interpolation without introducing much
        # error. Longer gaps likely reflect data-collection outages and
        # should be excluded rather than synthesised.
        gap_lengths = self._run_lengths(missing_before)

        # Mark rows in gaps longer than the threshold for dropping
        too_long = missing_before & (gap_lengths > MAX_INTERP_GAP_DAYS)
        n_drop = too_long.sum()

        # Interpolate the short gaps
        df["pm25_ugm3"] = df["pm25_ugm3"].interpolate(method="time", limit=MAX_INTERP_GAP_DAYS)
        # Also interpolate pm25_aqi for reference
        if "pm25_aqi" in df.columns:
            df["pm25_aqi"] = df["pm25_aqi"].interpolate(method="time", limit=MAX_INTERP_GAP_DAYS)

        # Drop the rows that belonged to long gaps
        df = df[~too_long]
        df.index.name = "date"

        self._cleaning_log.append(
            f"Gap interpolation: {missing_before.sum()} missing days in full range; "
            f"interpolated up to {MAX_INTERP_GAP_DAYS} consecutive days; "
            f"dropped {n_drop} rows in longer gaps."
        )

        self.df = df

        # -- Summary report --
        full_range_all = pd.date_range(df.index.min(), df.index.max(), freq="D")
        n_missing = len(full_range_all) - len(df)
        print("=" * 60)
        print("DATASET SUMMARY AFTER CLEANING")
        print("=" * 60)
        print(f"  Rows before cleaning : {n0}")
        print(f"  Rows after  cleaning : {len(df)}")
        print(f"  Date range           : {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Missing days (dropped): {n_drop}")
        print(f"  Remaining NaN (pm25) : {df['pm25_ugm3'].isna().sum()}")
        print()
        for msg in self._cleaning_log:
            print(f"  LOG | {msg}")
        print("=" * 60)

        return self

    @staticmethod
    def _run_lengths(mask: pd.Series) -> pd.Series:
        """
        For each True position in a boolean Series, return the length of the
        consecutive True run it belongs to.  False positions get 0.

        Used to distinguish short interpolatable gaps from long data outages.
        """
        lengths = pd.Series(0, index=mask.index, dtype=int)
        run = 0
        run_indices = []
        for idx, val in mask.items():
            if val:
                run += 1
                run_indices.append(idx)
            else:
                if run_indices:
                    for i in run_indices:
                        lengths[i] = run
                run = 0
                run_indices = []
        # flush last run
        if run_indices:
            for i in run_indices:
                lengths[i] = run
        return lengths

    # ------------------------------------------------------------------
    # 3. ADD FEATURES
    # ------------------------------------------------------------------

    def add_features(self) -> "AirQualityDataset":
        """
        Engineer lag, rolling, and calendar features from pm25_ugm3.

        CRITICAL -- no data leakage:
          - Lag features use .shift(n): row t uses pm25 from t-n, never t.
          - Rolling features use .shift(1).rolling(w): the window ends at
            t-1, so the current day's value is excluded.  This matches
            the real-world forecasting scenario where yesterday's reading
            is known but today's is the prediction target.
          - We do NOT call .rolling() directly on the unshifted series.
        """
        df = self.df.copy()

        pm = df["pm25_ugm3"]   # shorthand

        # -- Lag features --
        for lag in [1, 2, 3, 7]:
            df[f"pm25_lag{lag}"] = pm.shift(lag)

        # -- Rolling means (shift first to avoid leakage) --
        pm_shifted = pm.shift(1)   # yesterday and earlier
        df["pm25_roll3"] = pm_shifted.rolling(3).mean()
        df["pm25_roll7"] = pm_shifted.rolling(7).mean()

        # -- Calendar features --
        df["month"]          = df.index.month
        df["day_of_year"]    = df.index.day_of_year
        df["day_of_week"]    = df.index.day_of_week
        # Smog season: October (10) through January (1)
        # WHY: crop residue burning peaks Oct-Nov; winter thermal inversions
        # peak Dec-Jan.  This binary flag lets tree models easily split on season.
        df["is_smog_season"] = df.index.month.isin([10, 11, 12, 1]).astype(int)

        self.df = df
        return self

    # ------------------------------------------------------------------
    # 4. TIME SPLIT
    # ------------------------------------------------------------------

    def time_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split into train / validation / test sets by time order.

        WHY time-based, not random:
          Random splits would put future observations into the training set,
          creating leakage through lag and rolling features.  We always
          train on the past and evaluate on the future.

        Returns (train_df, val_df, test_df).
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
        """Persist the processed DataFrame to CSV."""
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