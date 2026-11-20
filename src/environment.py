"""
environment.py -- AirQualityEnvironment for the Lahore PM2.5 agent.

Design
------
The environment replays the cleaned, feature-engineered series from
data/processed/lahore_merged.csv one calendar day at a time.

It acts as the "world" that the ForecastAgent interacts with.  The agent
calls env.step() to receive the next Percept, then uses that observation
to update its internal history and produce a forecast for the following day.

Key design decisions
--------------------
Calendar days, not row positions:
    After cleaning, 231 long-gap rows were dropped, so the DataFrame index
    is non-contiguous.  Iterating by row position would skip over these gaps
    silently, making the agent believe consecutive rows are consecutive days.
    Instead we maintain a full daily calendar cursor and ask "is this calendar
    date in the DataFrame?" at each step.

Start date:
    The first date with a valid 7-day lag history is used as the start, so
    the agent never receives a Percept for a day it cannot build features for.
    That date is determined from the data itself, not hardcoded.

Percept:
    A frozen dataclass -- the agent should never mutate the environment's
    state through the percept.  It carries only what a real sensor network
    would report at the end of day t.

reset(seed):
    Re-winds the cursor to the start date.  The optional seed is accepted
    for API compatibility with RL-style loops but does nothing (deterministic
    replay).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MERGED_CSV


# ---------------------------------------------------------------------------
# Percept dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Percept:
    """
    The observation the environment delivers to the agent at the end of day t.

    Fields
    ------
    date        : calendar date of this observation
    pm25        : measured PM2.5 (ug/m3) -- the actual sensor reading
    temperature : daily mean temperature (°C)
    humidity    : daily mean relative humidity (%)
    wind_speed  : maximum 10-m wind speed (km/h)
    rainfall    : total precipitation (mm)

    WHY frozen: the agent must not be able to modify the environment's
    observation.  Frozen dataclasses raise FrozenInstanceError on assignment.
    """
    date:        pd.Timestamp
    pm25:        float
    temperature: float
    humidity:    float
    wind_speed:  float
    rainfall:    float


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AirQualityEnvironment:
    """
    Replays the cleaned Lahore PM2.5 series day by day.

    Usage
    -----
        env = AirQualityEnvironment()
        env.reset()
        while not env.is_done():
            percept = env.step()
            agent.perceive(percept)
            prediction = agent.forecast()

    Parameters
    ----------
    csv_path : path to lahore_merged.csv (defaults to config.MERGED_CSV)
    min_history_days : number of prior calendar days needed before the first
                       valid Percept (default 7, matching MODEL_LAG_COLS lag6).
    """

    def __init__(
        self,
        csv_path: Path = MERGED_CSV,
        min_history_days: int = 7,
    ):
        self._csv_path         = Path(csv_path)
        self._min_history_days = min_history_days
        self._df: pd.DataFrame | None = None
        self._cursor: pd.Timestamp | None = None
        self._start: pd.Timestamp | None  = None
        self._end:   pd.Timestamp | None  = None
        self._load()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the CSV and determine start / end dates."""
        df = pd.read_csv(
            self._csv_path,
            parse_dates=["date"],
            index_col="date",
        ).sort_index()
        self._df = df

        # Start: first date that has at least min_history_days calendar days
        # of prior observations in the index.
        #
        # WHY not just take df.index[min_history_days]:
        #   That uses row position.  After gap-dropping, row N may be many
        #   calendar days after row N-7.  We need to find the first date d
        #   such that d - min_history_days calendar days is also in the index.
        #
        # Efficient approach: build the full daily calendar and check.
        full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
        present    = set(df.index)

        start = None
        for date in full_range:
            if date not in present:
                continue
            # Check that all 7 prior calendar days are in the index.
            prior_dates = pd.date_range(
                date - pd.Timedelta(days=self._min_history_days),
                date - pd.Timedelta(days=1),
                freq="D",
            )
            if all(d in present for d in prior_dates):
                start = date
                break

        if start is None:
            raise RuntimeError("No date found with a full 7-day prior history.")

        self._start = start
        self._end   = df.index.max()
        self._cursor = start

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> "AirQualityEnvironment":
        """
        Rewind cursor to the first valid date.

        seed is accepted for interface compatibility with RL training loops
        but has no effect -- the replay is deterministic.
        """
        self._cursor = self._start
        return self

    def step(self) -> Percept:
        """
        Return the Percept for the current cursor date and advance one
        calendar day (skipping missing dates in the index).

        Raises StopIteration if called after is_done() is True.
        """
        if self.is_done():
            raise StopIteration("Environment has reached the end of the series.")

        current = self._cursor
        row     = self._df.loc[current]

        percept = Percept(
            date        = current,
            pm25        = float(row["pm25_ugm3"]),
            temperature = float(row["temperature_2m_mean"]),
            humidity    = float(row["relative_humidity_2m_mean"]),
            wind_speed  = float(row["wind_speed_10m_max"]),
            rainfall    = float(row["precipitation_sum"]),
        )

        # Advance to the next date that is present in the index.
        # WHY skip missing dates: we still want the cursor to stay on
        # observed dates.  The agent handles calendar gaps internally via
        # its history buffer.
        present = set(self._df.index)
        next_date = current + pd.Timedelta(days=1)
        # Walk forward until we find the next present date or exceed end
        while next_date <= self._end and next_date not in present:
            next_date += pd.Timedelta(days=1)

        self._cursor = next_date  # may be > self._end, is_done() catches that

        return percept

    def is_done(self) -> bool:
        """True when the cursor has moved past the last available date."""
        return self._cursor > self._end

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def start_date(self) -> pd.Timestamp:
        return self._start

    @property
    def end_date(self) -> pd.Timestamp:
        return self._end

    @property
    def df(self) -> pd.DataFrame:
        """Read-only access to the underlying DataFrame (for the agent)."""
        return self._df.copy()

    @property
    def n_steps(self) -> int:
        """Number of observed dates from start to end inclusive."""
        present = set(self._df.index)
        return sum(1 for d in pd.date_range(self._start, self._end, freq="D")
                   if d in present)