"""
download_weather.py -- fetch Open-Meteo historical weather for Lahore.

Open-Meteo provides free historical weather data with no API key required.
We request daily aggregates matching the AQI CSV date range.

WHY Open-Meteo:
  - No API key / no rate-limit for research use
  - ERA5-based reanalysis: consistent, gap-free, well-documented
  - Native Python SDK (openmeteo-requests) handles retries and caching

Usage:
    python -m src.download_weather        # saves to data/raw/lahore_weather.csv
    python -m src.download_weather --help
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

# Add project root to path so src.config resolves correctly when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import WEATHER_CSV, DATA_RAW

# Lahore coordinates (US Embassy monitor location is in Lahore, Punjab)
LATITUDE  = 31.5204
LONGITUDE = 74.3587

# Daily variables we want from Open-Meteo
# WHY these variables:
#   temperature_2m_mean  -- thermal inversions trap PM2.5 near the surface
#   precipitation_sum    -- rain scavenges particulates (wet deposition)
#   wind_speed_10m_max   -- horizontal dispersion of pollutants
#   wind_direction_10m_dominant -- identifies smog transport corridors
#   relative_humidity_2m_mean   -- hygroscopic growth increases apparent PM2.5
DAILY_VARIABLES = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",
]

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily weather for Lahore between start_date and end_date
    (both inclusive, format YYYY-MM-DD).

    Returns a DataFrame indexed by date with columns matching DAILY_VARIABLES.
    Raises requests.HTTPError on a non-200 response.
    """
    params = {
        "latitude":        LATITUDE,
        "longitude":       LONGITUDE,
        "start_date":      start_date,
        "end_date":        end_date,
        "daily":           ",".join(DAILY_VARIABLES),
        "timezone":        "Asia/Karachi",
        "wind_speed_unit": "kmh",
    }

    print(f"Fetching Open-Meteo data: {start_date} to {end_date} ...")
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()

    payload = resp.json()
    daily   = payload["daily"]

    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "date"})
    df = df.set_index("date")

    print(f"  Received {len(df)} daily rows.")
    return df


def main(start_date: str = "2019-01-01", end_date: str = "2025-02-28") -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    df = fetch_weather(start_date, end_date)
    df.to_csv(WEATHER_CSV)
    print(f"Saved to {WEATHER_CSV}")
    print(df.head())
    print(df.describe().round(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Open-Meteo weather for Lahore.")
    parser.add_argument("--start", default="2019-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2025-02-28", help="End date   YYYY-MM-DD")
    args = parser.parse_args()
    main(args.start, args.end)