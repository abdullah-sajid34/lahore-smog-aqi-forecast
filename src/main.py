"""
main.py -- Console menu for the Lahore PM2.5 Forecast System.

Usage:
    python src/main.py

Menu options
------------
1. EDA Summary        -- reads outputs/results/eda_summary.json
2. Model Comparison   -- reads outputs/results/model_comparison.json
3. Forecast for date  -- enter YYYY-MM-DD, get next-day PM2.5 and advisory
4. Quit

Error handling
--------------
- Missing result files  : clear message + "run python src/run_experiments.py"
- Missing model file    : same message
- Bad date string       : tells user the expected format
- Out-of-range date     : shows the valid date range
- Incomplete 7-day hist : explains which days are missing
None of these conditions crash the program; all are caught and reported.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    AGENT_MODEL_FILE,
    AQI_ADVISORY_DISCLAIMER,
    MERGED_CSV,
    OUTPUTS_RESULTS,
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

MONTH_ABBR = {
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec",
}

WEEKDAY_NAME = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

DIVIDER = "─" * 62


def _header(title: str) -> None:
    print()
    print(DIVIDER)
    print(f"  {title}")
    print(DIVIDER)


def _check_files() -> bool:
    """
    Return True if all required outputs exist.
    Print a helpful message and return False otherwise.
    """
    required = [
        OUTPUTS_RESULTS / "eda_summary.json",
        OUTPUTS_RESULTS / "model_comparison.json",
        AGENT_MODEL_FILE,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print()
        print("  [!] Required files are missing:")
        for m in missing:
            print(f"      {m}")
        print()
        print("  Run the following command to generate them:")
        print("      python src/run_experiments.py")
        return False
    return True


# ---------------------------------------------------------------------------
# Menu option 1: EDA Summary
# ---------------------------------------------------------------------------

def show_eda_summary() -> None:
    _header("EDA SUMMARY  (training data: May 2019 – Jun 2023)")
    path = OUTPUTS_RESULTS / "eda_summary.json"
    if not path.exists():
        print("  [!] eda_summary.json not found.")
        print("  Run: python src/run_experiments.py")
        return

    with open(path) as f:
        s = json.load(f)

    # Monthly means
    print("  Monthly mean PM2.5 (ug/m3):")
    monthly = s.get("monthly_mean_by_month", {})
    for m in range(1, 13):
        val = monthly.get(str(m), monthly.get(m, "N/A"))
        bar_len = int(float(val) / 10) if val != "N/A" else 0
        bar = "█" * bar_len
        print(f"    {MONTH_ABBR[m]:>3}: {str(val):>6}  {bar}")

    print()
    print(f"  Peak month            : {MONTH_ABBR[s.get('peak_month', 0)]} "
          f"({s.get('peak_month_mean_ugm3','?'):.1f} ug/m3)")
    print(f"  Smog season mean      : {s.get('smog_season_mean_ugm3','?'):.1f} ug/m3  "
          f"(Oct – Jan)")
    print(f"  Off-season mean       : {s.get('off_season_mean_ugm3','?'):.1f} ug/m3  "
          f"(Feb – Sep)")
    print(f"  Season ratio          : {s.get('smog_vs_offseason_ratio','?'):.2f}×")
    print()
    print(f"  Lag-1 autocorrelation : {s.get('autocorr_lag1','?'):.3f}")
    print(f"  Lag-7 autocorrelation : {s.get('autocorr_lag7','?'):.3f}")
    wc = s.get("weather_correlations", {})
    if wc:
        print("  Weather correlations with PM2.5:")
        for var, r in wc.items():
            short = var.replace("_2m_mean","").replace("_10m_max","").replace("_sum","")
            print(f"    {short:<30}: r = {r:+.3f}")
    print()
    wd = s.get("weekday_means_ugm3", {})
    if wd:
        print(f"  Weekday effect: {s.get('max_weekday','?')} highest "
              f"({wd.get(s.get('max_weekday',''),'?'):.1f} ug/m3), "
              f"{s.get('min_weekday','?')} lowest "
              f"({wd.get(s.get('min_weekday',''),'?'):.1f} ug/m3), "
              f"range {s.get('weekday_range_ugm3','?'):.1f} ug/m3")


# ---------------------------------------------------------------------------
# Menu option 2: Model Comparison
# ---------------------------------------------------------------------------

def show_model_comparison() -> None:
    _header("MODEL COMPARISON  (test set: Apr 2024 – Feb 2025)")
    path = OUTPUTS_RESULTS / "model_comparison.json"
    if not path.exists():
        print("  [!] model_comparison.json not found.")
        print("  Run: python src/run_experiments.py")
        return

    with open(path) as f:
        rows = json.load(f)

    # Header
    print(f"  {'Model':<22} {'RMSE':>7} {'±':>5} {'MAE':>7} {'±':>5} "
          f"{'F1':>6} {'Skill':>7}")
    print(f"  {'─'*22} {'─'*7} {'─'*5} {'─'*7} {'─'*5} {'─'*6} {'─'*7}")
    for r in rows:
        rmse_s = f"±{r.get('rmse_std',0):.3f}" if r.get('rmse_std',0) > 0 else "      "
        mae_s  = f"±{r.get('mae_std',0):.3f}"  if r.get('mae_std',0)  > 0 else "      "
        skill  = f"{r['skill']:+.4f}" if r.get('skill') is not None else "    --"
        print(f"  {r['model']:<22} {r['rmse']:>7.3f} {rmse_s:>5} "
              f"{r['mae']:>7.3f} {mae_s:>5} {r['f1']:>6.4f} {skill:>7}")

    print()
    print("  RMSE, MAE in ug/m3.  Skill = 1 − RMSE(model)/RMSE(persistence).")
    print("  Best model by validation CV-RMSE: RandomForest")


# ---------------------------------------------------------------------------
# Menu option 3: Forecast for date
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> pd.Timestamp | None:
    """
    Parse YYYY-MM-DD; return None and print error on failure.

    WHY check for NaT separately: pd.Timestamp("") returns NaT rather than
    raising, so we must treat NaT as a parse failure alongside exceptions.
    """
    stripped = date_str.strip()
    if not stripped:
        print(f"  [!] Cannot parse '{date_str}' as a date.")
        print("      Expected format: YYYY-MM-DD  (e.g. 2024-11-01)")
        return None
    try:
        ts = pd.Timestamp(stripped)
        if pd.isna(ts):
            raise ValueError("NaT")
        return ts
    except (ValueError, TypeError):
        print(f"  [!] Cannot parse '{date_str}' as a date.")
        print("      Expected format: YYYY-MM-DD  (e.g. 2024-11-01)")
        return None


def show_forecast(date_str: str) -> None:
    """
    Print next-day PM2.5 forecast and health advisory for a given date.
    All error conditions are caught and reported; never raises.
    """
    _header(f"FORECAST  for the day after {date_str.strip()}")

    # -- check required files --
    if not MERGED_CSV.exists():
        print("  [!] Cleaned dataset not found.")
        print("  Run: python src/run_experiments.py")
        return
    if not AGENT_MODEL_FILE.exists():
        print("  [!] Model file not found.")
        print(f"      Expected: {AGENT_MODEL_FILE}")
        print("  Run: python src/run_experiments.py")
        return

    # -- parse date --
    query_date = _parse_date(date_str)
    if query_date is None:
        return

    # -- load data --
    try:
        df = pd.read_csv(
            MERGED_CSV, parse_dates=["date"], index_col="date"
        ).sort_index()
    except Exception as exc:
        print(f"  [!] Could not load dataset: {exc}")
        return

    data_start = df.index.min()
    data_end   = df.index.max()

    # -- range check: query must be within data range and not the last date
    #    (we need t+1 to exist as a forecast date, though we don't need its data) --
    if query_date < data_start:
        print(f"  [!] Date {query_date.date()} is before the dataset start "
              f"({data_start.date()}).")
        return
    if query_date >= data_end:
        print(f"  [!] Date {query_date.date()} is at or after the last available "
              f"date ({data_end.date()}).")
        print(f"      Choose a date between {data_start.date()} and "
              f"{(data_end - pd.Timedelta(days=1)).date()}.")
        return

    # -- load agent and produce forecast --
    try:
        from src.agent import ForecastAgent
        agent  = ForecastAgent(model_path=AGENT_MODEL_FILE)
        result = agent.forecast_for_date(query_date, df)
    except FileNotFoundError as exc:
        print(f"  [!] {exc}")
        return
    except ValueError as exc:
        print(f"  [!] {exc}")
        return
    except Exception as exc:
        print(f"  [!] Unexpected error: {exc}")
        return

    # -- display --
    print(f"  Input date  (today)    : {result['query_date']}")
    print(f"  Forecast date (tomorrow): {result['forecast_date']}")
    print()
    print(f"  Predicted PM2.5        : {result['pm25_forecast']:.1f} ug/m3")
    print(f"  AQI Category           : {result['category']}")
    print()
    print(f"  Advisory  : {result['advisory']}")
    print()
    print(f"  ⚠  {result['disclaimer']}")


# ---------------------------------------------------------------------------
# Main menu loop
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      Lahore Air Quality Forecast System  (Phase 6)       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Warn once at startup if outputs are missing
    if not _check_files():
        print()
        input("  Press Enter to continue anyway (menu will show errors)...")

    while True:
        print()
        print("  1. EDA Summary")
        print("  2. Model Comparison")
        print("  3. Forecast for a date")
        print("  4. Quit")
        print()
        choice = input("  Enter choice (1-4): ").strip()

        if choice == "1":
            show_eda_summary()
        elif choice == "2":
            show_model_comparison()
        elif choice == "3":
            date_str = input("  Enter date (YYYY-MM-DD): ").strip()
            show_forecast(date_str)
        elif choice == "4":
            print()
            print("  Goodbye.")
            break
        else:
            print(f"  [!] '{choice}' is not a valid choice.  Enter 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()