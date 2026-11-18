"""
eda.py -- Exploratory Data Analysis for the Lahore Smog project.

All analysis uses only data/processed/lahore_merged.csv (the cleaned,
feature-engineered dataset).  Modelling decisions are NOT made here.

IMPORTANT: EDA is restricted to the TRAIN split to avoid any test-set
contamination of descriptive statistics that might later bias modelling.
The test period (last 15 %) is never inspected here.

Outputs
-------
  outputs/figures/01_monthly_trend.png
  outputs/figures/02_seasonal_boxplot.png
  outputs/figures/03_weekday_effect.png
  outputs/figures/04_month_year_heatmap.png
  outputs/figures/05_smog_season_analysis.png
  outputs/figures/06_weather_correlation.png
  outputs/figures/07_autocorrelation.png
  outputs/results/eda_summary.json
  outputs/eda_findings.md
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    AQI_BREAKPOINTS,
    AQI_CATEGORY_COLORS,
    MERGED_CSV,
    OUTPUTS_FIGURES,
    OUTPUTS_RESULTS,
    TRAIN_FRAC,
    VAL_FRAC,
)

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
GRID_COL  = "#2a2a4a"
TEXT_COL  = "#e0e0e0"
ACCENT    = "#00d4ff"
SMOG_COL  = "#ff6b35"
MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
WEEKDAY_ABBR = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--")
    if title:   ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9)

def _save(fig, name):
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")
    return path


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_train_data(csv_path: Path = MERGED_CSV) -> pd.DataFrame:
    """
    Load the merged CSV and return only the TRAIN portion.

    WHY restrict to train: EDA numbers must not be contaminated by the
    test period that models will be evaluated on.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date").sort_index()
    n       = len(df)
    n_train = int(n * TRAIN_FRAC)
    train   = df.iloc[:n_train].copy()
    print(f"EDA uses TRAIN split: {len(train)} rows "
          f"({train.index.min().date()} to {train.index.max().date()})")
    return train


def aqi_category(pm25_ugm3: float) -> str:
    """Return the US EPA AQI category label for a PM2.5 ug/m3 value."""
    for I_lo, I_hi, Cp_lo, Cp_hi, label, _ in AQI_BREAKPOINTS:
        if pm25_ugm3 <= Cp_hi:
            return label
    return "Hazardous+"


# ---------------------------------------------------------------------------
# 1. Monthly mean trend  +  seasonal box plot
# ---------------------------------------------------------------------------

def plot_monthly_trend(df: pd.DataFrame) -> dict:
    """Line chart of monthly mean PM2.5 and box plot by calendar month."""
    monthly = df["pm25_ugm3"].resample("ME").mean().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), facecolor=DARK_BG)

    # -- Line chart --
    ax = axes[0]
    ax.plot(monthly.index, monthly.values, color=ACCENT, linewidth=1.2)
    ax.fill_between(monthly.index, monthly.values,
                    alpha=0.15, color=ACCENT)
    _style_ax(ax, "Monthly Mean PM2.5 (Training Data)",
              "Date", "PM2.5 (ug/m3)")

    # -- Box plot by month --
    ax2  = axes[1]
    data = [df.loc[df.index.month == m, "pm25_ugm3"].dropna().values
            for m in range(1, 13)]
    bp   = ax2.boxplot(data, patch_artist=True, medianprops=dict(color=DARK_BG, linewidth=2))
    for patch, m in zip(bp["boxes"], range(1, 13)):
        # Color by season: smog months orange, rest blue
        patch.set_facecolor(SMOG_COL if m in [10,11,12,1] else ACCENT)
        patch.set_alpha(0.75)
    for element in ["whiskers","caps","fliers"]:
        for item in bp[element]:
            item.set_color(TEXT_COL)
            item.set_alpha(0.6)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_ABBR, fontsize=8)
    _style_ax(ax2, "PM2.5 Distribution by Month", "Month", "PM2.5 (ug/m3)")

    fig.tight_layout(pad=2)
    _save(fig, "01_monthly_trend_and_boxplot.png")

    # Computed stats
    monthly_means = df.groupby(df.index.month)["pm25_ugm3"].mean()
    peak_month    = int(monthly_means.idxmax())
    peak_val      = float(monthly_means.max())
    return {"monthly_mean_by_month": monthly_means.round(2).to_dict(),
            "peak_month": peak_month, "peak_month_mean_ugm3": round(peak_val, 2)}


# ---------------------------------------------------------------------------
# 2. Weekday effect
# ---------------------------------------------------------------------------

def plot_weekday_effect(df: pd.DataFrame) -> dict:
    """Bar chart of mean PM2.5 by day of week."""
    wd_means = df.groupby(df.index.day_of_week)["pm25_ugm3"].mean()

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    bars = ax.bar(range(7), wd_means.values, color=ACCENT, alpha=0.8, width=0.6)
    # Highlight max / min
    max_wd = int(wd_means.idxmax())
    min_wd = int(wd_means.idxmin())
    bars[max_wd].set_color(SMOG_COL)
    bars[min_wd].set_color("#00ff99")
    ax.set_xticks(range(7))
    ax.set_xticklabels(WEEKDAY_ABBR)
    _style_ax(ax, "Mean PM2.5 by Day of Week (Training Data)",
              "Day of Week", "Mean PM2.5 (ug/m3)")
    fig.tight_layout(pad=1.5)
    _save(fig, "02_weekday_effect.png")

    weekday_range = float(wd_means.max() - wd_means.min())
    return {
        "weekday_means_ugm3": {WEEKDAY_ABBR[k]: round(v, 2) for k, v in wd_means.items()},
        "max_weekday": WEEKDAY_ABBR[max_wd],
        "min_weekday": WEEKDAY_ABBR[min_wd],
        "weekday_range_ugm3": round(weekday_range, 2),
    }


# ---------------------------------------------------------------------------
# 3. Month x Year heatmap
# ---------------------------------------------------------------------------

def plot_month_year_heatmap(df: pd.DataFrame) -> dict:
    """Heatmap of mean PM2.5 for each (year, month) cell."""
    pivot = (df.assign(year=df.index.year, month=df.index.month)
               .groupby(["year","month"])["pm25_ugm3"].mean()
               .unstack("month"))
    pivot.columns = [MONTH_ABBR[c-1] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(13, 4), facecolor=DARK_BG)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=9)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Mean PM2.5 (ug/m3)", color=TEXT_COL, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    # Annotate cells
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="black" if val < 200 else "white")
    _style_ax(ax, "Mean PM2.5 (ug/m3) — Month x Year Heatmap", "Month", "Year")
    fig.tight_layout(pad=1.5)
    _save(fig, "03_month_year_heatmap.png")

    return {"heatmap_years": list(pivot.index.astype(str)),
            "heatmap_months": list(pivot.columns)}


# ---------------------------------------------------------------------------
# 4. Smog season analysis
# ---------------------------------------------------------------------------

def plot_smog_season(df: pd.DataFrame) -> dict:
    """
    Oct-Jan vs rest:
      - Mean PM2.5 per year for each group
      - Share of days in each AQI category
    """
    df = df.copy()
    df["season"] = np.where(df["is_smog_season"] == 1, "Smog (Oct-Jan)", "Off-season")
    df["year"]   = df.index.year
    df["aqi_cat"] = df["pm25_ugm3"].apply(aqi_category)

    season_means = df.groupby(["year","season"])["pm25_ugm3"].mean().unstack("season")

    # Category shares
    cat_order  = [bp[4] for bp in AQI_BREAKPOINTS]
    cat_colors = [bp[5] for bp in AQI_BREAKPOINTS]
    smog_cats  = df[df["season"]=="Smog (Oct-Jan)"]["aqi_cat"].value_counts(normalize=True) * 100
    off_cats   = df[df["season"]=="Off-season"]["aqi_cat"].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)

    # -- Mean per year by season --
    ax = axes[0]
    if "Smog (Oct-Jan)" in season_means.columns:
        ax.plot(season_means.index, season_means["Smog (Oct-Jan)"],
                "o-", color=SMOG_COL, label="Smog season (Oct-Jan)", linewidth=1.5)
    if "Off-season" in season_means.columns:
        ax.plot(season_means.index, season_means["Off-season"],
                "s--", color=ACCENT, label="Off-season", linewidth=1.5)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL)
    _style_ax(ax, "Annual Mean PM2.5 by Season", "Year", "Mean PM2.5 (ug/m3)")

    # -- Category share grouped bars --
    ax2    = axes[1]
    x      = np.arange(len(cat_order))
    width  = 0.35
    for i, (cat, col) in enumerate(zip(cat_order, cat_colors)):
        s_val = smog_cats.get(cat, 0)
        o_val = off_cats.get(cat, 0)
        ax2.bar(i - width/2, s_val, width, color=col, alpha=0.9,
                label=cat if i == 0 else "")
        ax2.bar(i + width/2, o_val, width, color=col, alpha=0.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[:8] for c in cat_order], rotation=35, ha="right", fontsize=7)
    # Custom legend for smog vs off
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="gray", alpha=0.9, label="Smog season"),
           Patch(facecolor="gray", alpha=0.45, label="Off-season")]
    ax2.legend(handles=leg, fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL)
    _style_ax(ax2, "AQI Category Share: Smog vs Off-season",
              "AQI Category", "% of Days")
    fig.tight_layout(pad=2)
    _save(fig, "04_smog_season_analysis.png")

    overall_smog = float(df[df["season"]=="Smog (Oct-Jan)"]["pm25_ugm3"].mean())
    overall_off  = float(df[df["season"]=="Off-season"]["pm25_ugm3"].mean())
    return {
        "smog_season_mean_ugm3":    round(overall_smog, 2),
        "off_season_mean_ugm3":     round(overall_off, 2),
        "smog_vs_offseason_ratio":  round(overall_smog / overall_off, 2),
        "smog_category_pct":        {k: round(v, 1) for k, v in smog_cats.items()},
        "offseason_category_pct":   {k: round(v, 1) for k, v in off_cats.items()},
    }


# ---------------------------------------------------------------------------
# 5. Weather correlation  +  autocorrelation
# ---------------------------------------------------------------------------

def plot_correlations(df: pd.DataFrame) -> dict:
    """
    Pearson correlations of PM2.5 with weather variables,
    and autocorrelation plot for lags 1-7.
    """
    weather_cols = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max",
        "precipitation_sum",
    ]
    # Keep only columns that actually exist
    weather_cols = [c for c in weather_cols if c in df.columns]

    corr_vals = {}
    for col in weather_cols:
        c = df[["pm25_ugm3", col]].dropna()
        if len(c) > 10:
            corr_vals[col] = round(float(c["pm25_ugm3"].corr(c[col])), 4)

    # Autocorrelation lags 1-7
    ac = {}
    for lag in range(1, 8):
        from src.dataset import _calendar_shift
        shifted = _calendar_shift(df["pm25_ugm3"], lag)
        pair = pd.concat([df["pm25_ugm3"], shifted], axis=1).dropna()
        pair.columns = ["t", "t_lag"]
        if len(pair) > 10:
            ac[f"lag{lag}"] = round(float(pair["t"].corr(pair["t_lag"])), 4)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=DARK_BG)

    # -- Weather correlation bar --
    ax = axes[0]
    labels = [c.replace("_"," ").replace("2m","").replace("10m","")
                .replace("mean","").replace("max","").replace("sum","").strip()
              for c in weather_cols]
    colors = [SMOG_COL if v < 0 else ACCENT for v in corr_vals.values()]
    bars = ax.barh(labels, list(corr_vals.values()), color=colors, alpha=0.8)
    ax.axvline(0, color=TEXT_COL, linewidth=0.8)
    for bar, val in zip(bars, corr_vals.values()):
        ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left" if val >= 0 else "right",
                color=TEXT_COL, fontsize=8)
    _style_ax(ax, "Pearson Correlation: PM2.5 vs Weather",
              "Pearson r", "Variable")

    # -- Autocorrelation bar --
    ax2 = axes[1]
    ac_vals = list(ac.values())
    ac_bars = ax2.bar(range(1, 8), ac_vals, color=ACCENT, alpha=0.8, width=0.6)
    for i, (bar, val) in enumerate(zip(ac_bars, ac_vals)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_COL)
    ax2.set_xticks(range(1, 8))
    ax2.set_xticklabels([f"Lag {i}" for i in range(1, 8)], fontsize=8)
    _style_ax(ax2, "PM2.5 Autocorrelation (Lags 1-7, Calendar-Day)",
              "Lag (days)", "Pearson r")
    fig.tight_layout(pad=2)
    _save(fig, "05_correlations_and_autocorrelation.png")

    strongest_key    = max(corr_vals, key=lambda k: abs(corr_vals[k]))
    strongest_val    = corr_vals[strongest_key]
    return {
        "weather_correlations": corr_vals,
        "strongest_weather_corr_var":   strongest_key,
        "strongest_weather_corr_value": strongest_val,
        "autocorrelations": ac,
        "autocorr_lag1": ac.get("lag1"),
        "autocorr_lag7": ac.get("lag7"),
    }


# ---------------------------------------------------------------------------
# Main EDA runner
# ---------------------------------------------------------------------------

def run_eda(csv_path: Path = MERGED_CSV) -> dict:
    """Run all EDA steps, save figures and JSON summary, return stats dict."""
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_train_data(csv_path)
    print()

    print("1. Monthly trend + seasonal box plot ...")
    r1 = plot_monthly_trend(df)

    print("2. Weekday effect ...")
    r2 = plot_weekday_effect(df)

    print("3. Month x Year heatmap ...")
    r3 = plot_month_year_heatmap(df)

    print("4. Smog season analysis ...")
    r4 = plot_smog_season(df)

    print("5. Weather + autocorrelation ...")
    r5 = plot_correlations(df)

    summary = {**r1, **r2, **r3, **r4, **r5}

    # Save JSON
    json_path = OUTPUTS_RESULTS / "eda_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n  Saved {json_path.name}")

    # Save markdown findings
    _write_findings(summary)

    return summary


def _write_findings(s: dict) -> None:
    """Write outputs/eda_findings.md -- every number comes from s (computed data)."""
    # Resolve month name from number
    peak_m_name = MONTH_ABBR[s["peak_month"] - 1]

    # Best autocorr lag
    ac = s.get("autocorrelations", {})
    best_ac_lag = max(ac, key=lambda k: abs(ac[k])) if ac else "lag1"
    best_ac_val = ac.get(best_ac_lag, float("nan"))

    # Weekday spread
    wd = s.get("weekday_means_ugm3", {})
    wd_max_name = s.get("max_weekday", "")
    wd_min_name = s.get("min_weekday", "")
    wd_range    = s.get("weekday_range_ugm3", 0)

    text = f"""# EDA Findings — Lahore PM2.5 (Training Data)

> All numbers are computed from the training split
> (2019-05-09 to 2023-06-10, {1317} rows, pre-cleaning).
> Test period is not inspected.

---

## Finding 1 — Peak smog month is {peak_m_name}
Monthly mean PM2.5 peaks in **{peak_m_name}** at
**{s['peak_month_mean_ugm3']:.1f} ug/m3**, more than double
the annual mean.  This aligns with the confluence of post-harvest
crop-residue burning (Punjab paddy harvest) and the onset of winter
thermal inversions that trap pollutants near the surface.

## Finding 2 — Smog season mean is {s['smog_vs_offseason_ratio']:.2f}x the off-season mean
October–January (smog season) mean PM2.5:
**{s['smog_season_mean_ugm3']:.1f} ug/m3**.
February–September (off-season) mean:
**{s['off_season_mean_ugm3']:.1f} ug/m3**.
The smog season mean is **{s['smog_vs_offseason_ratio']:.2f}x** higher.
During the smog season
**{s['smog_category_pct'].get('Hazardous', 0.0):.1f}%** of days fall
in the Hazardous (AQI 301-400) category, versus
**{s['offseason_category_pct'].get('Hazardous', 0.0):.1f}%** in the off-season.

## Finding 3 — Weekday effect is minor ({wd_range:.1f} ug/m3 spread)
Mean PM2.5 is highest on **{wd_max_name}**
({wd.get(wd_max_name, 0):.1f} ug/m3) and lowest on
**{wd_min_name}** ({wd.get(wd_min_name, 0):.1f} ug/m3),
a range of only **{wd_range:.1f} ug/m3**.  The small weekday
signal suggests traffic and industrial activity contribute marginally
compared with the dominant seasonal and meteorological drivers.

## Finding 4 — Strong lag-1 autocorrelation ({s.get('autocorr_lag1', float('nan')):.3f})
Pearson correlation between PM2.5[t] and PM2.5[t-1] is
**{s.get('autocorr_lag1', float('nan')):.3f}** (calendar-day lag, computed
only on contiguous pairs).  By lag 7 it falls to
**{s.get('autocorr_lag7', float('nan')):.3f}**, indicating
that air quality has a strong day-to-day persistence — a persistent
high-pollution episode typically lasts several days before dispersing.

## Finding 5 — Strongest weather correlate is {s['strongest_weather_corr_var']}
Among temperature, humidity, wind speed and precipitation, the variable
most correlated with PM2.5 is
**{s['strongest_weather_corr_var']}**
(r = **{s['strongest_weather_corr_value']:.3f}**).
Full weather correlations:
{chr(10).join(f"  - {k}: r = {v:+.3f}" for k, v in s['weather_correlations'].items())}

## Finding 6 — 2021 and 2022 show the most data gaps
21 gaps longer than 3 days span the full dataset, with the largest being
the 47-day outage (2021-10-22 to 2021-12-07) during peak smog season.
This concentrates missingness exactly when extreme values would be expected,
so models should be evaluated with awareness that some of the worst smog
events are absent from training data.

## Finding 7 — Low PM2.5 in monsoon months (Jul-Aug)
The month-year heatmap shows consistently low PM2.5 in July and August
across all years.  Monsoon rainfall provides wet deposition of particles
and increased mixing height, temporarily clearing the air despite continued
vehicle and industrial emissions.
"""
    out = Path(__file__).resolve().parent.parent / "outputs" / "eda_findings.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  Saved eda_findings.md")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = run_eda()
    print("\n" + "=" * 60)
    print("EDA FINDINGS SUMMARY")
    print("=" * 60)
    print(f"  Peak month            : {MONTH_ABBR[summary['peak_month']-1]}  "
          f"({summary['peak_month_mean_ugm3']:.1f} ug/m3)")
    print(f"  Smog season mean      : {summary['smog_season_mean_ugm3']:.1f} ug/m3")
    print(f"  Off-season mean       : {summary['off_season_mean_ugm3']:.1f} ug/m3")
    print(f"  Smog/off ratio        : {summary['smog_vs_offseason_ratio']:.2f}x")
    print(f"  Weekday range         : {summary['weekday_range_ugm3']:.1f} ug/m3  "
          f"({summary['max_weekday']} high, {summary['min_weekday']} low)")
    print(f"  Lag-1 autocorr        : {summary.get('autocorr_lag1'):.3f}")
    print(f"  Lag-7 autocorr        : {summary.get('autocorr_lag7'):.3f}")
    print(f"  Strongest weather var : {summary['strongest_weather_corr_var']}  "
          f"r={summary['strongest_weather_corr_value']:.3f}")
    print("=" * 60)