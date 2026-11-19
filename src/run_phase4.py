"""
run_phase4.py -- Kalman filter experiment: raw vs filtered PM2.5 as input.

Experiment design
-----------------
Hypothesis: replacing raw PM2.5 lags with Kalman-filtered lags gives the
Random Forest and Ridge forecasters a smoother, less noisy input signal,
which may improve generalisation on the test period.

What changes from Phase 3:
  - The lag and rolling features (pm25_lag1..lag6, pm25_roll7) are built
    from the FILTERED series instead of the raw series.
  - Weather and calendar features are UNCHANGED.
  - The TARGET is still raw PM2.5 at t+1 (we want to predict real, noisy
    PM2.5 -- the forecast is what matters, not the smooth proxy).

What does NOT change:
  - Time split: same 70/15/15 as Phases 1-3.
  - Model hyper-parameters: reused from best_params.json.
  - Evaluation code: same RMSE / MAE / macro-F1 / skill as Phase 3.
  - Random seed grid: seeds 42-51 for RF.

The Kalman filter runs CAUSALLY over the ENTIRE timeline (train+val+test)
in date order.  This is correct: the filter only uses past observations,
so test estimates do not leak future information.  R and Q are chosen on
train+val only; the test-period filter simply continues the already-fitted
filter forward.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    HORIZON_1,
    MERGED_CSV,
    MODEL_FEATURES,
    OUTPUTS_FIGURES,
    OUTPUTS_MODELS,
    OUTPUTS_RESULTS,
    RANDOM_SEED,
)
from src.dataset import _calendar_shift, _calendar_rolling
from src.forecasters import (
    PersistenceForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    build_model_features,
    pm25_to_aqi_category,
)
from src.kalman import KalmanFilter1D, estimate_R_from_training, tune_Q
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Metric helper (duplicated from run_phase3 to keep Phase 4 self-contained)
# ---------------------------------------------------------------------------

def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

def _mae(a, b):
    return float(mean_absolute_error(a, b))

def _f1(a, b):
    return float(f1_score(
        pm25_to_aqi_category(a), pm25_to_aqi_category(b),
        average="macro", zero_division=0,
    ))

def compute_metrics(y_true, y_pred, rmse_baseline=None):
    r = _rmse(y_true, y_pred)
    m = _mae(y_true,  y_pred)
    f = _f1(y_true,   y_pred)
    skill = round(1.0 - r / rmse_baseline, 4) if rmse_baseline else None
    return {"rmse": round(r,4), "mae": round(m,4), "f1": round(f,4), "skill": skill}


# ---------------------------------------------------------------------------
# Build filtered feature matrix
# ---------------------------------------------------------------------------

def build_filtered_features(
    df: pd.DataFrame,
    filtered_series: pd.Series,
    horizon: int = HORIZON_1,
) -> pd.DataFrame:
    """
    Build the MODEL_FEATURES matrix using filtered PM2.5 for lag/rolling
    columns, keeping weather and calendar features from the original df.

    WHY replace only lag/rolling columns:
      Lag and rolling features carry the dominant noise from the raw sensor.
      Filtering them reduces input noise while leaving weather and calendar
      features (which have different noise characteristics) untouched.
      The TARGET stays raw pm25_ugm3 so we evaluate against real readings.

    Calendar-day correctness:
      _calendar_shift and _calendar_rolling are used exactly as in Phase 3.
      filtered_series has the same DatetimeIndex as df, so the gap-handling
      is inherited automatically.
    """
    out = df.copy()

    # Replace raw lag features with filtered equivalents
    for lag in [1, 2, 3, 4, 5, 6]:
        out[f"pm25_lag{lag}"] = _calendar_shift(filtered_series, lag)

    # Replace rolling feature
    out["pm25_roll7"] = _calendar_rolling(filtered_series, 7)

    # Build target (raw pm25_ugm3 at t+horizon, calendar-safe)
    full_idx    = pd.date_range(out.index.min(), out.index.max(), freq="D")
    pm_dense    = out["pm25_ugm3"].reindex(full_idx)
    target_dense = pm_dense.shift(-horizon)
    out["target"] = target_dense.reindex(out.index)

    # Add cyclic and lag4-6 columns (they come from filtered series above)
    out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12)
    out["doy_sin"]   = np.sin(2 * np.pi * out.index.day_of_year / 365)
    out["doy_cos"]   = np.cos(2 * np.pi * out.index.day_of_year / 365)

    cols_needed = MODEL_FEATURES + ["target"]
    missing = [c for c in MODEL_FEATURES if c not in out.columns]
    if missing:
        raise ValueError(f"build_filtered_features: missing cols {missing}")

    return out[cols_needed].dropna()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COL = "#2a2a4a"
TEXT_COL = "#e0e0e0"
CYAN     = "#22D3EE"


def plot_filtered_vs_raw(
    df: pd.DataFrame,
    filtered_series: pd.Series,
    zoom_start: str = "2024-10-01",
    zoom_end:   str = "2025-02-18",
) -> None:
    """
    Two-panel figure:
      Top:    full timeline, raw (faded) + Kalman filtered (bold cyan)
      Bottom: one smog-season zoom (Oct 2024 - Feb 2025)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)

    dates_raw   = df.index
    raw_vals    = df["pm25_ugm3"].values
    filt_vals   = filtered_series.reindex(df.index).values

    # -- Top: full timeline --
    ax1.plot(dates_raw, raw_vals,  color=TEXT_COL, alpha=0.3, linewidth=0.6,
             label="Raw PM2.5")
    ax1.plot(dates_raw, filt_vals, color=CYAN,     alpha=0.95, linewidth=1.4,
             label="Kalman filtered")
    ax1.set_facecolor(PANEL_BG)
    ax1.tick_params(colors=TEXT_COL, labelsize=8)
    ax1.xaxis.label.set_color(TEXT_COL); ax1.yaxis.label.set_color(TEXT_COL)
    ax1.title.set_color(TEXT_COL)
    for sp in ax1.spines.values(): sp.set_edgecolor(GRID_COL)
    ax1.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--")
    ax1.set_title("Lahore PM2.5: Raw vs Kalman Filtered (full timeline)", fontsize=11)
    ax1.set_ylabel("PM2.5 (ug/m3)", fontsize=9)
    leg = ax1.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL,
                     framealpha=0.8)

    # -- Bottom: zoom on one smog season --
    zm_start = pd.Timestamp(zoom_start)
    zm_end   = pd.Timestamp(zoom_end)
    mask     = (df.index >= zm_start) & (df.index <= zm_end)
    ax2.plot(df.index[mask], raw_vals[mask],  color=TEXT_COL, alpha=0.4,
             linewidth=0.8, label="Raw PM2.5")
    ax2.plot(df.index[mask], filt_vals[mask], color=CYAN,     alpha=0.95,
             linewidth=1.8, label="Kalman filtered")
    ax2.set_facecolor(PANEL_BG)
    ax2.tick_params(colors=TEXT_COL, labelsize=8)
    ax2.xaxis.label.set_color(TEXT_COL); ax2.yaxis.label.set_color(TEXT_COL)
    ax2.title.set_color(TEXT_COL)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID_COL)
    ax2.grid(True, color=GRID_COL, linewidth=0.4, linestyle="--")
    ax2.set_title(f"Zoom: smog season {zoom_start} to {zoom_end}", fontsize=10)
    ax2.set_ylabel("PM2.5 (ug/m3)", fontsize=9)
    ax2.set_xlabel("Date", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.8)

    fig.tight_layout(pad=2.5)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_FIGURES / "kalman_filtered_vs_raw.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(csv_path: Path = MERGED_CSV) -> None:
    set_seed()
    OUTPUTS_RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)

    # -- Load --
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date").sort_index()
    n       = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train    = df.iloc[:n_train]
    val      = df.iloc[n_train : n_train + n_val]
    test     = df.iloc[n_train + n_val :]
    trainval = df.iloc[:n_train + n_val]

    print("=" * 70)
    print("PHASE 4 — KALMAN FILTER EXPERIMENT")
    print("=" * 70)
    print(f"  Train    : {len(train)} rows  "
          f"({train.index.min().date()} to {train.index.max().date()})")
    print(f"  Val      : {len(val)} rows  "
          f"({val.index.min().date()} to {val.index.max().date()})")
    print(f"  Test     : {len(test)} rows  "
          f"({test.index.min().date()} to {test.index.max().date()})")
    print()

    # ------------------------------------------------------------------
    # 1. Estimate R (training period only)
    # ------------------------------------------------------------------
    R = estimate_R_from_training(train["pm25_ugm3"])
    print(f"  R (obs noise variance, from training residuals): {R:.4f} (ug/m3)^2")
    print(f"  R (std): {R**0.5:.4f} ug/m3")

    # ------------------------------------------------------------------
    # 2. Tune Q (train+val, test never touched)
    # ------------------------------------------------------------------
    print("  Tuning Q on train+val (TimeSeriesSplit grid) ...")
    q_result = tune_Q(
        train["pm25_ugm3"],
        val["pm25_ugm3"],
        R=R,
        q_fracs=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    )
    Q = q_result["best_Q"]
    print(f"  Q/R scores: {q_result['all_scores']}")
    print(f"  => Best Q={Q:.4f}  (Q/R ratio={q_result['best_q_frac']})  "
          f"CV-RMSE={q_result['cv_rmse']:.4f}")
    print()

    # Save Kalman params
    kalman_params = {"Q": Q, "R": R, "best_q_frac": q_result["best_q_frac"],
                     "cv_rmse": q_result["cv_rmse"],
                     "R_estimation_method": (
                         "Variance of residuals from 3-day centred rolling mean "
                         "applied to training period only"
                     )}
    with open(OUTPUTS_RESULTS / "kalman_params.json", "w") as fh:
        json.dump(kalman_params, fh, indent=2)
    print(f"  Saved kalman_params.json")

    # ------------------------------------------------------------------
    # 3. Run Kalman filter CAUSALLY over entire timeline (train+val+test)
    # ------------------------------------------------------------------
    # WHY run over the full timeline:
    #   The filter must have seen all prior observations to produce a settled
    #   estimate at the start of the test period.  Running only on train+val
    #   would leave the filter "cold" at the start of test.
    #   Because the filter is causal (uses only past z), running it over test
    #   is not data leakage -- it is the production scenario.
    kf_full = KalmanFilter1D(Q=Q, R=R)
    filtered_vals, filtered_vars = kf_full.filter_series(
        df["pm25_ugm3"], return_variance=True
    )
    filtered_series = pd.Series(filtered_vals, index=df.index, name="pm25_filtered")

    # Save kalman_series.json
    kalman_records = []
    for i, (date, raw) in enumerate(df["pm25_ugm3"].items()):
        kalman_records.append({
            "date":     str(date.date()),
            "raw":      round(float(raw), 4),
            "filtered": round(float(filtered_vals[i]), 4),
            "variance": round(float(filtered_vars[i]), 4),
        })
    ks_path = OUTPUTS_RESULTS / "kalman_series.json"
    with open(ks_path, "w") as fh:
        json.dump({"Q": Q, "R": R, "series": kalman_records}, fh, indent=2)
    print(f"  Saved kalman_series.json  ({len(kalman_records)} records)")

    # Smoothness report
    raw_diff_std  = float(df["pm25_ugm3"].diff().std())
    filt_diff_std = float(filtered_series.diff().std())
    print(f"  Smoothing check: raw day-to-day std={raw_diff_std:.2f}  "
          f"filtered day-to-day std={filt_diff_std:.2f}  "
          f"ratio={filt_diff_std/raw_diff_std:.3f}")

    # ------------------------------------------------------------------
    # 4. Build feature matrices (raw vs filtered)
    # ------------------------------------------------------------------
    feat_trainval_raw  = build_model_features(trainval, horizon=HORIZON_1)
    feat_test_raw      = build_model_features(test,     horizon=HORIZON_1)

    feat_trainval_filt = build_filtered_features(trainval, filtered_series,  horizon=HORIZON_1)
    feat_test_filt     = build_filtered_features(test,     filtered_series,  horizon=HORIZON_1)

    print(f"\n  Feature rows trainval raw : {len(feat_trainval_raw)}")
    print(f"  Feature rows test     raw : {len(feat_test_raw)}")
    print(f"  Feature rows trainval filt: {len(feat_trainval_filt)}")
    print(f"  Feature rows test     filt: {len(feat_test_filt)}")

    # Verify no test contamination in feature build
    assert len(set(feat_trainval_filt.index) & set(feat_test_filt.index)) == 0

    # ------------------------------------------------------------------
    # 5. Load Phase 3 hyper-parameters
    # ------------------------------------------------------------------
    with open(OUTPUTS_RESULTS / "best_params.json") as fh:
        bp = json.load(fh)

    ridge_alpha = bp["ridge_h1"]["best_alpha"]
    rf_n_est    = bp["rf_h1"]["best_n_estimators"]
    rf_max_d    = bp["rf_h1"]["best_max_depth"]
    rf_min_sl   = bp["rf_h1"]["best_min_samples_leaf"]

    print(f"\n  Using Phase 3 params: Ridge alpha={ridge_alpha}, "
          f"RF n_est={rf_n_est}, max_depth={rf_max_d}, min_leaf={rf_min_sl}")

    # ------------------------------------------------------------------
    # 6. Evaluate: Persistence baseline (same for both raw and filtered)
    # ------------------------------------------------------------------
    y_te_raw  = feat_test_raw["target"].values
    y_te_filt = feat_test_filt["target"].values

    # Persistence on the RAW test features (reference baseline)
    pers = PersistenceForecaster(horizon=HORIZON_1)
    pers.fit(feat_test_raw[MODEL_FEATURES], pd.Series(y_te_raw))
    pers_preds   = pers.predict(feat_test_raw[MODEL_FEATURES])
    pers_metrics = compute_metrics(y_te_raw, pers_preds)
    rmse_pers    = pers_metrics["rmse"]

    # ------------------------------------------------------------------
    # 7. RF seeds 42-51: raw vs filtered
    # ------------------------------------------------------------------
    def _seed_loop(feat_tv, feat_te, label):
        y_te  = feat_te["target"].values
        rmses, maes, f1s = [], [], []
        for seed in range(42, 52):
            m = RandomForestForecaster(
                n_estimators=rf_n_est, max_depth=rf_max_d,
                min_samples_leaf=rf_min_sl, random_state=seed
            )
            m.fit(feat_tv[MODEL_FEATURES], feat_tv["target"])
            p = m.predict(feat_te[MODEL_FEATURES])
            rmses.append(_rmse(y_te, p))
            maes.append(_mae(y_te,  p))
            f1s.append(_f1(y_te,    p))
        return {
            "rmse_mean": round(float(np.mean(rmses)), 4),
            "rmse_std":  round(float(np.std(rmses)),  4),
            "mae_mean":  round(float(np.mean(maes)),  4),
            "mae_std":   round(float(np.std(maes)),   4),
            "f1_mean":   round(float(np.mean(f1s)),   4),
            "f1_std":    round(float(np.std(f1s)),    4),
            "skill":     round(1.0 - float(np.mean(rmses)) / rmse_pers, 4),
        }

    print("\n  Running RF seed loop (raw) ...")
    rf_raw_stats  = _seed_loop(feat_trainval_raw,  feat_test_raw,  "raw")
    print("  Running RF seed loop (filtered) ...")
    rf_filt_stats = _seed_loop(feat_trainval_filt, feat_test_filt, "filtered")

    # ------------------------------------------------------------------
    # 8. Ridge raw vs filtered (single fit, deterministic)
    # ------------------------------------------------------------------
    ridge_raw = RidgeForecaster(alpha=ridge_alpha)
    ridge_raw.fit(feat_trainval_raw[MODEL_FEATURES], feat_trainval_raw["target"])
    ridge_raw_preds   = ridge_raw.predict(feat_test_raw[MODEL_FEATURES])
    ridge_raw_metrics = compute_metrics(y_te_raw, ridge_raw_preds, rmse_pers)

    ridge_filt = RidgeForecaster(alpha=ridge_alpha)
    ridge_filt.fit(feat_trainval_filt[MODEL_FEATURES], feat_trainval_filt["target"])
    ridge_filt_preds   = ridge_filt.predict(feat_test_filt[MODEL_FEATURES])
    ridge_filt_metrics = compute_metrics(y_te_filt, ridge_filt_preds, rmse_pers)

    # Save filtered-input models
    rf_filt_model = RandomForestForecaster(
        n_estimators=rf_n_est, max_depth=rf_max_d,
        min_samples_leaf=rf_min_sl, random_state=RANDOM_SEED
    )
    rf_filt_model.fit(feat_trainval_filt[MODEL_FEATURES], feat_trainval_filt["target"])
    joblib.dump(rf_filt_model,  OUTPUTS_MODELS / "rf_filtered_h1.joblib")
    joblib.dump(ridge_filt,     OUTPUTS_MODELS / "ridge_filtered_h1.joblib")
    print(f"  Saved rf_filtered_h1.joblib  ridge_filtered_h1.joblib")

    # ------------------------------------------------------------------
    # 9. Print results table
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PHASE 4 RESULTS TABLE — test set only")
    print("=" * 70)
    print(f"  {'Model':<35s}  RMSE       MAE       F1      Skill")
    print(f"  {'-'*35}  ---------  --------  ------  --------")

    def _row(name, rmse_m, rmse_s, mae_m, mae_s, f1_m, f1_s, skill):
        sk = f"{skill:+.4f}" if skill is not None else "    --  "
        if rmse_s > 0:
            return (f"  {name:<35s}  {rmse_m:.3f}+/-{rmse_s:.3f}  "
                    f"{mae_m:.3f}+/-{mae_s:.3f}  {f1_m:.4f}+/-{f1_s:.4f}  {sk}")
        else:
            return (f"  {name:<35s}  {rmse_m:.3f}          "
                    f"{mae_m:.3f}          {f1_m:.4f}          {sk}")

    print(_row("Persistence (raw, baseline)",
               rmse_pers, 0, pers_metrics["mae"], 0, pers_metrics["f1"], 0, None))
    print(_row("Ridge — raw features",
               ridge_raw_metrics["rmse"], 0,
               ridge_raw_metrics["mae"],  0,
               ridge_raw_metrics["f1"],   0,
               ridge_raw_metrics["skill"]))
    print(_row("Ridge — Kalman-filtered features",
               ridge_filt_metrics["rmse"], 0,
               ridge_filt_metrics["mae"],  0,
               ridge_filt_metrics["f1"],   0,
               ridge_filt_metrics["skill"]))
    print(_row("RF (seeds 42-51) — raw",
               rf_raw_stats["rmse_mean"],  rf_raw_stats["rmse_std"],
               rf_raw_stats["mae_mean"],   rf_raw_stats["mae_std"],
               rf_raw_stats["f1_mean"],    rf_raw_stats["f1_std"],
               rf_raw_stats["skill"]))
    print(_row("RF (seeds 42-51) — Kalman-filtered",
               rf_filt_stats["rmse_mean"], rf_filt_stats["rmse_std"],
               rf_filt_stats["mae_mean"],  rf_filt_stats["mae_std"],
               rf_filt_stats["f1_mean"],   rf_filt_stats["f1_std"],
               rf_filt_stats["skill"]))
    print("=" * 70)

    # Honest assessment
    rf_delta   = rf_filt_stats["rmse_mean"]  - rf_raw_stats["rmse_mean"]
    ridge_delta = ridge_filt_metrics["rmse"] - ridge_raw_metrics["rmse"]
    print()
    print(f"  RF   RMSE delta (filtered - raw): {rf_delta:+.4f} ug/m3  "
          f"({'better' if rf_delta < 0 else 'worse or no change'})")
    print(f"  Ridge RMSE delta:                 {ridge_delta:+.4f} ug/m3  "
          f"({'better' if ridge_delta < 0 else 'worse or no change'})")
    print()

    # ------------------------------------------------------------------
    # 10. Save JSON + CSV
    # ------------------------------------------------------------------
    results = {
        "kalman_params": {"Q": Q, "R": R, "q_frac": q_result["best_q_frac"]},
        "persistence":   pers_metrics,
        "ridge_raw":     ridge_raw_metrics,
        "ridge_filtered":ridge_filt_metrics,
        "rf_raw":        rf_raw_stats,
        "rf_filtered":   rf_filt_stats,
        "rf_delta_rmse_filtered_minus_raw": round(rf_delta,    4),
        "ridge_delta_rmse":                 round(ridge_delta, 4),
    }
    with open(OUTPUTS_RESULTS / "phase4_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    rows = [
        {"model": "Persistence",           "input": "raw",
         "rmse": pers_metrics["rmse"],    "rmse_std": 0,
         "mae":  pers_metrics["mae"],     "mae_std":  0,
         "f1":   pers_metrics["f1"],      "f1_std":   0,  "skill": None},
        {"model": "Ridge",                 "input": "raw",
         "rmse": ridge_raw_metrics["rmse"], "rmse_std": 0,
         "mae":  ridge_raw_metrics["mae"],  "mae_std":  0,
         "f1":   ridge_raw_metrics["f1"],   "f1_std":   0,
         "skill": ridge_raw_metrics["skill"]},
        {"model": "Ridge",                 "input": "kalman_filtered",
         "rmse": ridge_filt_metrics["rmse"], "rmse_std": 0,
         "mae":  ridge_filt_metrics["mae"],  "mae_std":  0,
         "f1":   ridge_filt_metrics["f1"],   "f1_std":   0,
         "skill": ridge_filt_metrics["skill"]},
        {"model": "RandomForest",          "input": "raw",
         "rmse": rf_raw_stats["rmse_mean"],  "rmse_std": rf_raw_stats["rmse_std"],
         "mae":  rf_raw_stats["mae_mean"],   "mae_std":  rf_raw_stats["mae_std"],
         "f1":   rf_raw_stats["f1_mean"],    "f1_std":   rf_raw_stats["f1_std"],
         "skill": rf_raw_stats["skill"]},
        {"model": "RandomForest",          "input": "kalman_filtered",
         "rmse": rf_filt_stats["rmse_mean"], "rmse_std": rf_filt_stats["rmse_std"],
         "mae":  rf_filt_stats["mae_mean"],  "mae_std":  rf_filt_stats["mae_std"],
         "f1":   rf_filt_stats["f1_mean"],   "f1_std":   rf_filt_stats["f1_std"],
         "skill": rf_filt_stats["skill"]},
    ]
    pd.DataFrame(rows).to_csv(OUTPUTS_RESULTS / "phase4_results.csv", index=False)
    print(f"  Saved phase4_results.json  phase4_results.csv")

    # ------------------------------------------------------------------
    # 11. Plot
    # ------------------------------------------------------------------
    print("  Generating plot ...")
    plot_filtered_vs_raw(df, filtered_series)
    print()
    print("Phase 4 complete.")


if __name__ == "__main__":
    run()