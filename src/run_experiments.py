"""
run_experiments.py -- Single entry-point that reproduces the full pipeline.

Usage:
    python src/run_experiments.py

Idempotency guarantee:
    Every random number generator is seeded with RANDOM_SEED=42 at the start.
    Re-running produces byte-identical JSON outputs and identical printed numbers.
    The Kalman filter and tree models are deterministic given fixed Q, R, and seed.

Model selection rule:
    The best model is chosen by VALIDATION CV-RMSE (from TimeSeriesSplit on
    train+val), never by test-set performance.  Test set is touched exactly
    once at the very end for final reporting.

Fail-fast on missing data:
    If the raw AQI CSV is absent the script prints a clear message and exits
    with code 1 before doing any computation.

Phases reproduced:
    0. Folder structure (already exists; verified, not recreated)
    1. Dataset: load, clean, features, split
    2. EDA figures and summary
    3. Forecasting models (persistence, MA, Ridge, RF) + tuning
    4. Kalman filter experiment (raw vs filtered inputs)
    5. Consolidated outputs: evaluator charts, JSON exports, final table
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    AQI_BREAKPOINTS,
    AQI_CATEGORY_LABELS,
    AQI_CSV,
    HORIZON_1,
    MERGED_CSV,
    MODEL_FEATURES,
    OUTPUTS_FIGURES,
    OUTPUTS_MODELS,
    OUTPUTS_RESULTS,
    RANDOM_SEED,
    WEATHER_CSV,
)
from src.dataset import AirQualityDataset
from src.eda import run_eda
from src.evaluator import (
    Evaluator,
    compute_all,
    confusion_matrix,
    f1_by_category,
    mae,
    results_table,
    rmse,
    skill_score,
)
from src.forecasters import (
    MovingAverageForecaster,
    PersistenceForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    build_model_features,
    pm25_to_aqi_category,
)
from src.kalman import KalmanFilter1D, estimate_R_from_training, tune_Q
from src.run_phase4 import build_filtered_features
from src.tuning import run_tuning
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    print()
    print("=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def _safe_json(obj):
    """Make obj JSON-serialisable: convert NaN->None, numpy types->Python."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_safe_json(data), fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    t0 = time.time()
    set_seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Fail-fast: check raw data
    # ------------------------------------------------------------------
    if not AQI_CSV.exists():
        print(f"\nERROR: Raw AQI data not found at {AQI_CSV}")
        print("Please place lahore-us-embassy-air-quality.csv in data/raw/ and re-run.")
        sys.exit(1)

    if not WEATHER_CSV.exists():
        print(f"\nERROR: Weather data not found at {WEATHER_CSV}")
        print("Run:  python src/download_weather.py")
        sys.exit(1)

    OUTPUTS_RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Load, clean, feature-engineer, split
    # ------------------------------------------------------------------
    _banner("PHASE 1 — DATA")

    ds = AirQualityDataset()
    ds.load()
    ds.clean()
    ds.add_features()
    ds.save()

    df       = pd.read_csv(MERGED_CSV, parse_dates=["date"], index_col="date").sort_index()
    n        = len(df)
    n_train  = int(n * 0.70)
    n_val    = int(n * 0.15)

    train    = df.iloc[:n_train]
    val      = df.iloc[n_train : n_train + n_val]
    test     = df.iloc[n_train + n_val :]
    trainval = df.iloc[:n_train + n_val]

    print(f"  Train   : {len(train)} rows  "
          f"({train.index.min().date()} to {train.index.max().date()})")
    print(f"  Val     : {len(val)} rows  "
          f"({val.index.min().date()} to {val.index.max().date()})")
    print(f"  Test    : {len(test)} rows  "
          f"({test.index.min().date()} to {test.index.max().date()})")

    # ------------------------------------------------------------------
    # Phase 2: EDA (train split only)
    # ------------------------------------------------------------------
    _banner("PHASE 2 — EDA")
    eda_summary = run_eda(MERGED_CSV)

    # ------------------------------------------------------------------
    # Phase 3: Tuning + model training
    # ------------------------------------------------------------------
    _banner("PHASE 3 — MODELS")

    best_params = run_tuning(trainval)

    ridge_alpha = best_params["ridge_h1"]["best_alpha"]
    rf_n_est    = best_params["rf_h1"]["best_n_estimators"]
    rf_max_d    = best_params["rf_h1"]["best_max_depth"]
    rf_min_sl   = best_params["rf_h1"]["best_min_samples_leaf"]

    # -- Best model chosen by VALIDATION CV-RMSE, never by test --
    # Ridge CV-RMSE: best_params["ridge_h1"]["cv_rmse"]
    # RF    CV-RMSE: best_params["rf_h1"]["cv_rmse"]
    # We pick whichever is lower.
    ridge_cv = best_params["ridge_h1"]["cv_rmse"]
    rf_cv    = best_params["rf_h1"]["cv_rmse"]
    best_model_by_val = "RandomForest" if rf_cv <= ridge_cv else "Ridge"
    print(f"\n  Ridge CV-RMSE: {ridge_cv:.4f}   RF CV-RMSE: {rf_cv:.4f}")
    print(f"  => Best model by validation: {best_model_by_val}")

    feat_trainval_h1 = build_model_features(trainval, horizon=HORIZON_1)
    feat_test_h1     = build_model_features(test,     horizon=HORIZON_1)

    X_tv = feat_trainval_h1[MODEL_FEATURES]
    y_tv = feat_trainval_h1["target"]
    X_te = feat_test_h1[MODEL_FEATURES]
    y_te = feat_test_h1["target"].values

    # Baselines
    pers = PersistenceForecaster(horizon=HORIZON_1)
    pers.fit(X_tv, y_tv)
    pers_preds   = pers.predict(X_te)
    rmse_pers    = rmse(y_te, pers_preds)

    ma7 = MovingAverageForecaster(horizon=HORIZON_1)
    ma7.fit(X_tv, y_tv)
    ma7_preds = ma7.predict(X_te)

    # Ridge
    ridge = RidgeForecaster(alpha=ridge_alpha, horizon=HORIZON_1)
    ridge.fit(X_tv, y_tv)
    ridge_preds = ridge.predict(X_te)

    # RF (seed=42, deterministic)
    set_seed(RANDOM_SEED)
    rf = RandomForestForecaster(
        n_estimators=rf_n_est, max_depth=rf_max_d,
        min_samples_leaf=rf_min_sl, random_state=RANDOM_SEED,
    )
    rf.fit(X_tv, y_tv)
    rf_preds = rf.predict(X_te)

    # RF seed stability (seeds 42-51)
    rf_seed_rmses, rf_seed_maes, rf_seed_f1s = [], [], []
    for seed in range(42, 52):
        m = RandomForestForecaster(
            n_estimators=rf_n_est, max_depth=rf_max_d,
            min_samples_leaf=rf_min_sl, random_state=seed,
        )
        m.fit(X_tv, y_tv)
        p = m.predict(X_te)
        rf_seed_rmses.append(rmse(y_te, p))
        rf_seed_maes.append(mae(y_te, p))
        rf_seed_f1s.append(f1_by_category(y_te, p))

    # Save models
    joblib.dump(ridge, OUTPUTS_MODELS / "ridge_h1.joblib")
    joblib.dump(rf,    OUTPUTS_MODELS / "rf_h1.joblib")

    # Save ridge coefficients and RF importances
    rf_importances = {col: float(v)
                      for col, v in zip(MODEL_FEATURES, rf.feature_importances_)}
    ridge_coef = {col: float(v)
                  for col, v in zip(MODEL_FEATURES, ridge.coef_)}
    _write_json(OUTPUTS_RESULTS / "rf_feature_importances.json",
                {"model": "rf_h1", "importances": rf_importances})
    _write_json(OUTPUTS_RESULTS / "ridge_coefficients.json",
                {"alpha": ridge_alpha, "coefficients_scaled": ridge_coef})

    # ------------------------------------------------------------------
    # Phase 4: Kalman filter experiment
    # ------------------------------------------------------------------
    _banner("PHASE 4 — KALMAN FILTER")

    R_val = estimate_R_from_training(train["pm25_ugm3"])
    q_res = tune_Q(train["pm25_ugm3"], val["pm25_ugm3"], R=R_val,
                   q_fracs=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    Q_val = q_res["best_Q"]
    print(f"  R={R_val:.4f}  Q={Q_val:.4f}  (Q/R={q_res['best_q_frac']})")

    _write_json(OUTPUTS_RESULTS / "kalman_params.json", {
        "Q": Q_val, "R": R_val,
        "best_q_frac": q_res["best_q_frac"],
        "cv_rmse": q_res["cv_rmse"],
        "R_estimation_method": (
            "Variance of residuals from 3-day centred rolling mean "
            "applied to training period only"
        ),
    })

    # Causal filter over full timeline
    kf      = KalmanFilter1D(Q=Q_val, R=R_val)
    filt_v, filt_var = kf.filter_series(df["pm25_ugm3"], return_variance=True)
    filt_s  = pd.Series(filt_v, index=df.index, name="pm25_filtered")

    # Save kalman series
    ks_records = [
        {"date": str(d.date()), "raw": round(float(r), 4),
         "filtered": round(float(filt_v[i]), 4),
         "variance": round(float(filt_var[i]), 4)}
        for i, (d, r) in enumerate(df["pm25_ugm3"].items())
    ]
    _write_json(OUTPUTS_RESULTS / "kalman_series.json",
                {"Q": Q_val, "R": R_val, "series": ks_records})

    # Filtered feature matrices
    feat_tv_filt = build_filtered_features(trainval, filt_s, horizon=HORIZON_1)
    feat_te_filt = build_filtered_features(test,     filt_s, horizon=HORIZON_1)
    y_te_filt    = feat_te_filt["target"].values

    # RF + Kalman (seed=42)
    set_seed(RANDOM_SEED)
    rf_k = RandomForestForecaster(
        n_estimators=rf_n_est, max_depth=rf_max_d,
        min_samples_leaf=rf_min_sl, random_state=RANDOM_SEED,
    )
    rf_k.fit(feat_tv_filt[MODEL_FEATURES], feat_tv_filt["target"])
    rf_k_preds = rf_k.predict(feat_te_filt[MODEL_FEATURES])
    joblib.dump(rf_k, OUTPUTS_MODELS / "rf_filtered_h1.joblib")

    # RF+Kalman seed stability (seeds 42-51)
    rfk_seed_rmses, rfk_seed_maes, rfk_seed_f1s = [], [], []
    for seed in range(42, 52):
        m = RandomForestForecaster(
            n_estimators=rf_n_est, max_depth=rf_max_d,
            min_samples_leaf=rf_min_sl, random_state=seed,
        )
        m.fit(feat_tv_filt[MODEL_FEATURES], feat_tv_filt["target"])
        p = m.predict(feat_te_filt[MODEL_FEATURES])
        rfk_seed_rmses.append(rmse(y_te_filt[:len(p)], p))
        rfk_seed_maes.append(mae(y_te_filt[:len(p)], p))
        rfk_seed_f1s.append(f1_by_category(y_te_filt[:len(p)], p))

    # Ridge + Kalman
    ridge_k = RidgeForecaster(alpha=ridge_alpha, horizon=HORIZON_1)
    ridge_k.fit(feat_tv_filt[MODEL_FEATURES], feat_tv_filt["target"])
    ridge_k_preds = ridge_k.predict(feat_te_filt[MODEL_FEATURES])
    joblib.dump(ridge_k, OUTPUTS_MODELS / "ridge_filtered_h1.joblib")

    # ------------------------------------------------------------------
    # Phase 5: Consolidated evaluation
    # ------------------------------------------------------------------
    _banner("PHASE 5 — EVALUATION & OUTPUTS")

    # Align predictions on common test dates (intersection of h1 and filtered)
    test_dates_raw  = feat_test_h1.index
    test_dates_filt = feat_te_filt.index

    # Evaluator is built on the raw h1 test set
    ev = Evaluator(
        y_true       = y_te,
        dates        = test_dates_raw,
        rmse_baseline= rmse_pers,
        figures_dir  = OUTPUTS_FIGURES,
    )

    # Metric dicts for each model
    met_pers  = compute_all(y_te, pers_preds)
    met_ma7   = compute_all(y_te, ma7_preds,  rmse_pers)
    met_ridge = compute_all(y_te, ridge_preds, rmse_pers)
    met_rf    = compute_all(y_te, rf_preds,    rmse_pers)

    # For RF+Kalman align on common dates
    met_rfk = compute_all(y_te_filt, rf_k_preds, rmse_pers)

    rf_mean_rmse  = float(np.mean(rf_seed_rmses))
    rf_mean_mae   = float(np.mean(rf_seed_maes))
    rf_mean_f1    = float(np.mean(rf_seed_f1s))
    rfk_mean_rmse = float(np.mean(rfk_seed_rmses))
    rfk_mean_mae  = float(np.mean(rfk_seed_maes))
    rfk_mean_f1   = float(np.mean(rfk_seed_f1s))

    # results_table rows — all five required models
    table_rows = [
        {"model": "Persistence",
         "rmse": met_pers["rmse"], "rmse_std": 0.0,
         "mae":  met_pers["mae"],  "mae_std":  0.0,
         "f1":   met_pers["f1"],   "f1_std":   0.0,
         "skill": None},
        {"model": "MovingAverage7",
         "rmse": met_ma7["rmse"], "rmse_std": 0.0,
         "mae":  met_ma7["mae"],  "mae_std":  0.0,
         "f1":   met_ma7["f1"],   "f1_std":   0.0,
         "skill": met_ma7["skill"]},
        {"model": "Ridge",
         "rmse": met_ridge["rmse"], "rmse_std": 0.0,
         "mae":  met_ridge["mae"],  "mae_std":  0.0,
         "f1":   met_ridge["f1"],   "f1_std":   0.0,
         "skill": met_ridge["skill"]},
        {"model": "RandomForest",
         "rmse": round(rf_mean_rmse, 4),
         "rmse_std": round(float(np.std(rf_seed_rmses)), 4),
         "mae":  round(rf_mean_mae,  4),
         "mae_std": round(float(np.std(rf_seed_maes)),  4),
         "f1":   round(rf_mean_f1,   4),
         "f1_std": round(float(np.std(rf_seed_f1s)),   4),
         "skill": round(skill_score(rf_mean_rmse, rmse_pers), 4)},
        {"model": "RandomForest+Kalman",
         "rmse": round(rfk_mean_rmse, 4),
         "rmse_std": round(float(np.std(rfk_seed_rmses)), 4),
         "mae":  round(rfk_mean_mae,  4),
         "mae_std": round(float(np.std(rfk_seed_maes)),  4),
         "f1":   round(rfk_mean_f1,   4),
         "f1_std": round(float(np.std(rfk_seed_f1s)),   4),
         "skill": round(skill_score(rfk_mean_rmse, rmse_pers), 4)},
    ]
    tbl = results_table(table_rows)

    print("\n")
    print("FINAL RESULTS TABLE — test set (Apr 2024 – Feb 2025)")
    print(tbl.to_string())

    # ------------------------------------------------------------------
    # Charts (Phase 5)
    # ------------------------------------------------------------------
    print("\n  Generating charts ...")

    # 1. Monthly trend (already from EDA -- re-use)
    # (already saved as 01_monthly_trend_and_boxplot.png by run_eda)

    # 2. Predicted vs actual
    preds_dict_aligned = {
        "Persistence": pers_preds,
        "MovingAvg-7": ma7_preds,
        "Ridge":       ridge_preds,
        "RF (best)":   rf_preds,
    }
    ev.plot_predicted_vs_actual(
        preds_dict_aligned,
        highlight_models=["RF (best)", "Ridge"],
        filename="predicted_vs_actual.png",
    )
    print("    Saved predicted_vs_actual.png")

    # 3. RMSE/MAE grouped bars
    bar_metrics = {
        "Persistence": {"rmse": met_pers["rmse"], "mae": met_pers["mae"]},
        "MovingAvg-7": {"rmse": met_ma7["rmse"],  "mae": met_ma7["mae"]},
        "Ridge":       {"rmse": met_ridge["rmse"],"mae": met_ridge["mae"]},
        "RF":          {"rmse": rf_mean_rmse,       "mae": rf_mean_mae},
        "RF+Kalman":   {"rmse": rfk_mean_rmse,      "mae": rfk_mean_mae},
    }
    ev.plot_rmse_mae_bars(bar_metrics, filename="rmse_mae_bars.png")
    print("    Saved rmse_mae_bars.png")

    # 4. Kalman filtered vs raw (already saved by phase 4 inline; regenerate)
    from src.run_phase4 import plot_filtered_vs_raw
    plot_filtered_vs_raw(df, filt_s)
    print("    Saved kalman_filtered_vs_raw.png")

    # 5. Confusion matrix — best model by val (RF)
    ev.plot_confusion_matrix(rf_preds, model_name="RandomForest",
                             filename="confusion_matrix.png")
    print("    Saved confusion_matrix.png")

    # 6. Feature importances
    ev.plot_feature_importances(rf_importances, filename="feature_importances.png")
    print("    Saved feature_importances.png")

    # 7. Skill scores
    skill_dict = {
        "MovingAvg-7": {"skill": met_ma7["skill"]},
        "Ridge":       {"skill": met_ridge["skill"]},
        "RF":          {"skill": round(skill_score(rf_mean_rmse, rmse_pers), 4)},
        "RF+Kalman":   {"skill": round(skill_score(rfk_mean_rmse, rmse_pers), 4)},
    }
    ev.plot_skill_scores(skill_dict, filename="skill_scores.png")
    print("    Saved skill_scores.png")

    # 8. RF seed spread
    ev.plot_rf_seed_spread(
        seed_rmses=rf_seed_rmses,
        seed_maes =rf_seed_maes,
        seeds     =list(range(42, 52)),
        filename  ="rf_seed_spread.png",
    )
    print("    Saved rf_seed_spread.png")

    # ------------------------------------------------------------------
    # JSON exports
    # ------------------------------------------------------------------
    print("\n  Exporting JSON files ...")

    # model_comparison.json
    mc_rows = []
    for row in table_rows:
        mc_rows.append({
            "model":    row["model"],
            "rmse":     row["rmse"],
            "rmse_std": row["rmse_std"],
            "mae":      row["mae"],
            "mae_std":  row["mae_std"],
            "f1":       row["f1"],
            "f1_std":   row["f1_std"],
            "skill":    row["skill"],
        })
    _write_json(OUTPUTS_RESULTS / "model_comparison.json", mc_rows)
    print("    model_comparison.json")

    # predictions_test.json
    pred_rows_json = []
    for i, d in enumerate(test_dates_raw):
        row_j = {
            "date":        d.strftime("%Y-%m-%d"),
            "actual":      round(float(y_te[i]), 4),
            "persistence": round(float(pers_preds[i]),  4),
            "moving_avg7": round(float(ma7_preds[i]),   4),
            "ridge":       round(float(ridge_preds[i]), 4),
            "rf":          round(float(rf_preds[i]),    4),
        }
        # RF+Kalman has potentially different set of dates
        date_str = d.strftime("%Y-%m-%d")
        if d in test_dates_filt:
            idx_k = list(test_dates_filt).index(d)
            row_j["rf_kalman"] = round(float(rf_k_preds[idx_k]), 4)
        else:
            row_j["rf_kalman"] = None
        pred_rows_json.append(row_j)
    _write_json(OUTPUTS_RESULTS / "predictions_test.json", pred_rows_json)
    print("    predictions_test.json")

    # eda_summary.json (already written by run_eda; just confirm)
    print("    eda_summary.json (already saved by EDA phase)")

    # kalman_series.json (already written above)
    print("    kalman_series.json (already saved)")

    # feature_importance.json (clean version)
    _write_json(OUTPUTS_RESULTS / "feature_importance.json",
                {"model": "RandomForest_h1", "importances": rf_importances})
    print("    feature_importance.json")

    # confusion_matrix.json
    cm_mat, cm_labels = confusion_matrix(y_te, rf_preds)
    cm_data = {
        "model":  "RandomForest",
        "labels": cm_labels,
        "matrix": cm_mat.tolist(),
    }
    _write_json(OUTPUTS_RESULTS / "confusion_matrix.json", cm_data)
    print("    confusion_matrix.json")

    # rf_seeds.json
    rf_seeds_data = {
        "model":     "RandomForest_h1",
        "seeds":     list(range(42, 52)),
        "rmse_mean": round(rf_mean_rmse, 4),
        "rmse_std":  round(float(np.std(rf_seed_rmses)), 4),
        "mae_mean":  round(rf_mean_mae,  4),
        "mae_std":   round(float(np.std(rf_seed_maes)),  4),
        "f1_mean":   round(rf_mean_f1,   4),
        "f1_std":    round(float(np.std(rf_seed_f1s)),   4),
        "per_seed":  [
            {"seed": 42+i, "rmse": round(rf_seed_rmses[i], 4),
             "mae": round(rf_seed_maes[i], 4), "f1": round(rf_seed_f1s[i], 4)}
            for i in range(10)
        ],
    }
    _write_json(OUTPUTS_RESULTS / "rf_seeds.json", rf_seeds_data)
    print("    rf_seeds.json")

    # kpis.json
    annual_means = df.groupby(df.index.year)["pm25_ugm3"].mean().round(2)
    peak_idx     = df["pm25_ugm3"].idxmax()
    kpis = {
        "annual_mean_pm25_ugm3":    {str(yr): float(v) for yr, v in annual_means.items()},
        "peak_pm25_ugm3":           round(float(df["pm25_ugm3"].max()), 2),
        "peak_date":                peak_idx.strftime("%Y-%m-%d"),
        "best_model_by_val":        best_model_by_val,
        "best_model_rmse_test":     met_rf["rmse"],
        "best_model_skill_test":    met_rf["skill"],
        "persistence_rmse_test":    round(rmse_pers, 4),
        "data_date_range_start":    df.index.min().strftime("%Y-%m-%d"),
        "data_date_range_end":      df.index.max().strftime("%Y-%m-%d"),
        "total_rows":               int(len(df)),
        "train_rows":               int(n_train),
        "val_rows":                 int(n_val),
        "test_rows":                int(len(test)),
    }
    _write_json(OUTPUTS_RESULTS / "kpis.json", kpis)
    print("    kpis.json")

    # phase3_results.json (consolidated)
    p3_res = {
        "split": {
            "train": f"{train.index.min().date()} to {train.index.max().date()} ({len(train)} rows)",
            "val":   f"{val.index.min().date()} to {val.index.max().date()} ({len(val)} rows)",
            "test":  f"{test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)",
        },
        "best_params": {"ridge_alpha": ridge_alpha, "rf_n_estimators": rf_n_est,
                        "rf_max_depth": rf_max_d,   "rf_min_samples_leaf": rf_min_sl},
        "h1": {
            "Persistence":    met_pers,
            "MovingAverage7": met_ma7,
            "Ridge":          met_ridge,
            "RandomForest":   met_rf,
            "RF_stability": {
                "rmse_mean": round(rf_mean_rmse, 4),
                "rmse_std":  round(float(np.std(rf_seed_rmses)), 4),
                "mae_mean":  round(rf_mean_mae, 4),
                "mae_std":   round(float(np.std(rf_seed_maes)), 4),
                "f1_mean":   round(rf_mean_f1, 4),
                "f1_std":    round(float(np.std(rf_seed_f1s)), 4),
                "seeds":     list(range(42, 52)),
            },
        },
    }
    _write_json(OUTPUTS_RESULTS / "phase3_results.json", p3_res)

    phase4_res = {
        "kalman_params": {"Q": Q_val, "R": R_val, "q_frac": q_res["best_q_frac"]},
        "persistence":    met_pers,
        "ridge_raw":      met_ridge,
        "ridge_filtered": compute_all(y_te_filt, ridge_k_preds, rmse_pers),
        "rf_raw":         {
            "rmse_mean": round(rf_mean_rmse, 4),
            "rmse_std":  round(float(np.std(rf_seed_rmses)), 4),
            "mae_mean":  round(rf_mean_mae, 4), "mae_std": round(float(np.std(rf_seed_maes)),4),
            "f1_mean":   round(rf_mean_f1, 4),  "f1_std":  round(float(np.std(rf_seed_f1s)), 4),
            "skill":     round(skill_score(rf_mean_rmse, rmse_pers), 4),
        },
        "rf_filtered":    {
            "rmse_mean": round(rfk_mean_rmse, 4),
            "rmse_std":  round(float(np.std(rfk_seed_rmses)), 4),
            "mae_mean":  round(rfk_mean_mae, 4), "mae_std": round(float(np.std(rfk_seed_maes)),4),
            "f1_mean":   round(rfk_mean_f1, 4),  "f1_std":  round(float(np.std(rfk_seed_f1s)), 4),
            "skill":     round(skill_score(rfk_mean_rmse, rmse_pers), 4),
        },
    }
    _write_json(OUTPUTS_RESULTS / "phase4_results.json", phase4_res)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    _banner("DONE")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"\n  Figures saved to: {OUTPUTS_FIGURES}")
    print(f"  Results saved to: {OUTPUTS_RESULTS}")
    print(f"  Models  saved to: {OUTPUTS_MODELS}")
    print()
    print("FINAL RESULTS TABLE")
    print(tbl.to_string())


if __name__ == "__main__":
    run()