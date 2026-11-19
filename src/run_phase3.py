"""
run_phase3.py -- Full Phase 3 evaluation pipeline.

Workflow
--------
1. Load lahore_merged.csv (cleaned, feature-engineered).
2. Split into train / val / test using the same fractions as Phase 1.
3. Tune Ridge and RF on train+val (TimeSeriesSplit, test never touched).
4. Refit each model on train+val, evaluate ONCE on test.
5. RF: repeat with seeds 42-51, report mean +/- std.
6. Horizon-3 bonus: same models, target is PM2.5 at t+3.
7. Save results, predictions, models, and feature weights.

Metric definitions
------------------
RMSE       = sqrt(mean((y_pred - y_true)^2))  -- primary metric
MAE        = mean(|y_pred - y_true|)
Macro F1   = F1 score over AQI category labels (macro averaging treats
             each category equally regardless of class frequency)
Skill score = 1 - RMSE(model) / RMSE(persistence)
             0 means as good as persistence, 1 means perfect, <0 means worse.

WHY macro F1:
  The class distribution is highly imbalanced (many Unhealthy days, few Good).
  Macro averaging prevents the metric from being dominated by the majority class.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    HORIZON_1,
    HORIZON_3,
    MERGED_CSV,
    MODEL_FEATURES,
    OUTPUTS_MODELS,
    OUTPUTS_RESULTS,
    RANDOM_SEED,
)
from src.forecasters import (
    MovingAverageForecaster,
    PersistenceForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    build_model_features,
    pm25_to_aqi_category,
)
from src.tuning import run_tuning
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rmse_baseline: float | None = None,
) -> dict:
    """Compute RMSE, MAE, macro F1, and skill score."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    cats_true = pm25_to_aqi_category(y_true)
    cats_pred = pm25_to_aqi_category(y_pred)
    f1 = float(f1_score(cats_true, cats_pred, average="macro", zero_division=0))

    skill = None
    if rmse_baseline is not None and rmse_baseline > 0:
        skill = round(1.0 - rmse / rmse_baseline, 4)

    return {
        "rmse":  round(rmse,  4),
        "mae":   round(mae,   4),
        "f1":    round(f1,    4),
        "skill": skill,
    }


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    feat_test: pd.DataFrame,
    rmse_baseline: float | None = None,
) -> tuple[dict, np.ndarray]:
    """Predict on feat_test and return (metrics_dict, predictions_array)."""
    X_test = feat_test[MODEL_FEATURES]
    y_test = feat_test["target"].values
    preds  = model.predict(X_test)
    return compute_metrics(y_test, preds, rmse_baseline), preds


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(csv_path: Path = MERGED_CSV) -> None:
    set_seed()
    OUTPUTS_RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and split
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date").sort_index()
    n       = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train    = df.iloc[:n_train].copy()
    val      = df.iloc[n_train : n_train + n_val].copy()
    test     = df.iloc[n_train + n_val :].copy()
    trainval = df.iloc[:n_train + n_val].copy()

    print("=" * 70)
    print("PHASE 3 — FORECASTING MODELS")
    print("=" * 70)
    print(f"  Train    : {len(train)} rows  "
          f"({train.index.min().date()} to {train.index.max().date()})")
    print(f"  Val      : {len(val)} rows  "
          f"({val.index.min().date()} to {val.index.max().date()})")
    print(f"  Test     : {len(test)} rows  "
          f"({test.index.min().date()} to {test.index.max().date()})")
    print()

    # ------------------------------------------------------------------
    # 2. Build feature matrices (calendar-day safe, done once per horizon)
    # ------------------------------------------------------------------
    feat_trainval_h1 = build_model_features(trainval, horizon=HORIZON_1)
    feat_test_h1     = build_model_features(test,     horizon=HORIZON_1)
    feat_trainval_h3 = build_model_features(trainval, horizon=HORIZON_3)
    feat_test_h3     = build_model_features(test,     horizon=HORIZON_3)

    print(f"  Feature rows (trainval h=1): {len(feat_trainval_h1)}")
    print(f"  Feature rows (test     h=1): {len(feat_test_h1)}")
    print(f"  Feature rows (trainval h=3): {len(feat_trainval_h3)}")
    print(f"  Feature rows (test     h=3): {len(feat_test_h3)}")
    print()

    # Verify test set contamination: no test index should appear in trainval
    tv_dates   = set(feat_trainval_h1.index)
    test_dates = set(feat_test_h1.index)
    assert len(tv_dates & test_dates) == 0, "LEAK: test dates found in trainval!"

    # ------------------------------------------------------------------
    # 3. Tune (train+val only)
    # ------------------------------------------------------------------
    best_params = run_tuning(trainval)
    print()

    ridge_alpha = best_params["ridge_h1"]["best_alpha"]
    rf_n_est    = best_params["rf_h1"]["best_n_estimators"]
    rf_max_d    = best_params["rf_h1"]["best_max_depth"]
    rf_min_sl   = best_params["rf_h1"]["best_min_samples_leaf"]

    # ------------------------------------------------------------------
    # 4. Refit on train+val, evaluate on test  (h=1)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("HORIZON 1 (t+1 day ahead)")
    print("=" * 70)

    X_tv1 = feat_trainval_h1[MODEL_FEATURES]
    y_tv1 = feat_trainval_h1["target"]
    X_te1 = feat_test_h1[MODEL_FEATURES]
    y_te1 = feat_test_h1["target"].values

    # Baselines (no fitting)
    pers_h1 = PersistenceForecaster(horizon=HORIZON_1)
    pers_h1.fit(X_tv1, y_tv1)
    pers_preds_h1 = pers_h1.predict(X_te1)
    pers_metrics_h1 = compute_metrics(y_te1, pers_preds_h1)
    rmse_pers_h1 = pers_metrics_h1["rmse"]

    ma_h1 = MovingAverageForecaster(horizon=HORIZON_1)
    ma_h1.fit(X_tv1, y_tv1)
    ma_preds_h1    = ma_h1.predict(X_te1)
    ma_metrics_h1  = compute_metrics(y_te1, ma_preds_h1, rmse_pers_h1)

    # Ridge
    ridge_h1 = RidgeForecaster(alpha=ridge_alpha, horizon=HORIZON_1)
    ridge_h1.fit(X_tv1, y_tv1)
    ridge_preds_h1   = ridge_h1.predict(X_te1)
    ridge_metrics_h1 = compute_metrics(y_te1, ridge_preds_h1, rmse_pers_h1)

    # RF (seed=42, deterministic)
    rf_h1 = RandomForestForecaster(
        n_estimators=rf_n_est, max_depth=rf_max_d,
        min_samples_leaf=rf_min_sl, random_state=RANDOM_SEED, horizon=HORIZON_1,
    )
    rf_h1.fit(X_tv1, y_tv1)
    rf_preds_h1   = rf_h1.predict(X_te1)
    rf_metrics_h1 = compute_metrics(y_te1, rf_preds_h1, rmse_pers_h1)

    # RF seed stability (seeds 42-51)
    rf_seed_rmses, rf_seed_maes, rf_seed_f1s = [], [], []
    for seed in range(42, 52):
        m = RandomForestForecaster(
            n_estimators=rf_n_est, max_depth=rf_max_d,
            min_samples_leaf=rf_min_sl, random_state=seed, horizon=HORIZON_1,
        )
        m.fit(X_tv1, y_tv1)
        p = m.predict(X_te1)
        met = compute_metrics(y_te1, p, rmse_pers_h1)
        rf_seed_rmses.append(met["rmse"])
        rf_seed_maes.append(met["mae"])
        rf_seed_f1s.append(met["f1"])

    rf_stability_h1 = {
        "rmse_mean": round(float(np.mean(rf_seed_rmses)),  4),
        "rmse_std":  round(float(np.std(rf_seed_rmses)),   4),
        "mae_mean":  round(float(np.mean(rf_seed_maes)),   4),
        "mae_std":   round(float(np.std(rf_seed_maes)),    4),
        "f1_mean":   round(float(np.mean(rf_seed_f1s)),    4),
        "f1_std":    round(float(np.std(rf_seed_f1s)),     4),
        "seeds":     list(range(42, 52)),
    }

    # ------------------------------------------------------------------
    # 5. Horizon 3 (t+3 bonus)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("HORIZON 3 (t+3 days ahead)")
    print("=" * 70)

    X_tv3 = feat_trainval_h3[MODEL_FEATURES]
    y_tv3 = feat_trainval_h3["target"]
    X_te3 = feat_test_h3[MODEL_FEATURES]
    y_te3 = feat_test_h3["target"].values

    pers_h3 = PersistenceForecaster(horizon=HORIZON_3)
    pers_h3.fit(X_tv3, y_tv3)
    pers_preds_h3   = pers_h3.predict(X_te3)
    pers_metrics_h3 = compute_metrics(y_te3, pers_preds_h3)
    rmse_pers_h3    = pers_metrics_h3["rmse"]

    ma_h3 = MovingAverageForecaster(horizon=HORIZON_3)
    ma_h3.fit(X_tv3, y_tv3)
    ma_preds_h3   = ma_h3.predict(X_te3)
    ma_metrics_h3 = compute_metrics(y_te3, ma_preds_h3, rmse_pers_h3)

    ridge_h3 = RidgeForecaster(alpha=ridge_alpha, horizon=HORIZON_3)
    ridge_h3.fit(X_tv3, y_tv3)
    ridge_preds_h3   = ridge_h3.predict(X_te3)
    ridge_metrics_h3 = compute_metrics(y_te3, ridge_preds_h3, rmse_pers_h3)

    rf_h3 = RandomForestForecaster(
        n_estimators=rf_n_est, max_depth=rf_max_d,
        min_samples_leaf=rf_min_sl, random_state=RANDOM_SEED, horizon=HORIZON_3,
    )
    rf_h3.fit(X_tv3, y_tv3)
    rf_preds_h3   = rf_h3.predict(X_te3)
    rf_metrics_h3 = compute_metrics(y_te3, rf_preds_h3, rmse_pers_h3)

    # ------------------------------------------------------------------
    # 6. Print results table
    # ------------------------------------------------------------------
    def _row(name, m, extra=""):
        skill_str = f"{m['skill']:+.4f}" if m.get("skill") is not None else "  0.0000"
        return (f"  {name:<30s}  RMSE={m['rmse']:7.3f}  MAE={m['mae']:7.3f}"
                f"  F1={m['f1']:.4f}  Skill={skill_str}  {extra}")

    print()
    print("=" * 70)
    print("RESULTS TABLE — test set only")
    print("=" * 70)
    print("--- Horizon 1 (t+1) ---")
    print(_row("Persistence (baseline)", pers_metrics_h1))
    print(_row("MovingAverage-7",        ma_metrics_h1))
    print(_row("Ridge",                  ridge_metrics_h1))
    print(_row("RandomForest (seed=42)", rf_metrics_h1))
    print(f"  {'RF seed stability (42-51)':<30s}  "
          f"RMSE={rf_stability_h1['rmse_mean']:.3f}+/-{rf_stability_h1['rmse_std']:.3f}  "
          f"MAE={rf_stability_h1['mae_mean']:.3f}+/-{rf_stability_h1['mae_std']:.3f}  "
          f"F1={rf_stability_h1['f1_mean']:.4f}+/-{rf_stability_h1['f1_std']:.4f}")
    print()
    print("--- Horizon 3 (t+3) ---")
    print(_row("Persistence (baseline)", pers_metrics_h3))
    print(_row("MovingAverage-7",        ma_metrics_h3))
    print(_row("Ridge",                  ridge_metrics_h3))
    print(_row("RandomForest (seed=42)", rf_metrics_h3))
    print("=" * 70)

    # ------------------------------------------------------------------
    # 7. Save predictions CSV
    # ------------------------------------------------------------------
    pred_rows = []
    dates_h1  = feat_test_h1.index
    for i, d in enumerate(dates_h1):
        pred_rows.append({
            "date":               str(d.date()),
            "actual_h1":          round(float(y_te1[i]), 4),
            "persistence_h1":     round(float(pers_preds_h1[i]),  4),
            "moving_avg_h1":      round(float(ma_preds_h1[i]),    4),
            "ridge_h1":           round(float(ridge_preds_h1[i]), 4),
            "rf_h1":              round(float(rf_preds_h1[i]),    4),
        })
    pred_df = pd.DataFrame(pred_rows).set_index("date")

    # Add h3 predictions (may have fewer rows if target is NaN near end)
    dates_h3 = feat_test_h3.index
    h3_dict  = {str(d.date()): {
        "actual_h3":      round(float(y_te3[i]), 4),
        "persistence_h3": round(float(pers_preds_h3[i]),  4),
        "moving_avg_h3":  round(float(ma_preds_h3[i]),    4),
        "ridge_h3":       round(float(ridge_preds_h3[i]), 4),
        "rf_h3":          round(float(rf_preds_h3[i]),    4),
    } for i, d in enumerate(dates_h3)}

    for col in ["actual_h3","persistence_h3","moving_avg_h3","ridge_h3","rf_h3"]:
        pred_df[col] = pred_df.index.map(lambda d: h3_dict.get(d, {}).get(col))

    pred_path = OUTPUTS_RESULTS / "predictions_test_phase3.csv"
    pred_df.to_csv(pred_path)
    print(f"\n  Saved {pred_path.name}  ({len(pred_df)} rows)")

    # ------------------------------------------------------------------
    # 8. Save models
    # ------------------------------------------------------------------
    for model, fname in [
        (ridge_h1, "ridge_h1.joblib"),
        (ridge_h3, "ridge_h3.joblib"),
        (rf_h1,    "rf_h1.joblib"),
        (rf_h3,    "rf_h3.joblib"),
    ]:
        p = OUTPUTS_MODELS / fname
        joblib.dump(model, p)
        print(f"  Saved model {fname}")

    # ------------------------------------------------------------------
    # 9. Feature weights
    # ------------------------------------------------------------------
    ridge_coef = {
        col: round(float(v), 6)
        for col, v in zip(MODEL_FEATURES, ridge_h1.coef_)
    }
    coef_path = OUTPUTS_RESULTS / "ridge_coefficients.json"
    with open(coef_path, "w") as fh:
        json.dump({"alpha": ridge_alpha, "coefficients_scaled": ridge_coef}, fh, indent=2)
    print(f"  Saved {coef_path.name}")

    rf_imp = {
        col: round(float(v), 6)
        for col, v in zip(MODEL_FEATURES, rf_h1.feature_importances_)
    }
    imp_path = OUTPUTS_RESULTS / "rf_feature_importances.json"
    with open(imp_path, "w") as fh:
        json.dump({"model": "rf_h1", "importances": rf_imp}, fh, indent=2)
    print(f"  Saved {imp_path.name}")

    # ------------------------------------------------------------------
    # 10. Save full results JSON + CSV
    # ------------------------------------------------------------------
    results = {
        "split": {
            "train": f"{train.index.min().date()} to {train.index.max().date()} ({len(train)} rows)",
            "val":   f"{val.index.min().date()} to {val.index.max().date()} ({len(val)} rows)",
            "test":  f"{test.index.min().date()} to {test.index.max().date()} ({len(test)} rows)",
        },
        "best_params": {
            "ridge_alpha":        ridge_alpha,
            "rf_n_estimators":    rf_n_est,
            "rf_max_depth":       rf_max_d,
            "rf_min_samples_leaf":rf_min_sl,
        },
        "h1": {
            "Persistence":    pers_metrics_h1,
            "MovingAverage7": ma_metrics_h1,
            "Ridge":          ridge_metrics_h1,
            "RandomForest":   rf_metrics_h1,
            "RF_stability":   rf_stability_h1,
        },
        "h3": {
            "Persistence":    pers_metrics_h3,
            "MovingAverage7": ma_metrics_h3,
            "Ridge":          ridge_metrics_h3,
            "RandomForest":   rf_metrics_h3,
        },
    }

    json_path = OUTPUTS_RESULTS / "phase3_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"  Saved {json_path.name}")

    # Flat CSV
    rows_csv = []
    for horizon_key, horizon_int in [("h1", 1), ("h3", 3)]:
        for model_name, met in results[horizon_key].items():
            if model_name == "RF_stability":
                continue
            rows_csv.append({
                "horizon":    horizon_int,
                "model":      model_name,
                "rmse":       met.get("rmse"),
                "mae":        met.get("mae"),
                "f1":         met.get("f1"),
                "skill":      met.get("skill"),
            })
    csv_path = OUTPUTS_RESULTS / "phase3_results.csv"
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
    print(f"  Saved {csv_path.name}")

    print()
    print("Phase 3 complete.")


if __name__ == "__main__":
    run()