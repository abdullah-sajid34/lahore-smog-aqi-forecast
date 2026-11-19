"""
tuning.py -- Hyperparameter tuning for Ridge and RandomForest forecasters.

RULES (enforced by code, not just comments):
  1. Only train + val data are used.  Test rows are never passed in here.
  2. TimeSeriesSplit is used (no shuffling) so each fold's validation set
     is strictly after its training set -- exactly the production scenario.
  3. We use RMSE as the tuning criterion (same as the final evaluation metric).

WHY TimeSeriesSplit instead of KFold:
  KFold with shuffling would put future observations into the training fold,
  leaking lag and rolling features.  TimeSeriesSplit mimics the real
  walk-forward evaluation: train on past, predict next window.

WHY 5 splits:
  5 splits on ~1600 rows (train+val) gives ~320-row validation windows,
  each containing several weeks of data -- enough to estimate seasonal bias.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODEL_FEATURES,
    N_CV_SPLITS,
    OUTPUTS_RESULTS,
    RANDOM_SEED,
    RF_PARAM_GRID,
    RIDGE_ALPHA_GRID,
)
from src.forecasters import (
    RandomForestForecaster,
    RidgeForecaster,
    build_model_features,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _cv_rmse(
    forecaster_cls,
    init_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_CV_SPLITS,
) -> float:
    """
    Walk-forward cross-validation RMSE.

    Uses integer positional splits (TimeSeriesSplit) on the DataFrame that
    has already been ordered chronologically.  Each fold trains on rows
    0..split_point-1 and validates on rows split_point..split_point+gap-1.

    WHY we pass X and y as positional arrays to tss.split():
      TimeSeriesSplit.split() only uses the length of its argument; it does
      not look at index values.  Since X is already sorted by date, this is
      equivalent to a date-ordered split.
    """
    tss    = TimeSeriesSplit(n_splits=n_splits)
    rmses  = []
    X_arr  = X.values
    y_arr  = y.values

    for train_idx, val_idx in tss.split(X_arr):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_vl, y_vl = X.iloc[val_idx],   y.iloc[val_idx]

        model = forecaster_cls(**init_kwargs)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_vl)
        rmses.append(_rmse(y_vl.values, preds))

    return float(np.mean(rmses))


def tune_ridge(trainval_df: pd.DataFrame, horizon: int = 1) -> dict:
    """
    Grid-search Ridge alpha on train+val with TimeSeriesSplit.

    Returns dict with best_alpha and all cv_scores.
    """
    print(f"  Tuning Ridge (horizon={horizon}) ...")
    feat_df = build_model_features(trainval_df, horizon=horizon)
    X = feat_df[MODEL_FEATURES]
    y = feat_df["target"]

    best_alpha = None
    best_rmse  = float("inf")
    scores     = {}

    for alpha in RIDGE_ALPHA_GRID:
        cv = _cv_rmse(
            RidgeForecaster,
            {"alpha": alpha, "horizon": horizon},
            X, y,
        )
        scores[alpha] = round(cv, 4)
        print(f"    alpha={alpha:7.2f}  CV-RMSE={cv:.4f}")
        if cv < best_rmse:
            best_rmse  = cv
            best_alpha = alpha

    print(f"  => best alpha={best_alpha}  CV-RMSE={best_rmse:.4f}")
    return {"best_alpha": best_alpha, "cv_rmse": round(best_rmse, 4), "all_scores": scores}


def tune_rf(trainval_df: pd.DataFrame, horizon: int = 1) -> dict:
    """
    Grid-search RF (n_estimators, max_depth, min_samples_leaf) on
    train+val with TimeSeriesSplit.

    WHY small grid: RF is expensive; 3x3x3 = 27 combinations x 5 folds
    = 135 fits.  Each fit on ~900 rows with 200 trees takes ~1-2 s, so
    total ~3-5 min -- acceptable.  For a production system we would use
    RandomizedSearchCV with more candidates.
    """
    print(f"  Tuning RandomForest (horizon={horizon}) ...")
    feat_df = build_model_features(trainval_df, horizon=horizon)
    X = feat_df[MODEL_FEATURES]
    y = feat_df["target"]

    best_params = None
    best_rmse   = float("inf")
    all_scores  = []

    n_estimators_grid    = RF_PARAM_GRID["n_estimators"]
    max_depth_grid       = RF_PARAM_GRID["max_depth"]
    min_samples_leaf_grid = RF_PARAM_GRID["min_samples_leaf"]

    for n_est in n_estimators_grid:
        for max_d in max_depth_grid:
            for min_sl in min_samples_leaf_grid:
                cv = _cv_rmse(
                    RandomForestForecaster,
                    {
                        "n_estimators":     n_est,
                        "max_depth":        max_d,
                        "min_samples_leaf": min_sl,
                        "random_state":     RANDOM_SEED,
                        "horizon":          horizon,
                    },
                    X, y,
                )
                entry = {
                    "n_estimators": n_est,
                    "max_depth":    max_d,
                    "min_samples_leaf": min_sl,
                    "cv_rmse": round(cv, 4),
                }
                all_scores.append(entry)
                if cv < best_rmse:
                    best_rmse   = cv
                    best_params = entry.copy()

    print(f"  => best params: {best_params}")
    return {
        "best_n_estimators":     best_params["n_estimators"],
        "best_max_depth":        best_params["max_depth"],
        "best_min_samples_leaf": best_params["min_samples_leaf"],
        "cv_rmse":               round(best_rmse, 4),
        "all_scores":            all_scores,
    }


def run_tuning(trainval_df: pd.DataFrame) -> dict:
    """
    Run all tuning and save results to outputs/results/best_params.json.

    Only horizons 1 and 3 are tuned (models are re-used at both horizons
    with the same hyper-parameters; tuning per-horizon would increase
    complexity without much benefit given the small dataset).
    """
    print("=" * 60)
    print("HYPERPARAMETER TUNING (train+val only, TimeSeriesSplit k=5)")
    print("=" * 60)

    ridge_h1 = tune_ridge(trainval_df, horizon=1)
    rf_h1    = tune_rf(trainval_df,    horizon=1)

    best = {
        "ridge_h1": ridge_h1,
        "rf_h1":    rf_h1,
        # For h=3 we reuse h=1 params -- the feature set is identical,
        # only the target shifts.  Separate tuning per horizon would be
        # better in production but would triple runtime here.
        "ridge_h3": {"best_alpha": ridge_h1["best_alpha"],
                     "note": "reused h1 params"},
        "rf_h3":    {"best_n_estimators":     rf_h1["best_n_estimators"],
                     "best_max_depth":        rf_h1["best_max_depth"],
                     "best_min_samples_leaf": rf_h1["best_min_samples_leaf"],
                     "note": "reused h1 params"},
    }

    OUTPUTS_RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_RESULTS / "best_params.json"
    with open(out_path, "w") as fh:
        json.dump(best, fh, indent=2, default=str)
    print(f"\nSaved {out_path}")
    return best


if __name__ == "__main__":
    from src.dataset import AirQualityDataset
    from src.config  import MERGED_CSV

    df = pd.read_csv(MERGED_CSV, parse_dates=["date"], index_col="date").sort_index()
    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    trainval = df.iloc[:n_train + n_val].copy()
    run_tuning(trainval)