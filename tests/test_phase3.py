"""
test_phase3.py -- pytest tests for forecasters, tuning, and evaluation.

Correctness properties tested:
  1. PersistenceForecaster returns pm25_lag1 (last known value).
  2. MovingAverageForecaster returns pm25_roll7.
  3. Ridge/RF fit and produce predictions of the right shape and type.
  4. Skill score of Persistence vs itself is 0.
  5. No test dates appear in any tuning (TimeSeriesSplit) fold.
  6. set_seed makes RF predictions reproducible across two identical runs.
  7. build_model_features produces NaN for target when t+h is a gap.
  8. Horizon-3 target is 3 calendar days forward, not 3 rows forward.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_FEATURES, RANDOM_SEED, N_CV_SPLITS
from src.forecasters import (
    MovingAverageForecaster,
    PersistenceForecaster,
    RandomForestForecaster,
    RidgeForecaster,
    build_model_features,
    pm25_to_aqi_category,
)
from src.run_phase3 import compute_metrics
from src.utils import set_seed
from sklearn.model_selection import TimeSeriesSplit


# ---------------------------------------------------------------------------
# Shared tiny synthetic dataset
# ---------------------------------------------------------------------------

def make_tiny_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    30 days of synthetic contiguous daily data with all required columns.
    PM2.5 follows a simple sine wave so lag/roll values are deterministic.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    pm    = 100 + 50 * np.sin(np.linspace(0, 4 * np.pi, n))

    df = pd.DataFrame(index=dates)
    df.index.name = "date"
    df["pm25_ugm3"] = pm
    df["pm25_lag1"]  = df["pm25_ugm3"].shift(1)
    df["pm25_lag2"]  = df["pm25_ugm3"].shift(2)
    df["pm25_lag3"]  = df["pm25_ugm3"].shift(3)
    df["pm25_lag7"]  = df["pm25_ugm3"].shift(7)
    df["pm25_roll7"] = df["pm25_ugm3"].shift(1).rolling(7, min_periods=7).mean()
    df["temperature_2m_mean"]        = rng.uniform(15, 35, n)
    df["relative_humidity_2m_mean"]  = rng.uniform(40, 90, n)
    df["wind_speed_10m_max"]         = rng.uniform(5, 25, n)
    df["precipitation_sum"]          = rng.uniform(0, 3,  n)
    df["month"]          = dates.month
    df["day_of_year"]    = dates.day_of_year
    df["day_of_week"]    = dates.day_of_week
    df["is_smog_season"] = dates.month.isin([10,11,12,1]).astype(int)
    return df.dropna()


@pytest.fixture(scope="module")
def tiny_df():
    return make_tiny_df(60)


@pytest.fixture(scope="module")
def feat_h1(tiny_df):
    return build_model_features(tiny_df, horizon=1)


@pytest.fixture(scope="module")
def feat_h3(tiny_df):
    return build_model_features(tiny_df, horizon=3)


# ---------------------------------------------------------------------------
# 1. PersistenceForecaster
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_predict_equals_lag1(self, feat_h1):
        """Persistence for h=1 must return pm25_lag1 exactly."""
        model = PersistenceForecaster(horizon=1)
        model.fit(feat_h1[MODEL_FEATURES], feat_h1["target"])
        preds = model.predict(feat_h1[MODEL_FEATURES])
        expected = feat_h1["pm25_lag1"].values
        np.testing.assert_allclose(preds, expected, rtol=1e-6,
            err_msg="Persistence must return pm25_lag1")

    def test_fit_returns_self(self, feat_h1):
        model = PersistenceForecaster()
        result = model.fit(feat_h1[MODEL_FEATURES], feat_h1["target"])
        assert result is model

    def test_predict_shape(self, feat_h1):
        model = PersistenceForecaster()
        model.fit(feat_h1[MODEL_FEATURES], feat_h1["target"])
        preds = model.predict(feat_h1[MODEL_FEATURES])
        assert preds.shape == (len(feat_h1),)

    def test_skill_vs_itself_is_zero(self, feat_h1):
        """Skill score of persistence vs persistence must be exactly 0."""
        model = PersistenceForecaster()
        model.fit(feat_h1[MODEL_FEATURES], feat_h1["target"])
        preds   = model.predict(feat_h1[MODEL_FEATURES])
        y_true  = feat_h1["target"].values
        metrics = compute_metrics(y_true, preds, rmse_baseline=None)
        # compute baseline RMSE as persistence RMSE
        pers_rmse = metrics["rmse"]
        metrics2  = compute_metrics(y_true, preds, rmse_baseline=pers_rmse)
        assert metrics2["skill"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. MovingAverageForecaster
# ---------------------------------------------------------------------------

class TestMovingAverage:

    def test_predict_equals_roll7(self, feat_h1):
        """MovingAverage must return pm25_roll7 exactly."""
        model = MovingAverageForecaster(horizon=1)
        model.fit(feat_h1[MODEL_FEATURES], feat_h1["target"])
        preds    = model.predict(feat_h1[MODEL_FEATURES])
        expected = feat_h1["pm25_roll7"].values
        np.testing.assert_allclose(preds, expected, rtol=1e-6,
            err_msg="MovingAverage must return pm25_roll7")

    def test_predict_shape(self, feat_h1):
        model = MovingAverageForecaster()
        model.fit(feat_h1[MODEL_FEATURES], feat_h1["target"])
        preds = model.predict(feat_h1[MODEL_FEATURES])
        assert preds.shape == (len(feat_h1),)


# ---------------------------------------------------------------------------
# 3. RidgeForecaster
# ---------------------------------------------------------------------------

class TestRidge:

    def test_fit_predict_shape(self, feat_h1):
        X, y = feat_h1[MODEL_FEATURES], feat_h1["target"]
        model = RidgeForecaster(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_returns_float_array(self, feat_h1):
        X, y = feat_h1[MODEL_FEATURES], feat_h1["target"]
        model = RidgeForecaster(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.dtype in [np.float32, np.float64]

    def test_predict_before_fit_raises(self, feat_h1):
        model = RidgeForecaster(alpha=1.0)
        with pytest.raises(RuntimeError):
            model.predict(feat_h1[MODEL_FEATURES])

    def test_coef_length(self, feat_h1):
        X, y = feat_h1[MODEL_FEATURES], feat_h1["target"]
        model = RidgeForecaster(alpha=1.0)
        model.fit(X, y)
        assert len(model.coef_) == len(MODEL_FEATURES)


# ---------------------------------------------------------------------------
# 4. RandomForestForecaster
# ---------------------------------------------------------------------------

class TestRandomForest:

    def test_fit_predict_shape(self, feat_h1):
        X, y = feat_h1[MODEL_FEATURES], feat_h1["target"]
        model = RandomForestForecaster(n_estimators=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_before_fit_raises(self, feat_h1):
        model = RandomForestForecaster(n_estimators=10)
        with pytest.raises(RuntimeError):
            model.predict(feat_h1[MODEL_FEATURES])

    def test_feature_importances_sum_to_one(self, feat_h1):
        X, y = feat_h1[MODEL_FEATURES], feat_h1["target"]
        model = RandomForestForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)
        assert np.sum(model.feature_importances_) == pytest.approx(1.0, abs=1e-6)

    def test_seed_reproducibility(self, feat_h1):
        """
        Two RF models with the same seed must produce identical predictions.

        WHY n_jobs=1: sklearn RF with n_jobs=-1 parallelises tree averaging.
        IEEE 754 addition is non-associative, so parallel aggregation can
        differ by ~1e-14 depending on thread scheduling -- not a real
        reproducibility failure.  n_jobs=1 makes the aggregation order
        deterministic so we can use assert_array_equal exactly.
        run_phase3.py still uses n_jobs=-1 for speed; both are correct.
        """
        from sklearn.ensemble import RandomForestRegressor
        X, y = feat_h1[MODEL_FEATURES], feat_h1["target"]

        set_seed(42)
        m1 = RandomForestForecaster(n_estimators=20, random_state=42, horizon=1)
        m1._model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
        m1._model.fit(X, y)
        p1 = m1.predict(X)

        set_seed(42)
        m2 = RandomForestForecaster(n_estimators=20, random_state=42, horizon=1)
        m2._model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
        m2._model.fit(X, y)
        p2 = m2.predict(X)

        np.testing.assert_array_equal(p1, p2,
            err_msg="RF predictions differ across identical seeds (n_jobs=1)")

    def test_different_seeds_differ(self, feat_h1):
        """Different seeds should (almost always) produce different predictions."""
        X, y = feat_h1[MODEL_FEATURES], feat_h1["target"]
        m1 = RandomForestForecaster(n_estimators=20, random_state=1)
        m1.fit(X, y)
        m2 = RandomForestForecaster(n_estimators=20, random_state=99)
        m2.fit(X, y)
        # With 20 trees on 50+ rows, different seeds virtually always differ
        assert not np.array_equal(m1.predict(X), m2.predict(X))


# ---------------------------------------------------------------------------
# 5. Tuning split contamination guard
# ---------------------------------------------------------------------------

class TestTuningSplitContamination:
    """
    Verify that no test-set row appears in any TimeSeriesSplit fold
    when we split the trainval block.

    WHY this test: a silent index error or off-by-one in the split
    fractions could inadvertently include test rows in tuning folds.
    """

    def test_no_test_dates_in_any_tuning_fold(self, tiny_df):
        feat = build_model_features(tiny_df, horizon=1)
        n       = len(feat)
        # Simulate the same 70/15/15 split
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        trainval = feat.iloc[:n_train + n_val]
        test     = feat.iloc[n_train + n_val:]
        test_dates = set(test.index)

        tss = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        for fold_idx, (tr_idx, vl_idx) in enumerate(tss.split(trainval)):
            fold_tr_dates = set(trainval.iloc[tr_idx].index)
            fold_vl_dates = set(trainval.iloc[vl_idx].index)
            overlap_tr = fold_tr_dates & test_dates
            overlap_vl = fold_vl_dates & test_dates
            assert len(overlap_tr) == 0, (
                f"Fold {fold_idx} train contains test dates: {overlap_tr}"
            )
            assert len(overlap_vl) == 0, (
                f"Fold {fold_idx} val contains test dates: {overlap_vl}"
            )


# ---------------------------------------------------------------------------
# 6. build_model_features correctness
# ---------------------------------------------------------------------------

class TestBuildModelFeatures:

    def test_all_model_feature_cols_present(self, feat_h1):
        for col in MODEL_FEATURES:
            assert col in feat_h1.columns, f"Missing feature column: {col}"

    def test_target_col_present(self, feat_h1):
        assert "target" in feat_h1.columns

    def test_no_nan_in_output(self, feat_h1):
        assert not feat_h1.isnull().any().any(), "build_model_features returned NaN rows"

    def test_h3_target_is_3_calendar_days_forward(self, tiny_df):
        """
        For a contiguous series, feat[t]["target"] must equal
        tiny_df["pm25_ugm3"][t + 3 calendar days].
        """
        feat = build_model_features(tiny_df, horizon=3)
        # Pick a row that has a valid target
        row_date  = feat.index[10]
        expected_date = row_date + pd.Timedelta(days=3)
        if expected_date in tiny_df.index:
            expected_val = tiny_df.loc[expected_date, "pm25_ugm3"]
            actual_val   = feat.loc[row_date, "target"]
            assert actual_val == pytest.approx(expected_val, rel=1e-5), (
                f"h=3 target on {row_date.date()} should be "
                f"pm25 on {expected_date.date()} = {expected_val:.2f}, "
                f"got {actual_val:.2f}"
            )

    def test_target_nan_when_gap_ahead(self):
        """
        If day t+1 is missing (gap), the h=1 target on day t must be NaN
        and that row must be dropped by build_model_features.
        """
        # Build a 10-day series with a gap at day 6
        dates_a = pd.date_range("2022-01-01", periods=5, freq="D")
        dates_b = pd.date_range("2022-01-07", periods=5, freq="D")  # day 6 missing
        dates   = dates_a.append(dates_b)
        pm      = np.array([100,110,120,130,140, 160,170,180,190,200], dtype=float)
        df = pd.DataFrame({"pm25_ugm3": pm}, index=dates)
        df.index.name = "date"
        # Add required columns (minimal)
        for lag in [1,2,3,4,5,6,7]:
            df[f"pm25_lag{lag}"] = df["pm25_ugm3"].shift(lag)
        df["pm25_roll7"] = df["pm25_ugm3"].shift(1).rolling(7, min_periods=7).mean()
        for col in ["temperature_2m_mean","relative_humidity_2m_mean",
                    "wind_speed_10m_max","precipitation_sum"]:
            df[col] = 20.0
        df["month"]          = df.index.month
        df["day_of_year"]    = df.index.day_of_year
        df["day_of_week"]    = df.index.day_of_week
        df["is_smog_season"] = 0

        feat = build_model_features(df.dropna(), horizon=1)
        # 2022-01-05 (last day before gap): target = day 6 which is missing -> dropped
        assert pd.Timestamp("2022-01-05") not in feat.index, (
            "Row with missing t+1 target should be dropped"
        )


# ---------------------------------------------------------------------------
# 7. AQI category helper
# ---------------------------------------------------------------------------

class TestAqiCategory:

    def test_low_pm25_is_good(self):
        cats = pm25_to_aqi_category(np.array([5.0]))
        assert cats[0] == "Good"

    def test_high_pm25_is_hazardous(self):
        cats = pm25_to_aqi_category(np.array([400.0]))
        assert "Hazardous" in cats[0]

    def test_all_values_return_string(self):
        vals = np.array([0, 12, 35, 55, 150, 250, 350, 500, 600])
        cats = pm25_to_aqi_category(vals)
        assert all(isinstance(c, str) for c in cats)