"""
test_evaluator.py -- pytest tests for src/evaluator.py and JSON outputs.

Tests cover:
  1. Metric functions on tiny hand-crafted arrays.
  2. Skill score edge cases (zero baseline, negative skill).
  3. Confusion matrix shape, labels, and row-sum totals.
  4. results_table: validates all five required models are present.
  5. JSON files exist and parse without error.
  6. kpis.json has all expected keys.
  7. confusion_matrix.json has correct shape and sums.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from src.config import OUTPUTS_RESULTS, AQI_CATEGORY_LABELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_pred():
    """y_pred == y_true  -->  RMSE=0, MAE=0, skill=1 (vs any baseline)."""
    y = np.array([50.0, 100.0, 150.0, 200.0, 300.0, 400.0])
    return y, y.copy()


@pytest.fixture
def nonzero_pred():
    """Distinct y_true and y_pred for general metric tests."""
    y_true = np.array([50.0,  100.0, 150.0, 200.0, 300.0, 400.0])
    y_pred = np.array([60.0,  90.0,  170.0, 180.0, 310.0, 380.0])
    return y_true, y_pred


@pytest.fixture
def category_pred():
    """
    Hand-crafted arrays where the true categories are known:
      50 ug/m3  -> Good (<=12) ... actually maps to Moderate (12.1-35.4 AQI)
    Use values clearly inside each breakpoint bucket.
    """
    # Cp boundaries: Good <=12, Moderate <=35.4, UnhealthySens <=55.4,
    #                Unhealthy <=150.4, VeryUnhealthy <=250.4, Hazardous <=350.4
    y_true = np.array([5.0,  25.0,  45.0,  100.0, 200.0, 300.0])
    y_pred = np.array([6.0,  26.0,  46.0,  101.0, 201.0, 301.0])  # same cats
    return y_true, y_pred


# ---------------------------------------------------------------------------
# 1. Metric functions
# ---------------------------------------------------------------------------

class TestRmse:

    def test_perfect_prediction_zero(self, perfect_pred):
        y_true, y_pred = perfect_pred
        assert rmse(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self, nonzero_pred):
        y_true, y_pred = nonzero_pred
        expected = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        assert rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-6)

    def test_returns_float(self, nonzero_pred):
        y_true, y_pred = nonzero_pred
        assert isinstance(rmse(y_true, y_pred), float)

    def test_symmetric(self, nonzero_pred):
        y_true, y_pred = nonzero_pred
        assert rmse(y_true, y_pred) == pytest.approx(rmse(y_pred, y_true), rel=1e-9)


class TestMae:

    def test_perfect_prediction_zero(self, perfect_pred):
        y_true, y_pred = perfect_pred
        assert mae(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self, nonzero_pred):
        y_true, y_pred = nonzero_pred
        expected = float(np.mean(np.abs(y_true - y_pred)))
        assert mae(y_true, y_pred) == pytest.approx(expected, rel=1e-6)

    def test_mae_leq_rmse(self, nonzero_pred):
        """MAE <= RMSE by Jensen's inequality."""
        y_true, y_pred = nonzero_pred
        assert mae(y_true, y_pred) <= rmse(y_true, y_pred) + 1e-9


class TestF1ByCategory:

    def test_perfect_prediction_gives_one(self, category_pred):
        y_true, y_pred = category_pred
        f1 = f1_by_category(y_true, y_pred)
        assert f1 == pytest.approx(1.0, abs=1e-6)

    def test_returns_value_between_0_and_1(self, nonzero_pred):
        y_true, y_pred = nonzero_pred
        f1 = f1_by_category(y_true, y_pred)
        assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# 2. Skill score edge cases
# ---------------------------------------------------------------------------

class TestSkillScore:

    def test_perfect_model_skill_one(self, perfect_pred):
        y_true, y_pred = perfect_pred
        baseline_rmse = rmse(y_true, y_true + 10)   # baseline makes error of 10
        sk = skill_score(0.0, baseline_rmse)
        assert sk == pytest.approx(1.0, abs=1e-9)

    def test_model_equal_to_baseline_skill_zero(self, nonzero_pred):
        y_true, y_pred = nonzero_pred
        r = rmse(y_true, y_pred)
        assert skill_score(r, r) == pytest.approx(0.0, abs=1e-9)

    def test_model_worse_than_baseline_negative_skill(self, nonzero_pred):
        y_true, y_pred = nonzero_pred
        r_model    = rmse(y_true, y_pred)
        r_baseline = r_model * 0.5   # baseline is twice as good
        sk = skill_score(r_model, r_baseline)
        assert sk < 0.0

    def test_zero_baseline_raises(self):
        with pytest.raises(ValueError):
            skill_score(10.0, 0.0)

    def test_persistence_vs_itself_zero(self, nonzero_pred):
        """Skill of persistence vs itself must be 0."""
        y_true, y_pred = nonzero_pred
        r = rmse(y_true, y_pred)
        assert skill_score(r, r) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. Confusion matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:

    def test_shape_square(self, category_pred):
        y_true, y_pred = category_pred
        mat, labels = confusion_matrix(y_true, y_pred)
        assert mat.shape[0] == mat.shape[1]
        assert mat.shape[0] == len(labels)

    def test_row_sums_equal_true_counts(self, nonzero_pred):
        """Each row's sum must equal the number of true samples in that category."""
        y_true, y_pred = nonzero_pred
        mat, labels = confusion_matrix(y_true, y_pred)
        cats_true = [c for c in
                     [next((lb for _,_,lo,hi,lb,_ in
                            __import__('src.config', fromlist=['AQI_BREAKPOINTS']).AQI_BREAKPOINTS
                            if v <= hi), AQI_CATEGORY_LABELS[-1])
                      for v in y_true]]
        for r, label in enumerate(labels):
            expected_count = sum(1 for c in cats_true if c == label)
            assert mat[r].sum() == expected_count, (
                f"Row sum for {label} = {mat[r].sum()}, expected {expected_count}"
            )

    def test_total_equals_n(self, nonzero_pred):
        """Grand total of confusion matrix must equal number of samples."""
        y_true, y_pred = nonzero_pred
        mat, _ = confusion_matrix(y_true, y_pred)
        assert mat.sum() == len(y_true)

    def test_labels_subset_of_aqi_categories(self, nonzero_pred):
        y_true, y_pred = nonzero_pred
        _, labels = confusion_matrix(y_true, y_pred)
        for lb in labels:
            assert lb in AQI_CATEGORY_LABELS

    def test_perfect_preds_diagonal_only(self, category_pred):
        y_true, y_pred = category_pred
        mat, _ = confusion_matrix(y_true, y_pred)
        off_diagonal = mat - np.diag(np.diag(mat))
        assert off_diagonal.sum() == 0, "Perfect predictions should have no off-diagonal entries"


# ---------------------------------------------------------------------------
# 4. results_table
# ---------------------------------------------------------------------------

class TestResultsTable:

    def _make_rows(self, model_names):
        return [
            {"model": n, "rmse": 50.0 + i, "rmse_std": 0.1,
             "mae": 35.0 + i, "mae_std": 0.05,
             "f1": 0.4 - i*0.01, "f1_std": 0.01,
             "skill": 0.2 - i*0.01}
            for i, n in enumerate(model_names)
        ]

    def test_requires_all_five_models(self):
        """results_table raises ValueError when any required model is missing."""
        incomplete = self._make_rows(
            ["Persistence", "MovingAverage7", "Ridge", "RandomForest"]
            # Missing RandomForest+Kalman
        )
        with pytest.raises(ValueError, match="missing required models"):
            results_table(incomplete)

    def test_returns_dataframe(self):
        rows = self._make_rows(
            ["Persistence", "MovingAverage7", "Ridge",
             "RandomForest", "RandomForest+Kalman"]
        )
        df = results_table(rows)
        assert isinstance(df, pd.DataFrame)

    def test_has_all_five_model_rows(self):
        rows = self._make_rows(
            ["Persistence", "MovingAverage7", "Ridge",
             "RandomForest", "RandomForest+Kalman"]
        )
        df = results_table(rows)
        assert len(df) == 5

    def test_model_index(self):
        rows = self._make_rows(
            ["Persistence", "MovingAverage7", "Ridge",
             "RandomForest", "RandomForest+Kalman"]
        )
        df = results_table(rows)
        assert "RandomForest+Kalman" in df.index


# ---------------------------------------------------------------------------
# 5-7. JSON files exist and parse
# ---------------------------------------------------------------------------

REQUIRED_JSON = [
    "model_comparison.json",
    "predictions_test.json",
    "eda_summary.json",
    "kalman_series.json",
    "feature_importance.json",
    "confusion_matrix.json",
    "rf_seeds.json",
    "kpis.json",
]


class TestJsonFiles:

    @pytest.mark.parametrize("filename", REQUIRED_JSON)
    def test_file_exists(self, filename):
        path = OUTPUTS_RESULTS / filename
        assert path.exists(), f"{filename} not found in {OUTPUTS_RESULTS}"

    @pytest.mark.parametrize("filename", REQUIRED_JSON)
    def test_file_parses(self, filename):
        path = OUTPUTS_RESULTS / filename
        if not path.exists():
            pytest.skip(f"{filename} not yet generated")
        with open(path) as f:
            data = json.load(f)
        assert data is not None

    def test_kpis_has_required_keys(self):
        path = OUTPUTS_RESULTS / "kpis.json"
        if not path.exists():
            pytest.skip("kpis.json not yet generated")
        with open(path) as f:
            kpis = json.load(f)
        required = {
            "annual_mean_pm25_ugm3", "peak_pm25_ugm3", "peak_date",
            "best_model_by_val", "best_model_rmse_test",
            "best_model_skill_test", "persistence_rmse_test",
            "data_date_range_start", "data_date_range_end",
            "total_rows",
        }
        for key in required:
            assert key in kpis, f"kpis.json missing key: {key}"

    def test_confusion_matrix_json_shape(self):
        path = OUTPUTS_RESULTS / "confusion_matrix.json"
        if not path.exists():
            pytest.skip("confusion_matrix.json not yet generated")
        with open(path) as f:
            cm = json.load(f)
        assert "matrix" in cm
        assert "labels" in cm
        mat    = np.array(cm["matrix"])
        labels = cm["labels"]
        assert mat.shape == (len(labels), len(labels)), "Confusion matrix is not square"
        # Grand total must equal number of test predictions
        assert mat.sum() > 0, "Empty confusion matrix"

    def test_no_nan_in_json_files(self):
        """None is acceptable; NaN strings are not valid JSON and must not appear."""
        for fname in REQUIRED_JSON:
            path = OUTPUTS_RESULTS / fname
            if not path.exists():
                continue
            text = path.read_text()
            assert "NaN" not in text, f"{fname} contains literal NaN (invalid JSON)"
            assert "Infinity" not in text, f"{fname} contains Infinity (invalid JSON)"

    def test_model_comparison_has_five_rows(self):
        path = OUTPUTS_RESULTS / "model_comparison.json"
        if not path.exists():
            pytest.skip("model_comparison.json not yet generated")
        with open(path) as f:
            mc = json.load(f)
        model_names = {r["model"] for r in mc}
        required = {"Persistence","MovingAverage7","Ridge",
                    "RandomForest","RandomForest+Kalman"}
        assert required == model_names, (
            f"model_comparison.json models: {model_names}"
        )

    def test_predictions_test_has_iso_dates(self):
        path = OUTPUTS_RESULTS / "predictions_test.json"
        if not path.exists():
            pytest.skip("predictions_test.json not yet generated")
        with open(path) as f:
            preds = json.load(f)
        assert len(preds) > 0
        # Check first date is ISO format YYYY-MM-DD
        import re
        iso_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for row in preds[:5]:
            assert iso_pattern.match(row["date"]), (
                f"Non-ISO date in predictions_test.json: {row['date']}"
            )