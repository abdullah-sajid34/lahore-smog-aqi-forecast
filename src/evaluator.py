"""
evaluator.py -- Reusable metrics and plots for the Lahore PM2.5 project.

All metric functions accept plain NumPy arrays so they can be used
independently of the forecaster classes.  The Evaluator class bundles
them together with plotting helpers that share the dark-theme style and
AQI colors from config.py.

Design rule: Evaluator never loads or modifies data.  It only receives
arrays / DataFrames from the caller, computes statistics, and optionally
saves figures.  This keeps it testable with tiny synthetic arrays.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AQI_BREAKPOINTS, AQI_CATEGORY_COLORS, AQI_CATEGORY_LABELS
from src.forecasters import pm25_to_aqi_category

# ---------------------------------------------------------------------------
# Dark-theme palette (consistent with EDA figures)
# ---------------------------------------------------------------------------
DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
GRID_COL  = "#2a2a4a"
TEXT_COL  = "#e0e0e0"
ACCENT    = "#00d4ff"
SMOG_COL  = "#ff6b35"
CYAN      = "#22D3EE"
GOLD      = "#ffd166"
GREEN     = "#06d6a0"

MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--")
    if title:   ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9)


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (ug/m3)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error (ug/m3)."""
    return float(mean_absolute_error(y_true, y_pred))


def f1_by_category(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-averaged F1 over US EPA AQI categories.

    WHY macro: imbalanced classes (many Unhealthy, few Good days).
    Macro gives each category equal weight so rare extreme-event categories
    are not drowned out by the common middle categories.
    """
    cats_true = pm25_to_aqi_category(y_true)
    cats_pred = pm25_to_aqi_category(y_pred)
    return float(f1_score(cats_true, cats_pred, average="macro", zero_division=0))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Return (matrix, labels) where labels are the AQI categories
    present in y_true or y_pred, in EPA severity order.

    WHY return labels with the matrix: the category set depends on the data
    (e.g. 'Good' may never appear in winter test data), so the caller must
    know which row/column maps to which category.
    """
    cats_true = pm25_to_aqi_category(y_true)
    cats_pred = pm25_to_aqi_category(y_pred)
    all_cats  = AQI_CATEGORY_LABELS  # fixed EPA order
    present   = [c for c in all_cats
                 if c in set(cats_true) or c in set(cats_pred)]
    mat = sk_confusion_matrix(cats_true, cats_pred, labels=present)
    return mat, present


def skill_score(rmse_model: float, rmse_baseline: float) -> float:
    """
    Skill score = 1 - RMSE(model) / RMSE(baseline).

    0 means as good as the baseline (persistence).
    1 means perfect.
    Negative means worse than persistence.

    WHY this formula: it is the standard meteorological skill score, making
    it easy to compare across datasets with different PM2.5 scale.
    """
    if rmse_baseline <= 0:
        raise ValueError("rmse_baseline must be positive")
    return float(1.0 - rmse_model / rmse_baseline)


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rmse_baseline: float | None = None,
) -> dict:
    """Convenience wrapper returning all metrics in one dict."""
    r = rmse(y_true, y_pred)
    m = mae(y_true,  y_pred)
    f = f1_by_category(y_true, y_pred)
    sk = round(skill_score(r, rmse_baseline), 4) if rmse_baseline else None
    return {"rmse": round(r, 4), "mae": round(m, 4), "f1": round(f, 4), "skill": sk}


def results_table(rows: list[dict]) -> pd.DataFrame:
    """
    Format a list of metric dicts into a readable DataFrame.

    Each dict must have keys: model, rmse, mae, f1, skill.
    Optional: rmse_std, mae_std, f1_std for seed-stability rows.
    Returns a DataFrame with model as index.

    WHY a dedicated function: centralises column ordering and
    rounding so every caller (run_experiments, tests) sees the same table.
    """
    required_models = {
        "Persistence", "MovingAverage7", "Ridge",
        "RandomForest", "RandomForest+Kalman",
    }
    model_names = {r["model"] for r in rows}
    missing = required_models - model_names
    if missing:
        raise ValueError(f"results_table: missing required models {missing}")

    df = pd.DataFrame(rows).set_index("model")
    # Enforce column order
    ordered_cols = ["rmse", "rmse_std", "mae", "mae_std", "f1", "f1_std", "skill"]
    cols = [c for c in ordered_cols if c in df.columns]
    return df[cols].round(4)


# ---------------------------------------------------------------------------
# Evaluator class (bundles metrics + plots)
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Bundles metric functions and plotting helpers for a fixed test set.

    Parameters
    ----------
    y_true          : ground-truth PM2.5 values (ug/m3), 1-D array
    dates           : DatetimeIndex aligned with y_true
    rmse_baseline   : RMSE of the persistence model (for skill scores)
    figures_dir     : where to save PNGs
    """

    def __init__(
        self,
        y_true: np.ndarray,
        dates: pd.DatetimeIndex,
        rmse_baseline: float,
        figures_dir: Path,
    ):
        self.y_true       = np.asarray(y_true, dtype=float)
        self.dates        = dates
        self.rmse_baseline = float(rmse_baseline)
        self.figures_dir  = Path(figures_dir)

    # -- Metric methods --

    def rmse(self, y_pred: np.ndarray) -> float:
        return rmse(self.y_true, y_pred)

    def mae(self, y_pred: np.ndarray) -> float:
        return mae(self.y_true, y_pred)

    def f1_by_category(self, y_pred: np.ndarray) -> float:
        return f1_by_category(self.y_true, y_pred)

    def confusion_matrix(self, y_pred: np.ndarray) -> tuple[np.ndarray, list[str]]:
        return confusion_matrix(self.y_true, y_pred)

    def skill_score(self, y_pred: np.ndarray) -> float:
        return skill_score(self.rmse(y_pred), self.rmse_baseline)

    def compute_all(self, y_pred: np.ndarray) -> dict:
        return compute_all(self.y_true, y_pred, self.rmse_baseline)

    @staticmethod
    def results_table(rows: list[dict]) -> pd.DataFrame:
        return results_table(rows)

    # -- Plot methods --

    def plot_predicted_vs_actual(
        self,
        predictions: dict[str, np.ndarray],
        highlight_models: list[str] | None = None,
        filename: str = "predicted_vs_actual.png",
    ) -> Path:
        """
        Line plot of actual vs predicted PM2.5 on the test period.

        predictions : {model_name: pred_array} aligned with self.dates
        highlight_models : names to draw bold; others drawn faded.
        """
        if highlight_models is None:
            highlight_models = list(predictions.keys())

        palette = [ACCENT, SMOG_COL, CYAN, GOLD, GREEN,
                   "#a8dadc", "#e63946", "#b5838d"]

        fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)

        # Actual
        ax.plot(self.dates, self.y_true, color=TEXT_COL, alpha=0.5,
                linewidth=0.8, label="Actual PM2.5", zorder=1)

        for i, (name, preds) in enumerate(predictions.items()):
            is_highlight = name in highlight_models
            color  = palette[i % len(palette)]
            lw     = 1.8 if is_highlight else 0.9
            alpha  = 0.9 if is_highlight else 0.4
            zorder = 3 if is_highlight else 2
            ax.plot(self.dates[:len(preds)], preds[:len(self.y_true)],
                    color=color, linewidth=lw, alpha=alpha,
                    label=name, zorder=zorder)

        _style_ax(ax, "Predicted vs Actual PM2.5 — Test Period (Apr 2024 – Feb 2025)",
                  "Date", "PM2.5 (ug/m3)")
        leg = ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL,
                        framealpha=0.8, ncol=3)
        fig.tight_layout(pad=1.5)
        path = self.figures_dir / filename
        _save_fig(fig, path)
        return path

    def plot_rmse_mae_bars(
        self,
        model_metrics: dict[str, dict],
        filename: str = "rmse_mae_bars.png",
    ) -> Path:
        """
        Grouped bar chart of RMSE and MAE per model.

        model_metrics : {model_name: {"rmse": ..., "mae": ...}}
        """
        names  = list(model_metrics.keys())
        rmses  = [model_metrics[n]["rmse"] for n in names]
        maes   = [model_metrics[n]["mae"]  for n in names]

        x      = np.arange(len(names))
        width  = 0.38

        fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
        bars_r = ax.bar(x - width/2, rmses, width, color=SMOG_COL, alpha=0.85,
                        label="RMSE", zorder=2)
        bars_m = ax.bar(x + width/2, maes,  width, color=ACCENT,  alpha=0.85,
                        label="MAE", zorder=2)

        for bar, val in zip(bars_r, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=7, color=TEXT_COL)
        for bar, val in zip(bars_m, maes):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=7, color=TEXT_COL)

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=8)
        ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COL)
        _style_ax(ax, "RMSE and MAE by Model — Test Set", "Model", "Error (ug/m3)")
        fig.tight_layout(pad=1.5)
        path = self.figures_dir / filename
        _save_fig(fig, path)
        return path

    def plot_confusion_matrix(
        self,
        y_pred: np.ndarray,
        model_name: str = "",
        filename: str = "confusion_matrix.png",
    ) -> Path:
        """
        Heatmap confusion matrix over AQI categories.

        Rows = actual category, columns = predicted category.
        Cell text shows count; colour intensity shows proportion of row total.
        """
        mat, labels = confusion_matrix(self.y_true, y_pred)
        norm = mat.astype(float) / (mat.sum(axis=1, keepdims=True) + 1e-9)

        fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
        im = ax.imshow(norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

        for r in range(len(labels)):
            for c in range(len(labels)):
                cnt  = mat[r, c]
                nval = norm[r, c]
                ax.text(c, r, str(cnt),
                        ha="center", va="center", fontsize=9,
                        color="black" if nval > 0.4 else TEXT_COL,
                        fontweight="bold" if r == c else "normal")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([l[:8] for l in labels], rotation=35,
                           ha="right", fontsize=8)
        ax.set_yticklabels([l[:8] for l in labels], fontsize=8)

        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("Fraction of actual row", color=TEXT_COL, fontsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL, fontsize=8)

        title = f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix"
        _style_ax(ax, title, "Predicted Category", "Actual Category")
        fig.tight_layout(pad=1.5)
        path = self.figures_dir / filename
        _save_fig(fig, path)
        return path

    def plot_feature_importances(
        self,
        importances: dict[str, float],
        filename: str = "feature_importances.png",
    ) -> Path:
        """
        Horizontal bar chart of Random Forest feature importances.
        Shows top-15 features sorted by importance.
        """
        items  = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
        names  = [i[0] for i in items]
        vals   = [i[1] for i in items]

        fig, ax = plt.subplots(figsize=(9, 6), facecolor=DARK_BG)
        colors  = [SMOG_COL if v >= 0.1 else ACCENT for v in vals]
        bars    = ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8, color=TEXT_COL)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        _style_ax(ax, "Random Forest Feature Importances (h=1)",
                  "Importance", "Feature")
        fig.tight_layout(pad=1.5)
        path = self.figures_dir / filename
        _save_fig(fig, path)
        return path

    def plot_skill_scores(
        self,
        model_metrics: dict[str, dict],
        filename: str = "skill_scores.png",
    ) -> Path:
        """Bar chart of skill score per model (excluding baseline)."""
        names  = [n for n in model_metrics if model_metrics[n].get("skill") is not None]
        skills = [model_metrics[n]["skill"] for n in names]

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
        colors  = [GREEN if s > 0 else SMOG_COL for s in skills]
        bars    = ax.bar(range(len(names)), skills, color=colors, alpha=0.85, width=0.6)
        for bar, val in zip(bars, skills):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (0.003 if val >= 0 else -0.008),
                    f"{val:+.3f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8, color=TEXT_COL)
        ax.axhline(0, color=TEXT_COL, linewidth=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=9)
        _style_ax(ax, "Skill Score vs Persistence Baseline", "Model", "Skill Score")
        fig.tight_layout(pad=1.5)
        path = self.figures_dir / filename
        _save_fig(fig, path)
        return path

    def plot_rf_seed_spread(
        self,
        seed_rmses: list[float],
        seed_maes:  list[float],
        seeds:      list[int],
        filename:   str = "rf_seed_spread.png",
    ) -> Path:
        """Bar chart of per-seed RMSE and MAE for the RF stability analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK_BG)

        for ax, vals, label, color in [
            (axes[0], seed_rmses, "RMSE (ug/m3)", SMOG_COL),
            (axes[1], seed_maes,  "MAE (ug/m3)",  ACCENT),
        ]:
            ax.bar(range(len(seeds)), vals, color=color, alpha=0.85, width=0.6)
            ax.axhline(np.mean(vals), color=TEXT_COL, linewidth=1.0,
                       linestyle="--", label=f"Mean {np.mean(vals):.2f}")
            ax.set_xticks(range(len(seeds)))
            ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
            ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL)
            _style_ax(ax, f"RF {label} by Seed", "Seed", label)

        fig.suptitle("Random Forest Stability Across Seeds 42–51",
                     color=TEXT_COL, fontsize=11, y=1.01)
        fig.tight_layout(pad=1.5)
        path = self.figures_dir / filename
        _save_fig(fig, path)
        return path