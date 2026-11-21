# Report Notes -- Lahore Smog Analysis

Reference document for the final AI project report.

---

## 1. Problem and Motivation

- Lahore is consistently ranked among the top 5 most polluted cities globally.
- PM2.5 (fine particulate matter) causes cardiovascular and respiratory disease.
- Existing public forecasts for Pakistan are coarse or absent.
- Goal: build a reproducible, honest forecast pipeline using only open data.

---

## 2. Data and Cleaning

**Source:** AQICN US Embassy Lahore monitor (US AQI scale, 2019-05-09 to 2025-02-18).
**Key cleaning decisions:**
- pm25 column is US AQI, not ug/m3. Inverse EPA piecewise-linear formula converts it.
- 2 outlier rows removed (Nov 2024, AQI 544 and 580): IQR x 3, lower clipped at 0.
- 231 days dropped (21 gaps > 3 consecutive days); gaps up to 3 days interpolated.
- **Calendar-day bug (Phase 2 fix):** pandas .shift(n) on a sparse index uses row
  positions, silently assigning stale values across data gaps. Fixed with
  _calendar_shift() that densifies to a full daily grid before shifting.
- Weather merged from Open-Meteo ERA5 by inner join (5 variables).

---

## 3. Method Per Phase

- **Phase 0:** Folder structure, config.py, utils.py set_seed.
- **Phase 1:** AirQualityDataset: load, clean, features (calendar-day safe), split. download_weather.py.
- **Phase 2:** EDA training-only. Calendar-day lag bug discovered and fixed.
- **Phase 3:** Ridge + RF, walk-forward tuning TimeSeriesSplit k=5, test evaluation.
- **Phase 4:** Kalman filter: R from residuals, Q from grid search, raw vs filtered comparison.
- **Phase 5:** Evaluator, run_experiments.py reproduces everything, JSON exports.
- **Phase 6:** AirQualityEnvironment, ForecastAgent, console menu.
- **Phase 7:** FastAPI backend: 9 endpoints, Pydantic, models loaded once at startup.
- **Phase 8:** React+Vite dashboard: 6 screens, code-split lazy routes, ECharts, r3f.
- **Phase 9:** Documentation, cleanup, fresh-clone check.

---

## 4. Results Tables

### Model Comparison (test set Apr 2024 - Feb 2025)

| Model | RMSE | MAE | Macro F1 | Skill |
|---|---|---|---|---|
| Persistence | 68.464 | 46.883 | 0.4444 | -- |
| MovingAverage7 | 62.076 | 43.419 | 0.4838 | +0.0933 |
| Ridge | 54.943 | 37.66 | 0.3611 | +0.1975 |
| RandomForest | 55.347 +/-0.134 | 37.639 +/-0.152 | 0.4186 +/-0.0077 | +0.1916 |
| RandomForest+Kalman | 55.126 +/-0.27 | 37.586 +/-0.161 | 0.3945 +/-0.0084 | +0.1948 |

Best params: Ridge alpha=0.01, RF 300/8/5

Kalman: Ridge 54.9426 -> 54.4703 (delta 0.4723)
Kalman: RF    55.3467 -> 55.1256 (delta 0.2211)

---

## 5. Figures to Include in Report

- outputs/figures/01_monthly_trend_and_boxplot.png -- Monthly mean + box plot by month
- outputs/figures/04_smog_season_analysis.png -- Smog season vs off-season analysis
- outputs/figures/05_correlations_and_autocorrelation.png -- Weather correlation and autocorrelation
- outputs/figures/03_month_year_heatmap.png -- Month x year heatmap
- outputs/figures/kalman_filtered_vs_raw.png -- Kalman filtered vs raw PM2.5
- outputs/figures/predicted_vs_actual.png -- Predicted vs actual on test set
- outputs/figures/rmse_mae_bars.png -- Model RMSE/MAE comparison bars
- outputs/figures/confusion_matrix.png -- RF confusion matrix
- outputs/figures/feature_importances.png -- RF feature importances
- outputs/figures/rf_seed_spread.png -- RF seed stability

---

## 6. Discussion

- Skill score ~0.19: meaningful given the difficulty. PM2.5 is non-stationary,
  right-skewed, with sharp seasonal regime changes.
- Kalman improvement < 0.5 ug/m3: modest because seasonal dominates noise.
- Ridge macro F1 (0.361) < RF (0.419): linear model misses nonlinear AQI thresholds.
- 47-day data gap in Oct-Dec 2021 creates blind spot in peak smog season.

---

## 7. Limitations

- Single station (US Embassy); spatial variability unmeasured.
- Sensor accuracy ~26 ug/m3; not calibrated against EPA reference instruments.
- 231 days dropped; worst-case events under-represented in training.
- AQI > 500 extrapolated (off-scale).
- t+1 primary target; multi-step forecasts less reliable.
- No real-time feed; last data: 2025-02-18.

---

## 8. Future Work

- Multi-station network (Pakistan EPA + private sensors).
- Deep learning: LSTM or Transformer for sequence modelling.
- Real-time data feed from AQICN API (hourly updates).
- PM10 and NO2 as additional pollutants.
- Ensemble: RF + Ridge + XGBoost stacking for lower variance.

---

## 9. Viva Questions and Short Answers

**Q1: Why did you convert AQI to ug/m3 instead of using AQI directly?**
ML models should predict a physical quantity. AQI has discrete breakpoints that introduce
artificial discontinuities in the target distribution. ug/m3 is continuous and allows the
models to learn smooth relationships.

**Q2: What is the calendar-day bug and why does it matter?**
pandas .shift(n) on a sparse DatetimeIndex shifts by row number, not calendar days.
Across a data gap, lag1 picks up a value from days earlier -- stale data as if no gap existed.
This is leakage. Fixed by densifying the index to a full daily grid before shifting.

**Q3: Why TimeSeriesSplit instead of KFold?**
KFold with shuffling puts future observations in the training fold, leaking future lag values.
TimeSeriesSplit ensures each fold trains on the past and validates on the future -- exactly the
production scenario.

**Q4: How did you estimate R for the Kalman filter?**
From the training period only. Fit a 3-day centred rolling mean as a proxy for the true signal,
then compute the variance of residuals (z - proxy). This is equivalent to a lab calibration:
measure the noise around a known smooth signal.

**Q5: Why is Kalman improvement small?**
The dominant signal driver is seasonal (Oct-Jan smog cycle, 2.77x ratio). The Kalman filter
removes day-to-day measurement noise, which is real but minor. You cannot Kalman-filter away a
seasonal effect.

**Q6: What does skill score mean and what is a good value?**
Skill = 1 - RMSE(model)/RMSE(persistence). 0 = no better than persistence. 1 = perfect.
Our ~0.19 means the model reduces persistence error by 19%. For daily PM2.5, this is competitive
with published regional models on similar short datasets.

**Q7: Why is Ridge macro-F1 lower than RF despite better RMSE?**
RMSE measures continuous error; F1 measures category assignment. The AQI thresholds are
nonlinear decision boundaries. Ridge cannot model these; RF naturally splits at them.
So Ridge predicts continuous values well but misclassifies extreme categories.

**Q8: How did you prevent test-set leakage throughout?**
Time split is strictly chronological. Scaler fitted only on train. Tuning done on train+val.
Test set touched exactly once at the end. Rolling features computed with shift(1) to exclude
current day. Kalman R and Q estimated on training period only.

**Q9: Why use Random Forest over XGBoost?**
RF was the Phase 3 scope. It provides interpretable feature importances, is robust to outliers,
and does not require gradient tuning. XGBoost typically improves by 2-5% but adds tuning
complexity -- left as future work.

**Q10: What would you do with more data?**
More stations (spatial coverage), longer history (more smog seasons), hourly resolution
(capture morning peaks), PM10 and NO2, and a real-time feed for live forecasting.

