# Viva Guide -- Module-by-Module Defence

For each module: what it does and WHY, in terms you can explain under exam pressure.

---

## dataset.py

- Loads lahore-us-embassy-air-quality.csv and lahore_weather.csv.
- Cleans data: strips BOM from column names, converts AQI->ug/m3 via inverse EPA formula, removes duplicates and outliers (IQR*3), interpolates short gaps, drops long ones.
- Adds features using _calendar_shift() and _calendar_rolling() -- densifies the DatetimeIndex to a full daily grid first, so gaps propagate NaN instead of silently using stale row values.
- Splits into train/val/test (70/15/15) strictly by date -- never shuffled.
- WHY: all downstream modules call these methods instead of re-implementing data prep, keeping the pipeline reproducible and leakage-free.

## eda.py

- Runs only on the training split (never the test set) to avoid contaminating evaluation.
- Produces 5 ECharts-ready figures: monthly trend, seasonal boxplot, weekday effect, month-year heatmap, weather correlations, autocorrelation.
- Writes eda_summary.json with all computed statistics (peak month=Dec, smog/off ratio=2.77, lag-1 autocorr=0.852).
- All numbers in the JSON are computed from real data, never hard-coded.
- WHY: a separate EDA module keeps exploration cleanly isolated from modelling and re-runnable without touching the pipeline.

## forecasters.py

- Defines abstract Forecaster base class (fit/predict), plus Persistence, MovingAverage, Ridge (Pipeline: StandardScaler + Ridge), and RandomForest subclasses.
- build_model_features() is the single source of truth for the feature vector: lag1-6, roll7, 4 weather vars, month_sin/cos, doy_sin/cos.
- Uses _calendar_shift() for lags -- this prevents the gap-across-index bug (Phase 2 fix) from resurfacing at inference time.
- The Ridge Pipeline ensures the scaler sees only training data; it cannot accidentally be fit on validation or test rows.
- WHY single feature builder: if training and inference used different feature code, any divergence would create silent prediction errors that are hard to detect.

## tuning.py

- Grid-searches Ridge alpha and RF (n_estimators, max_depth, min_samples_leaf) using TimeSeriesSplit k=5.
- TimeSeriesSplit is mandatory here: KFold with shuffling would put future observations in training folds, leaking lag features.
- Only train+val data are passed in; the test set is never seen.
- Uses RMSE as the tuning criterion (same as the final evaluation metric, so the two are aligned).
- WHY 5 splits: on ~1600 rows, 5 folds give ~320-row validation windows -- enough to estimate seasonal bias without tiny folds.

## kalman.py

- Pure-NumPy 1-D Kalman filter with random-walk state transition (no external filterpy dependency).
- R (sensor noise variance) is estimated from the TRAINING period only: variance of residuals z - 3-day centred MA. The centred window is used only for calibration, not production filtering.
- Q (process noise) is chosen by grid search on validation RMSE. Q=R=695.4 means the filter trusts measurements and model equally.
- filter_series() correctly handles calendar gaps: runs predict() for each missing calendar day (variance grows by Q), then update() when a reading exists.
- WHY causal: using a smoother (which uses future observations) would leak future PM2.5 values into lag features -- data leakage.

## evaluator.py

- Provides standalone metric functions: rmse(), mae(), f1_by_category() (macro), confusion_matrix(), skill_score().
- skill_score = 1 - RMSE(model)/RMSE(persistence): a standard meteorological metric, scale-independent.
- Macro F1 is used instead of accuracy or weighted F1 because the class distribution is imbalanced (many Unhealthy days, few Good days).
- The Evaluator class bundles metrics with plot methods (predicted vs actual, RMSE bars, confusion matrix, feature importances) so every chart uses the same color scheme.
- WHY separate from forecasters: metrics and plots are reusable across Phase 3, Phase 4, and the backend without importing model-specific code.

## environment.py

- AirQualityEnvironment replays the cleaned series one calendar day at a time (not one row at a time).
- reset() rewinds to the first date that has a full 7-day prior calendar history (not just 7 prior rows).
- step() advances to the next date that EXISTS in the DataFrame index, skipping dates that were dropped as long gaps.
- Returns a Percept frozen dataclass (date, pm25, temperature, humidity, wind_speed, rainfall) -- frozen so the agent cannot mutate the environment.
- WHY calendar days: iterating by row would treat a 47-day gap as a single step, making the agent think day 47 is the day after day 1.

## agent.py

- ForecastAgent loads a trained model (default: rf_h1.joblib, best by validation CV-RMSE) using joblib.
- perceive() stores the last 8 (Percept, date) pairs; forecast() calls _build_inference_features() on a mini-DataFrame built from that buffer.
- _build_inference_features() applies the identical lag/roll/cyclic transformations as build_model_features() in forecasters.py, but WITHOUT computing a target column -- so the query-date row is never dropped.
- advise() maps the forecast to an AQI category and returns advisory text from config.AQI_ADVISORY_TEXTS plus the disclaimer.
- WHY no duplicate feature logic: if training and inference used different feature code, any divergence would cause silent prediction errors.

## src/main.py (console menu)

- Provides a text menu: (1) EDA summary, (2) model comparison, (3) date forecast, (4) quit.
- _parse_date() returns None (not raises) for invalid input, treating NaT the same as None.
- show_forecast() calls ForecastAgent.forecast_for_date() and handles ValueError for missing history, out-of-range dates, and missing model files with a clear message.
- If outputs/results/ files are missing, the user is told to run python src/run_experiments.py.
- WHY no crash on bad input: the menu is intended for non-technical users; every error must produce a readable message, never a traceback.

## backend/main.py (FastAPI)

- FastAPI app with 9 GET endpoints: /api/health, /api/kpis, /api/today, /api/series, /api/forecast, /api/eda, /api/models, /api/kalman, /api/about.
- The lifespan context manager loads all JSON result files and joblib models ONCE at startup into app.state.ctx (an AppState dataclass).
- Every request reads from app.state.ctx -- zero disk I/O per request after startup.
- /api/forecast runs the REAL model on real features using _build_inference_features() (same function as the agent) -- no duplicated logic.
- Missing files return a 503 with a message telling the user to run src/run_experiments.py.

## frontend (App.tsx + pages)

- React 18 + Vite + TypeScript. All data fetched via TanStack Query hooks in src/hooks/useApi.ts; every component shows a skeleton while loading and an ErrorState on failure.
- Six pages: Dashboard (3D smog scene + gauge + KPI cards + 90-day chart), Forecast (date picker + confidence bar), EDA (5 charts + 3D toggle), Models (sortable table + replay animation), Kalman (replay animation + comparison), About (PEAS + team).
- All routes are code-split with React.lazy + Suspense; the 3D SmogScene is its own lazy chunk to avoid blocking the initial render.
- The 3D smog particle density, color, and speed are driven by real PM2.5 from /api/today; WebGL fallback shows a static gradient if unavailable.
- WHY TanStack Query: it handles caching, background refetch, and loading/error states automatically, keeping page components free of fetch boilerplate.

---

## General Defence Tips

- Every design decision has a WHY. If asked why you did X, the answer is always a correctness or leakage concern, not convenience.
- The calendar-day bug is the most defensible Phase 2 contribution. Know it cold: what it is, why it matters, how you fixed it, how you tested it.
- Skill score ~0.19 is honest. Do not oversell it. Explain what it means (19% improvement over doing nothing) and why it is hard to do better on a single-station dataset with large gaps.
- The Kalman filter is correct (causal, R from training only, Q from validation) even though the improvement is small. Defend correctness, not magnitude.
- If asked about a number, always trace it back to an output file (outputs/results/*.json) or a line of code. Never say approximately.
