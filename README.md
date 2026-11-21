# Lahore Smog Analysis and PM2.5 AQI Forecast

A full-stack machine-learning project that analyses six years of Lahore PM2.5 air-quality
data, trains Ridge and Random Forest forecasting models with a causal Kalman filter
preprocessing step, and serves results through a FastAPI backend and React dashboard.

**Course:** AL-2002 Artificial Intelligence | Fall 2026 | BSE-5B | FAST NUCES Karachi

> **Disclaimer:** Forecasts are informational only and not an official health advisory.

---

## Table of Contents

1. [Team](#team)
2. [Screenshots](#screenshots)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Setup](#setup)
6. [How to Get the Data](#how-to-get-the-data)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [Method Summary](#method-summary)
10. [Limitations](#limitations)
11. [Data Attribution](#data-attribution)
12. [AI-Assistance Declaration](#ai-assistance-declaration)

---

## Team

| Name | Student ID | Role |
|---|---|---|
| Muhammad Abdullah Sajid | 24K-3045 | Lead Developer & ML Engineer |
| Muhammad Hammad Rafi | 24K-3117 | Data Engineer & Backend Developer |
| Kabeer Danish | 24K-3118 | Frontend Developer & Visualisation |

**Course:** AL-2002 Artificial Intelligence, Fall 2026
**Institution:** FAST NUCES Karachi, BSE-5B

---

## Screenshots

Place screenshots in docs/screenshots/ after running the dashboard.

---

## Project Structure

`	ext
lahore-smog/
|-- backend/           FastAPI REST API (main.py, schemas.py, services.py)
|-- data/raw/          Place AQICN CSV here; weather auto-downloaded
|-- data/processed/    Cleaned merged CSV (auto-generated)
|-- docs/              screenshots/, REPORT_NOTES.md, VIVA_GUIDE.md
|-- frontend/          React + Vite + TypeScript dashboard
|-- outputs/
|   |-- figures/       PNG charts (auto-generated)
|   |-- models/        Trained models (auto-generated)
|   +-- results/       JSON/CSV metrics (auto-generated)
|-- src/
|   |-- config.py      Paths, seed=42, AQI breakpoints, feature lists
|   |-- dataset.py     Load, clean, feature-engineer, split
|   |-- download_weather.py  Open-Meteo ERA5 fetch (free, no API key)
|   |-- eda.py         EDA figures and summary stats
|   |-- forecasters.py Forecaster classes + build_model_features()
|   |-- tuning.py      Walk-forward TimeSeriesSplit hyperparameter tuning
|   |-- kalman.py      Pure-NumPy 1-D Kalman filter
|   |-- evaluator.py   RMSE / MAE / macro-F1 / skill score
|   |-- environment.py AirQualityEnvironment (calendar-day replay)
|   |-- agent.py       ForecastAgent (perceive / forecast / advise)
|   |-- main.py        Console menu (Phase 6)
|   +-- run_experiments.py  ONE command reproduces everything
|-- tests/             pytest suite (206 tests, 1 skipped)
|-- requirements.txt
|-- run_all.ps1        Windows one-click setup + run
+-- .gitignore
`

---

## Requirements

| Software | Minimum version |
|---|---|
| Python | 3.10 |
| Node.js | 18 |
| npm | 9 |

---

## Setup

### 1. Virtual environment and Python dependencies

`powershell
cd lahore-smog
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
`

### 2. Frontend dependencies

`powershell
cd frontend
npm install
cd ..
`

### 3. One-click Windows start

`powershell
.\run_all.ps1
`

---

## How to Get the Data

### PM2.5 (manual download, required)

1. Register at <https://aqicn.org/data-platform/register/>
2. Download **Lahore, US Embassy** daily CSV.
3. Rename to lahore-us-embassy-air-quality.csv and place in data/raw/.

The file must have two columns: date (format YYYY/M/D) and  pm25 (US AQI).

### Weather (automatic, free, no API key)

`powershell
python src/download_weather.py
`

Calls the Open-Meteo Historical Weather API (ERA5 reanalysis) and saves
data/raw/lahore_weather.csv.

---

## How to Run

`powershell
# Full ML pipeline (cleans data, runs EDA, tunes, trains, evaluates) ~3-5 min
python src/run_experiments.py

# Console menu: EDA summary, model comparison, date-based forecast
python src/main.py

# Backend REST API  (Swagger UI at http://localhost:8000/docs)
uvicorn backend.main:app --reload

# Frontend dashboard
cd frontend
npm run dev
# Open http://localhost:5173

# Tests
pytest -q   # expected: 206 passed, 1 skipped
`

---

## Results

All numbers are from the **test set only** (Apr 2024 - Feb 2025, 255 observations).
The best model was chosen by **validation CV-RMSE** only (never test performance).

### Model Comparison

| Model | RMSE (ug/m3) | MAE | Macro F1 | Skill |
|---|---|---|---|---|
| Persistence | 68.464 | 46.883 | 0.4444 | -- |
| MovingAverage7 | 62.076 | 43.419 | 0.4838 | +0.0933 |
| Ridge | 54.943 | 37.660 | 0.3611 | +0.1975 |
| RandomForest | 55.347 +/- 0.134 | 37.639 +/- 0.152 | 0.4186 +/- 0.0077 | +0.1916 |
| RandomForest+Kalman | 55.126 +/- 0.270 | 37.586 +/- 0.161 | 0.3945 +/- 0.0084 | +0.1948 |

RF mean +/- std across seeds 42-51 (10 runs). Skill = 1 - RMSE(model)/RMSE(persistence).

### Hyperparameters (walk-forward CV, train+val only)

| Model | Parameter | Value | Val CV-RMSE |
|---|---|---|---|
| Ridge | alpha | 0.01 | 54.7261 |
| RandomForest | n_estimators / max_depth / min_samples_leaf | 300 / 8 / 5 | 53.8891 |

### Kalman Filter Effect

| Model | Raw RMSE | Filtered RMSE | Improvement |
|---|---|---|---|
| Ridge | 54.9426 | 54.4703 | 0.4723 ug/m3 |
| RandomForest | 55.3467 | 55.1256 | 0.2211 ug/m3 |

### Discussion

Both Ridge and Random Forest beat the persistence baseline by roughly **20% in RMSE**
(skill ~0.19-0.20), which is meaningful given Lahore PM2.5 is volatile and skewed.

The Kalman filter provides a consistent but modest improvement (< 0.5 ug/m3 RMSE).
The dominant signal driver is the Oct-Jan seasonal smog cycle, not day-to-day
measurement noise, so the Kalman gain is limited.

Macro F1 is lower for Ridge (0.361) than RF (0.419). Ridge cannot capture the
nonlinear threshold between Very Unhealthy and Hazardous driven by wind x temperature.

A 47-day data gap in Oct-Dec 2021 (peak smog season) is absent from training data,
likely causing models to under-estimate extreme PM2.5 episodes.

---

## Method Summary

- **AQI to ug/m3 conversion:** Raw  pm25 column is US AQI. Converted via inverse US EPA piecewise-linear formula.
- **Outliers:** IQR x 3 rule; lower bound clipped at 0 (physically impossible). 2 rows removed (Nov 2024, AQI 544 & 580).
- **Gaps:** Interpolate up to 3 consecutive missing days; drop longer gaps (21 gaps, 231 rows dropped).
- **Calendar-day features:** _calendar_shift() densifies the DatetimeIndex to a contiguous daily grid before shifting. Without this fix, pandas .shift(n) uses row positions and silently assigns stale values across data gaps.
- **Time split:** 70/15/15, strictly chronological. Test set is never seen during tuning or model selection.
- **Tuning:** TimeSeriesSplit k=5 on train+val only. RMSE is the tuning criterion.
- **Kalman filter:** 1-D random-walk model. R estimated from training residuals (variance of z - 3-day centred MA). Q tuned by validation RMSE grid search. Causal: only past observations used.
- **Leakage prevention:** Rolling features use shift(1).rolling(w) (current day excluded). Scaler fitted only on training data (sklearn Pipeline). Test set touched once at final evaluation.

---

## Limitations

- **Single station:** One US Embassy monitor; spatial variability across Lahore not captured.
- **Sensor accuracy:** Measurement uncertainty ~26 ug/m3 (estimated as sqrt(R) from Kalman calibration).
- **Missing data:** 231 days dropped; largest gap is 47 days in Oct-Dec 2021 (peak smog season).
- **AQI > 500:** Off-scale; last EPA breakpoint segment extrapolated and documented.
- **Forecast horizon:** Primary target is t+1; t+3 bonus model available.
- **No real-time feed:** Last observation is 2025-02-18; pipeline must be re-run for updates.
- **Disclaimer:** Forecasts are informational only and not an official health advisory.

---

## Data Attribution

| Source | URL | Licence / Terms |
|---|---|---|
| AQICN / US Embassy Lahore | https://aqicn.org/city/pakistan/lahore/us-embassy/ | Non-commercial research use |
| Open-Meteo (ERA5 reanalysis) | https://open-meteo.com/ | CC BY 4.0 |
| US EPA AQI Technical Document | https://www.airnow.gov/ | Public domain |

---

## AI-Assistance Declaration

This project was built with assistance from **Anthropic Claude** (via the **Kiro IDE**).
The AI generated initial code skeletons, documentation drafts and debugging suggestions.
All code was reviewed, tested and understood by the team before submission.
The team is responsible for all design decisions, results and interpretations.
