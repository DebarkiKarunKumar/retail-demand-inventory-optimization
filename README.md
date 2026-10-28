# Retail Demand Forecasting & Inventory Decision Support

**Python · Pandas · scikit-learn · XGBoost · Time-series forecasting · Inventory analytics**

## Overview

Retailers must balance product availability against excess inventory. This project uses the public Walmart M5 sales dataset to investigate two questions: **Can machine learning improve short-term demand forecasts over simple baselines?** And **can those forecasts inform better replenishment decisions under explicitly stated assumptions?**

The project covers data checks, preparation, exploratory analysis, baseline forecasts, Ridge and XGBoost modeling, chronological evaluation, a frozen final holdout, and a paired inventory simulation.

## Dataset and scope

- **Source:** [M5 Forecasting — Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- **Scope:** 20 historically high-volume product series from store `CA_1`, department `FOODS_3`.
- **Original training:** `d_1`–`d_1857`; **validation:** `d_1858`–`d_1885`.
- **Final fit:** through `d_1885`; **untouched final holdout:** `d_1886`–`d_1913`.
- **Forecast horizon:** 28 days, fixed-origin predictions without observed future sales as model inputs.

The selected subset is a deliberate portfolio-sized experiment, not a representative sample of all Walmart stores or items.

## Forecasting approach

1. **Naive and weekly seasonal-naive baselines:** simple historical reference forecasts.
2. **Global Ridge regression:** horizon-aware regression using historical demand and calendar features.
3. **Global XGBoost:** horizon-aware boosted regression using product identity, historical lags and rolling statistics, known-in-advance calendar signals, and historical price information. Historical missing prices were retained as missing, not replaced with invented values.

XGBoost was chosen using historical backtests and validation evidence. The selected configuration was frozen before final holdout scoring. Predictions were generated and hashed separately from scoring.

### Validation results (same 560 product-days)

| Model | Pooled product-day WAPE ↓ | Pooled MAE ↓ |
|---|---:|---:|
| Weekly seasonal naive | 47.24% | 8.50 |
| Global Ridge | 44.20% | 7.96 |
| **XGBoost** | **39.87%** | **7.18** |

XGBoost reduced validation WAPE by approximately **15.6% relative to weekly seasonal naive**.

### Final XGBoost holdout (560 product-days)

| Metric | Value |
|---|---:|
| Pooled product-day WAPE | **34.80%** |
| Pooled MAE | 6.44 units |
| Pooled RMSE | 9.82 units |
| Pooled signed bias | +0.86 units |
| Aggregate daily WAPE | 12.27% |

The holdout is a different period from validation; the lower holdout WAPE does not prove accuracy improved over time. WAPE is an error metric, **not a percentage accuracy score**. Aggregate daily errors can cancel across products, so pooled product-day WAPE is the primary model-selection metric.

## Inventory decision simulation

A deterministic 28-day simulation compares two policies on the same recorded sales, used as a **demand proxy**:

- **Historical policy:** fixed weekday means from the last 56 days before the holdout.
- **Forecast-informed policy:** the original frozen 28-day XGBoost predictions, without refitting or updating them.

Both use identical initial inventory, daily review, lost-sales accounting, safety-stock rules, order timing and illustrative costs. Orders arrive before demand on day `t + lead_time`, never on the day placed. Nine predeclared scenarios vary lead time (1, 2, 3 days) and shortage cost (0.10, 0.50, 1.00); holding cost stays at 0.01 per unit per day.

### Base scenario (1-day lead time; shortage cost 0.50)

| Measure | Historical policy | XGBoost-informed |
|---|---:|---:|
| Fill rate | 98.73% | **98.93%** |
| Simulated unmet units | 132 | **111** |
| Product-days with shortages | 17 | **14** |
| Average daily inventory (units) | **502.50** | 530.04 |
| Total simulated cost (illustrative units) | 206.70 | **203.91** |

This is **15.9% fewer simulated unmet units** and **1.35% lower simulated cost** in the base scenario, alongside higher inventory. Across nine scenarios, the forecast-informed policy had lower simulated costs in **six**, and higher costs in **three**. The business outcome depends on shortage penalties and holding costs—not forecast error alone.

## Validation and quality controls

The latest local test output reported **457 passing tests** and no failures. Tests cover data checks, evaluation metrics, leakage safeguards, forecast integrity, and inventory accounting. The final forecast phase generated 560 frozen predictions, with SHA256 matching recorded metadata; the protected scoring phase ran separately without refitting or tuning.

## Repository guide

| Location | Purpose |
|---|---|
| `src/` | Data preparation, evaluation, forecasting and inventory simulation code |
| `tests/` | Synthetic and regression tests |
| `docs/METHODOLOGY.md` | Methods, assumptions and leakage controls |
| `docs/RESULTS.md` | Full model and inventory results |
| `requirements.txt` | Python dependencies |
| `data/` | Locally downloaded or prepared data; **not redistributed** |
| `outputs/` | Generated evaluation outputs; review before sharing |

## Running the project

This repository contains the code and tests; **the M5 dataset is not redistributed**. Download the required data from the [Kaggle competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) under its applicable terms.

For Windows PowerShell, from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -W error::DeprecationWarning -m pytest tests/ -q
```

These commands set up the environment and run the synthetic tests; they do **not** reproduce the numerical results without the source data and saved model outputs. The raw-data-to-results sequence must be checked in the clean public copy before claiming one-command reproducibility. The final holdout was evaluated once during development; **do not use it for further tuning**.

For the implementation details and output interpretation, see [Methodology](docs/METHODOLOGY.md) and [Results](docs/RESULTS.md).

## Limitations

- Only 20 selected high-volume product series in one store and department were studied.
- Recorded sales are not necessarily unconstrained customer demand; simulated shortages are **not measured historical lost sales**.
- Initial stock, lead times, safety stock, costs and replenishment constraints are illustrative assumptions.
- The inventory analysis is a deterministic proof of concept, not a live trial or evidence of actual Walmart savings.
- This experiment uses WAPE on a small subset, **not the official M5 leaderboard metric (WRMSSE)**.
- No production deployment or real operational impact is claimed.

## Takeaway

The forecasting model improved on the tested baselines, but the inventory results show why forecast accuracy alone is not the final business measure. The forecast-informed policy reduced simulated shortages; whether it lowered cost depended on how expensive shortages were assumed to be. This is a reproducible **portfolio experiment**, not a deployed replenishment system.
