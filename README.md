# Windowing Strategies and Box Office Prediction
### An Explainable Machine Learning Framework with a Netflix Application


---

## Overview

This repository contains the full replication code for the empirical analysis presented in the paper. The study examines whether incorporating **distribution window variables** (the time intervals between theatrical release and subsequent digital, physical, and television channels) improves the prediction of film box office performance across three ensemble learning models: **LightGBM**, **XGBoost**, and **Random Forest**.

Key findings:

- Window-related variables consistently improve predictive accuracy across all three model families, with statistically significant gains confirmed by Diebold–Mariano tests.
- SHAP analysis identifies the theatrical-to-digital interval as one of the most influential structural predictors of revenue.
- When applied to Netflix originals, window-agnostic models overestimate box office by a bootstrap mean of **260.8%**; incorporating window variables reduces this bias to **6.1%** under a forced day-and-date configuration.

---

## Repository Structure

```
windowing-strategies-prediction/
│
├── mod_lgbm.py                  # LightGBM: hyperparameter tuning (STANDARD vs EXTENDED)
├── mod_xgb.py                   # XGBoost: hyperparameter tuning (STANDARD vs EXTENDED)
├── mod_random_forest.py         # Random Forest: hyperparameter tuning (STANDARD vs EXTENDED)
│
├── diebold_lgbm.py              # Diebold–Mariano test for LightGBM specifications
├── feature_importance_lgbm.py   # Feature importance by gain and split (LightGBM EXTENDED)
├── shap_lgbm.py                 # SHAP explainability analysis (LightGBM EXTENDED)
├── netflix_lgbm.py              # Netflix scenario analysis + bootstrap robustness
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Data file:** Place `DB.xlsx` in the root of the repository before running any script. The dataset is not distributed with this code; see the *Data* section below.

---

## Experimental Design

### Two feature specifications

| Specification | Window variables | Description |
|---|---|---|
| **STANDARD** | Excluded | Baseline model using standard film attributes only |
| **EXTENDED** | Included | Adds `R.Dig-theatr`, `R.Phys-theatr`, `R.TV-theatr` |

Window variables capture the number of days between the theatrical release date and each subsequent distribution channel (digital, physical, television). Negative values indicate streaming-first or day-and-date strategies.

### Validation strategy

All models are evaluated using **forward-chaining TimeSeriesSplit** (5 folds), sorted chronologically by release year. This prevents look-ahead bias and ensures predictions are always made on future data relative to the training period.

### Statistical testing

The **Diebold–Mariano test** (Harvey, Leybourne & Newbold, 1997 correction) is applied to compare out-of-sample forecast errors between STANDARD and EXTENDED specifications, using squared error loss on the log₁p scale with a HAC variance estimator.

---

## Scripts

### `mod_lgbm.py`
Tunes LightGBM under both specifications using randomised search (25 iterations) combined with TimeSeriesSplit. Early stopping is applied within each fold (200 rounds). Outputs per-fold metrics, best hyperparameters, and a comparable summary table.

### `mod_xgb.py`
Same methodology as `mod_lgbm.py` applied to XGBoost (`xgb.train` API with early stopping).

### `mod_random_forest.py`
Same methodology applied to Random Forest (no early stopping; bagging-based model).

### `diebold_lgbm.py`
Reproduces the DM test reported in Table 2 of the paper. Uses the best hyperparameters found during tuning (hard-coded from `mod_lgbm.py` outputs) and generates full out-of-sample predictions across all 5 folds before applying the test.

### `feature_importance_lgbm.py`
Trains the EXTENDED LightGBM model on the full dataset and exports feature importance rankings (gain and split) as CSV tables and horizontal bar charts (top 20 features).

### `shap_lgbm.py`
Computes SHAP values on a random subsample (2,000 observations) from the full EXTENDED model. Exports a beeswarm summary plot, a bar summary plot, and a dependence plot for `R.Dig-theatr` (theatrical-to-digital window).

### `netflix_lgbm.py`
Trains on non-Netflix titles exclusively and scores the 608 Netflix originals under three scenarios:
1. **STANDARD** — window variables absent (implicit assumption of theatrical distribution)
2. **EXTENDED** — observed window structure for each title
3. **Day-and-date** — `R.Dig-theatr` forced to 0 for all titles

Reports aggregate predictions (Table 4A), point-estimate bias for the 15 titles with observable theatrical revenues (Table 4B), and bootstrap robustness intervals (Table 5, 10,000 replications).

---

## Setup

```bash
# Clone the repository
git clone https://github.com/caaa81-research/windowing-strategies-prediction.git
cd windowing-strategies-prediction

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Place DB.xlsx in the repository root, then run any script
python mod_lgbm.py
```

Typical execution times on a standard workstation (CPU only):

| Script | Approximate runtime |
|---|---|
| `mod_lgbm.py` | 2–4 h |
| `mod_xgb.py` | 3–5 h |
| `mod_random_forest.py` | 4–8 h |
| `diebold_lgbm.py` | 1–2 h |
| `feature_importance_lgbm.py` | 20–40 min |
| `shap_lgbm.py` | 30–60 min |
| `netflix_lgbm.py` | 20–40 min |

---

## Data

The dataset integrates information from the following publicly accessible sources:

- **IMDb** non-commercial datasets (core film metadata and identifiers)
- **OMDb API** (genres, cast, crew, ratings, awards)
- **TMDb API** (release dates across distribution channels — used to derive window variables)
- **The Numbers** (worldwide box office revenues, via web scraping)
- **Rotten Tomatoes / Metacritic** (critical reception scores)
- **IMDb lists** (Netflix Original identification)

The final dataset covers **150,004 feature films** released between 2016 and 2024, including a subset of **608 Netflix originals** used exclusively for scenario-based analysis.

Because the dataset incorporates data scraped under terms that restrict redistribution, `DB.xlsx` is not included in this repository. Researchers wishing to replicate the study should construct the dataset following the procedure described in Section 4 of the paper.

---

## Outputs

Each script writes its results to the working directory:

| Script | Outputs |
|---|---|
| `mod_lgbm.py` | `final_comparable_table_tss_lgbm.csv`, fold CSVs, best params CSVs |
| `mod_xgb.py` | `final_comparable_table_tss_es.csv`, fold CSVs, best params CSVs |
| `mod_random_forest.py` | `final_comparable_table_tss_rf.csv`, fold CSVs, best params CSVs |
| `diebold_lgbm.py` | Printed DM test results |
| `feature_importance_lgbm.py` | `feature_importance_lgbm_extended/` (CSV + PNG) |
| `shap_lgbm.py` | `shap_lgbm_extended/` (CSV + PNG) |
| `netflix_lgbm.py` | Multiple scenario and bootstrap CSVs |


---

## License

This code is released for academic research purposes. See `LICENSE` for details.
