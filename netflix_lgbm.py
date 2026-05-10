# =============================================================================
# Windowing Strategies and Box Office Prediction
# Netflix application: scenario-based analysis with LightGBM
#
#
# Trains on non-Netflix titles and applies the model to Netflix originals
# under three distribution scenarios:
#   1. STANDARD       - no window variables
#   2. EXTENDED       - observed window variables
#   3. Day-and-date   - forced theatrical-to-digital window = 0
#
# Reports aggregate predictions, bias relative to observed box office
# (n=15 titles with measurable theatrical grosses), and bootstrap
# robustness estimates (10,000 replications).
# =============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MultiLabelBinarizer


# -----------------------------------------------------------------------------
# 0) CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH        = "DB.xlsx"
TARGET           = "boxOfficeWorldwide"
TIME_COL         = "startYear"
NETFLIX_FLAG_COL = "nextlix_original"
ID_COL           = "tconst"
TITLE_COL        = "originalTitle"

WINDOW_VARS      = ["R.Dig-theatr", "R.Phys-theatr", "R.TV-theatr"]
WINDOW_DATE_COLS = ["RL.Theatrical", "RL.Premiere", "RL.Digital", "RL.Physical", "RL.TV"]

RANDOM_STATE = 42
N_JOBS       = -1
N_BOOT       = 10000
BOOT_SEED    = 42

MIN_FREQ_GENRES   = 25
MIN_FREQ_COUNTRY  = 25
MIN_FREQ_LANGUAGE = 25
TOP_K_DIRECTORS   = 10
TOP_K_WRITERS     = 10
TOP_K_ACTORS      = 10

# Best hyperparameters from tuning (mod_lgbm.py)
BEST_PARAMS_STANDARD = {
    "num_leaves": 127, "num_boost_round": 8000, "min_data_in_leaf": 100,
    "max_depth": -1, "learning_rate": 0.08, "lambda_l2": 0.0, "lambda_l1": 0.0,
    "feature_fraction": 0.7, "bagging_freq": 1, "bagging_fraction": 0.7,
}

BEST_PARAMS_EXTENDED = {
    "num_leaves": 31, "num_boost_round": 8000, "min_data_in_leaf": 20,
    "max_depth": -1, "learning_rate": 0.08, "lambda_l2": 1.0, "lambda_l1": 0.0,
    "feature_fraction": 0.7, "bagging_freq": 0, "bagging_fraction": 0.7,
}


# -----------------------------------------------------------------------------
# 1) HELPERS
# -----------------------------------------------------------------------------
def split_comma_list(x):
    return [i.strip() for i in str(x).split(",") if i.strip()]


def multihot_column(df_in: pd.DataFrame, col: str, prefix: str, min_freq: int = 20) -> pd.DataFrame:
    s = df_in[col].fillna("").astype(str).apply(split_comma_list)
    freq = pd.Series([i for sub in s for i in sub]).value_counts()
    vocab = freq[freq >= min_freq].index.tolist()
    mlb = MultiLabelBinarizer(classes=vocab)
    mat = mlb.fit_transform(s)
    return pd.DataFrame(mat, columns=[f"{prefix}_{c}" for c in mlb.classes_], index=df_in.index)


def top_k_entities(series: pd.Series, k: int = 10):
    s = series.fillna("").astype(str).apply(split_comma_list)
    freq = pd.Series([i for sub in s for i in sub]).value_counts()
    return freq.head(k).index.tolist()


def map_top_entity(series: pd.Series, top_list: list):
    def mapper(x):
        for i in split_comma_list(x):
            if i in top_list:
                return i
        return "OTHER"
    return series.fillna("").astype(str).apply(mapper)


def build_preprocessor(X: pd.DataFrame):
    cat_cols = [c for c in ["rated", "director_top", "writer_top", "actor_top"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("to_str",  FunctionTransformer(lambda Z: Z.astype(str), feature_names_out="one-to-one")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", cat_tf, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )


def assert_no_datetime(X: pd.DataFrame, label: str):
    dt_cols = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    if dt_cols:
        raise ValueError(f"Datetime columns still in X for {label}: {dt_cols}")


def prepare_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    for col, prefix, min_freq in [
        ("genres",   "genre",   MIN_FREQ_GENRES),
        ("country",  "country", MIN_FREQ_COUNTRY),
        ("language", "lang",    MIN_FREQ_LANGUAGE),
    ]:
        if col in df.columns:
            df = pd.concat([df, multihot_column(df, col, prefix, min_freq)], axis=1)
            df.drop(columns=[col], inplace=True)

    if "rated" in df.columns:
        df["rated"] = (
            df["rated"].astype(str).str.upper().str.strip()
            .replace({"N/A": "__MISSING__", "": "__MISSING__", "NONE": "__MISSING__", "NAN": "__MISSING__"})
        )
    else:
        df["rated"] = "__MISSING__"

    if "awards" in df.columns:
        s = df["awards"].astype(str)
        df["award_oscar"] = s.str.contains("Oscar",   na=False).astype(int)
        df["award_win"]   = s.str.contains("win",     na=False, case=False).astype(int)
        df["award_nom"]   = s.str.contains("nominat", na=False, case=False).astype(int)
        df.drop(columns=["awards"], inplace=True)
    else:
        df["award_oscar"] = df["award_win"] = df["award_nom"] = 0

    for col, k, outcol in [
        ("director", TOP_K_DIRECTORS, "director_top"),
        ("writer",   TOP_K_WRITERS,   "writer_top"),
        ("actors",   TOP_K_ACTORS,    "actor_top"),
    ]:
        if col in df.columns:
            df[outcol] = map_top_entity(df[col], top_k_entities(df[col], k=k))
            df.drop(columns=[col], inplace=True)
        else:
            df[outcol] = "OTHER"

    return df


def train_lgbm(X_train: pd.DataFrame, y_train_usd: np.ndarray, best_params: dict):
    prep = build_preprocessor(X_train)
    X_t  = prep.fit_transform(X_train)

    core_params = {
        "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
        "seed": RANDOM_STATE, "verbosity": -1,
        "num_threads": 0 if N_JOBS == -1 else int(N_JOBS),
    }
    train_params = core_params.copy()
    train_params.update({k: v for k, v in best_params.items() if k != "num_boost_round"})

    dtrain = lgb.Dataset(X_t, label=np.log1p(y_train_usd), free_raw_data=False)
    bst    = lgb.train(params=train_params, train_set=dtrain,
                       num_boost_round=int(best_params.get("num_boost_round", 2000)))
    return prep, bst


def predict_usd(prep, bst, X_df: pd.DataFrame) -> np.ndarray:
    pred_log = bst.predict(prep.transform(X_df))
    return np.clip(np.expm1(pred_log), 0.0, None)


# -----------------------------------------------------------------------------
# 2) LOAD AND PREPARE DATA
# -----------------------------------------------------------------------------
df_raw = pd.read_excel(DATA_PATH, engine="openpyxl")
df_raw[TARGET] = pd.to_numeric(df_raw[TARGET], errors="coerce")
for col in [TARGET, TIME_COL, NETFLIX_FLAG_COL]:
    if col not in df_raw.columns:
        raise ValueError(f"Missing required column: {col}")

df = prepare_df(df_raw)


# -----------------------------------------------------------------------------
# 3) TRAIN / TEST SPLIT (non-Netflix vs Netflix)
# -----------------------------------------------------------------------------
nf_flag = df[NETFLIX_FLAG_COL]
mask_nf = (nf_flag.astype(str).str.lower().isin(["1", "true", "yes"])
           if nf_flag.dtype != bool else nf_flag)

df_non_nf = df.loc[~mask_nf].copy()
df_nf     = df.loc[mask_nf].copy()

print(f"Non-Netflix (training): {len(df_non_nf):,}")
print(f"Netflix (scoring):      {len(df_nf):,}")


# -----------------------------------------------------------------------------
# 4) FEATURE SETS
# -----------------------------------------------------------------------------
COMMON_DROP = [
    ID_COL, TITLE_COL, TARGET,
    "boxOfficeDomestic", "boxOfficeInternational",
    NETFLIX_FLAG_COL, "productionMethod", "distributor", "production",
] + [c for c in WINDOW_DATE_COLS if c in df.columns]

DROP_STANDARD = COMMON_DROP + WINDOW_VARS
DROP_EXTENDED = COMMON_DROP

X_train_std = df_non_nf.drop(columns=DROP_STANDARD, errors="ignore")
X_train_ext = df_non_nf.drop(columns=DROP_EXTENDED, errors="ignore")
y_train     = df_non_nf[TARGET].fillna(0.0).astype(float).values

X_nf_std = df_nf.drop(columns=DROP_STANDARD, errors="ignore")
X_nf_ext = df_nf.drop(columns=DROP_EXTENDED, errors="ignore")

for X, label in [(X_train_std, "X_train_std"), (X_train_ext, "X_train_ext"),
                 (X_nf_std,    "X_nf_std"),    (X_nf_ext,    "X_nf_ext")]:
    assert_no_datetime(X, label)


# -----------------------------------------------------------------------------
# 5) TRAIN MODELS
# -----------------------------------------------------------------------------
prep_std, bst_std = train_lgbm(X_train_std, y_train, BEST_PARAMS_STANDARD)
prep_ext, bst_ext = train_lgbm(X_train_ext, y_train, BEST_PARAMS_EXTENDED)


# -----------------------------------------------------------------------------
# 6) SCORE NETFLIX ORIGINALS
# -----------------------------------------------------------------------------
keep_cols = [c for c in [ID_COL, TITLE_COL, TIME_COL, TARGET] + WINDOW_VARS if c in df_nf.columns]
out = df_nf[keep_cols].copy().rename(columns={TARGET: "actual_box_office"})

out["pred_usd_standard"]              = predict_usd(prep_std, bst_std, X_nf_std)
out["pred_usd_extended"]              = predict_usd(prep_ext, bst_ext, X_nf_ext)

out["delta_extended_minus_standard"]  = out["pred_usd_extended"] - out["pred_usd_standard"]
out["pct_change_extended_vs_standard"] = np.where(
    out["pred_usd_standard"] > 0,
    100.0 * out["delta_extended_minus_standard"] / out["pred_usd_standard"],
    np.nan
)

# Forced day-and-date counterfactual
X_nf_daydate = X_nf_ext.copy()
if "R.Dig-theatr" in X_nf_daydate.columns:
    X_nf_daydate["R.Dig-theatr"] = 0.0
out["pred_usd_daydate"] = predict_usd(prep_ext, bst_ext, X_nf_daydate)


# -----------------------------------------------------------------------------
# 7) AGGREGATE RESULTS (Table 4, Panel A — all 608 titles)
# -----------------------------------------------------------------------------
scenario_cols = {
    "STANDARD (no windows)":        "pred_usd_standard",
    "EXTENDED (observed windows)":  "pred_usd_extended",
    "EXTENDED (forced day-and-date)": "pred_usd_daydate",
}

standard_total = out["pred_usd_standard"].sum()
table4a = pd.DataFrame([
    {
        "Scenario":               scenario,
        "Total predicted (USD)":  out[col].sum(),
        "Delta vs STANDARD (USD)": out[col].sum() - standard_total,
        "Delta vs STANDARD (%)":  (out[col].sum() - standard_total) / standard_total * 100,
    }
    for scenario, col in scenario_cols.items() if col in out.columns
])

print("\nTable 4A — Aggregate predictions (all Netflix titles):")
print(table4a.to_string(index=False))


# -----------------------------------------------------------------------------
# 8) BIAS ANALYSIS (Table 4, Panel B — observed subset n=15)
# -----------------------------------------------------------------------------
out["actual_box_office"] = pd.to_numeric(out["actual_box_office"], errors="coerce")
observed = out[out["actual_box_office"].notna() & (out["actual_box_office"] > 0)].copy()
print(f"\nObserved Netflix titles for bias analysis: n={len(observed)}")

if len(observed) == 0:
    raise ValueError("No observed Netflix box office values found in the dataset.")

table4b = pd.DataFrame([
    {
        "Scenario":              scenario,
        "N observed":            len(observed),
        "Total predicted (USD)": observed[col].sum(),
        "Total actual (USD)":    observed["actual_box_office"].sum(),
        "Bias (USD)":            observed[col].sum() - observed["actual_box_office"].sum(),
        "Bias (%)":              (observed[col].sum() - observed["actual_box_office"].sum())
                                 / observed["actual_box_office"].sum() * 100,
    }
    for scenario, col in scenario_cols.items() if col in observed.columns
])

print("\nTable 4B — Bias relative to observed box office:")
print(table4b.to_string(index=False))


# -----------------------------------------------------------------------------
# 9) BOOTSTRAP ROBUSTNESS (Table 5)
# -----------------------------------------------------------------------------
def boot_bias_pct(df_in: pd.DataFrame, pred_col: str) -> float:
    total_pred   = df_in[pred_col].sum()
    total_actual = df_in["actual_box_office"].sum()
    return (total_pred - total_actual) / total_actual * 100


rng           = np.random.default_rng(BOOT_SEED)
boot_records  = []

for _ in range(N_BOOT):
    sample = observed.sample(n=len(observed), replace=True,
                             random_state=int(rng.integers(0, 1_000_000_000)))
    row = {}
    for scenario, col in scenario_cols.items():
        if col in sample.columns:
            row[scenario] = boot_bias_pct(sample, col)
    boot_records.append(row)

boot_df = pd.DataFrame(boot_records)

bootstrap_summary = pd.DataFrame([
    {
        "Scenario":                  scenario,
        "Bootstrap mean bias (%)":   boot_df[scenario].mean(),
        "Bootstrap median bias (%)": boot_df[scenario].median(),
        "CI95 low (%)":              boot_df[scenario].quantile(0.025),
        "CI95 high (%)":             boot_df[scenario].quantile(0.975),
        "n_bootstrap":               N_BOOT,
        "n_observed":                len(observed),
    }
    for scenario in scenario_cols if scenario in boot_df.columns
])

print("\nTable 5 — Bootstrap robustness:")
print(bootstrap_summary.to_string(index=False))

# Bias reduction relative to STANDARD
reductions = []
std_col = "STANDARD (no windows)"
for scenario in ["EXTENDED (observed windows)", "EXTENDED (forced day-and-date)"]:
    if scenario in boot_df.columns:
        diff = boot_df[std_col] - boot_df[scenario]
        reductions.append({
            "Comparison":                       f"STANDARD minus {scenario}",
            "Median reduction (pp)":            diff.median(),
            "CI95 low (pp)":                    diff.quantile(0.025),
            "CI95 high (pp)":                   diff.quantile(0.975),
        })

bootstrap_reduction = pd.DataFrame(reductions)
print("\nBootstrap bias reduction effects:")
print(bootstrap_reduction.to_string(index=False))


# -----------------------------------------------------------------------------
# 10) WINDOW DIAGNOSTIC
# -----------------------------------------------------------------------------
diag = []
for label, dfx in [("non_netflix", df_non_nf), ("netflix", df_nf)]:
    row = {"group": label, "n": len(dfx)}
    if "R.Dig-theatr" in dfx.columns:
        vals = pd.to_numeric(dfx["R.Dig-theatr"], errors="coerce")
        row["share_R_Dig_eq_0"] = float((vals == 0).mean())
        row["mean_R_Dig"]       = float(vals.mean())
        row["median_R_Dig"]     = float(vals.median())
    diag.append(row)

print("\nWindow diagnostic (R.Dig-theatr):")
print(pd.DataFrame(diag).to_string(index=False))


# -----------------------------------------------------------------------------
# 11) SAVE OUTPUTS
# -----------------------------------------------------------------------------
out.to_csv("netflix_lgbm_predictions_scenarios.csv", index=False)
table4a.to_csv("netflix_table4a_scenario_aggregates.csv", index=False)
table4b.to_csv("netflix_table4b_bias_observed.csv", index=False)
observed.to_csv("netflix_lgbm_observed_subset.csv", index=False)
bootstrap_summary.to_csv("netflix_lgbm_bootstrap_summary.csv", index=False)
boot_df.to_csv("netflix_lgbm_bootstrap_iterations.csv", index=False)
bootstrap_reduction.to_csv("netflix_lgbm_bootstrap_reduction.csv", index=False)
pd.DataFrame(diag).to_csv("netflix_lgbm_window_diagnostic.csv", index=False)
