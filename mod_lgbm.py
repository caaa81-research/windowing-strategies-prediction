# =============================================================================
# Windowing Strategies and Box Office Prediction
# LightGBM model: hyperparameter tuning with TimeSeriesSplit + early stopping
#
# Compares STANDARD (no window variables) vs EXTENDED (with window variables)
# specifications using forward-chaining temporal validation.
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MultiLabelBinarizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import lightgbm as lgb


# -----------------------------------------------------------------------------
# 0) CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH  = "DB.xlsx"
TARGET     = "boxOfficeWorldwide"
TIME_COL   = "startYear"

RANDOM_STATE         = 42
N_JOBS               = -1
N_SPLITS             = 5
N_ITER               = 25
EARLY_STOPPING_ROUNDS = 200

MIN_FREQ_GENRES   = 25
MIN_FREQ_COUNTRY  = 25
MIN_FREQ_LANGUAGE = 25
TOP_K_DIRECTORS   = 10
TOP_K_WRITERS     = 10
TOP_K_ACTORS      = 10


# -----------------------------------------------------------------------------
# 1) LOAD DATA
# -----------------------------------------------------------------------------
df = pd.read_excel(DATA_PATH, engine="openpyxl")
for col in [TARGET, TIME_COL]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")


# -----------------------------------------------------------------------------
# 2) FEATURE ENGINEERING HELPERS
# -----------------------------------------------------------------------------
def split_comma_list(x):
    return [i.strip() for i in str(x).split(",") if i.strip() != ""]


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


# -----------------------------------------------------------------------------
# 3) FEATURE ENGINEERING
# -----------------------------------------------------------------------------
if "genres" in df.columns:
    df = pd.concat([df, multihot_column(df, "genres", "genre", MIN_FREQ_GENRES)], axis=1)
    df.drop(columns=["genres"], inplace=True)

if "country" in df.columns:
    df = pd.concat([df, multihot_column(df, "country", "country", MIN_FREQ_COUNTRY)], axis=1)
    df.drop(columns=["country"], inplace=True)

if "language" in df.columns:
    df = pd.concat([df, multihot_column(df, "language", "lang", MIN_FREQ_LANGUAGE)], axis=1)
    df.drop(columns=["language"], inplace=True)

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
    df["award_oscar"] = 0
    df["award_win"]   = 0
    df["award_nom"]   = 0

for col, k, outcol in [
    ("director", TOP_K_DIRECTORS, "director_top"),
    ("writer",   TOP_K_WRITERS,   "writer_top"),
    ("actors",   TOP_K_ACTORS,    "actor_top"),
]:
    if col in df.columns:
        top_list = top_k_entities(df[col], k=k)
        df[outcol] = map_top_entity(df[col], top_list)
        df.drop(columns=[col], inplace=True)
    else:
        df[outcol] = "OTHER"


# -----------------------------------------------------------------------------
# 4) COLUMN DROP LISTS
# -----------------------------------------------------------------------------
COMMON_DROP = [
    "tconst", "originalTitle", TARGET,
    "boxOfficeDomestic", "boxOfficeInternational",
    "nextlix_original", "productionMethod", "distributor", "production",
    "RL.Theatrical", "RL.Premiere", "RL.Digital", "RL.Physical", "RL.TV",
]

WINDOW_VARS   = ["R.Dig-theatr", "R.Phys-theatr", "R.TV-theatr"]
DROP_STANDARD = COMMON_DROP + WINDOW_VARS
DROP_EXTENDED = COMMON_DROP


# -----------------------------------------------------------------------------
# 5) PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame):
    cat_cols = ["rated", "director_top", "writer_top", "actor_top"]
    cat_cols = [c for c in cat_cols if c in X.columns]
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


# -----------------------------------------------------------------------------
# 6) HYPERPARAMETER SEARCH SPACE
# -----------------------------------------------------------------------------
def get_param_space_lgbm():
    return {
        "learning_rate":    [0.02, 0.03, 0.05, 0.08],
        "num_leaves":       [31, 63, 127],
        "min_data_in_leaf": [20, 50, 100],
        "feature_fraction": [0.7, 0.85, 1.0],
        "bagging_fraction": [0.7, 0.85, 1.0],
        "bagging_freq":     [0, 1],
        "lambda_l1":        [0.0, 0.1, 1.0],
        "lambda_l2":        [0.0, 1.0, 5.0],
        "max_depth":        [-1, 6, 10],
        "num_boost_round":  [8000],
    }


# -----------------------------------------------------------------------------
# 7) SINGLE PARAMETER SET EVALUATION
# -----------------------------------------------------------------------------
def score_params_tss_lgbm(X: pd.DataFrame, y: np.ndarray, tss: TimeSeriesSplit, params: dict):
    rows = []

    core_params = {
        "objective":    "regression",
        "metric":       "rmse",
        "boosting_type": "gbdt",
        "seed":         RANDOM_STATE,
        "verbosity":    -1,
        "num_threads":  0 if N_JOBS == -1 else int(N_JOBS),
    }

    train_params = core_params.copy()
    train_params.update({k: v for k, v in params.items() if k != "num_boost_round"})
    num_boost_round = int(params["num_boost_round"])

    for fold, (tr, te) in enumerate(tss.split(X), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        y_tr_log = np.log1p(y_tr)
        y_te_log = np.log1p(y_te)

        prep    = build_preprocessor(X_tr)
        X_tr_t  = prep.fit_transform(X_tr)
        X_te_t  = prep.transform(X_te)

        dtrain = lgb.Dataset(X_tr_t, label=y_tr_log, free_raw_data=True)
        dvalid = lgb.Dataset(X_te_t, label=y_te_log, reference=dtrain, free_raw_data=True)

        bst = lgb.train(
            params=train_params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        best_iter = int(getattr(bst, "best_iteration", num_boost_round) or num_boost_round)
        pred_log  = bst.predict(X_te_t, num_iteration=best_iter)

        r2_log   = r2_score(y_te_log, pred_log)
        rmse_log = np.sqrt(mean_squared_error(y_te_log, pred_log))
        mae_log  = mean_absolute_error(y_te_log, pred_log)

        pred = np.clip(np.expm1(pred_log), 0.0, None)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae  = mean_absolute_error(y_te, pred)

        rows.append({
            "fold": fold, "best_iter": best_iter,
            "r2_log": float(r2_log), "rmse_log": float(rmse_log), "mae_log": float(mae_log),
            "rmse": float(rmse), "mae": float(mae), "n_test": int(len(te)),
        })

    fold_df        = pd.DataFrame(rows)
    mean_rmse_log  = float(fold_df["rmse_log"].mean())
    best_iter_mean = float(fold_df["best_iter"].mean())
    return mean_rmse_log, best_iter_mean, fold_df


# -----------------------------------------------------------------------------
# 8) TUNING LOOP
# -----------------------------------------------------------------------------
def tune_timeseries_lgbm(df_in: pd.DataFrame, feature_drop: list, model_label: str):
    df_sorted = df_in.sort_values([TIME_COL]).reset_index(drop=True)
    X = df_sorted.drop(columns=feature_drop, errors="ignore")
    y = df_sorted[TARGET].fillna(0.0).astype(float).values

    assert_no_datetime(X, model_label)

    tss        = TimeSeriesSplit(n_splits=N_SPLITS)
    candidates = list(ParameterSampler(get_param_space_lgbm(), n_iter=N_ITER, random_state=RANDOM_STATE))
    best       = {"rmse_log": np.inf, "params": None, "folds": None, "best_iter_mean": None}

    print("\n" + "=" * 80)
    print(f"TUNING LightGBM: {model_label}")
    print(f"n_splits={N_SPLITS} | n_iter={N_ITER} | early_stopping={EARLY_STOPPING_ROUNDS}")
    print("=" * 80)

    for i, p in enumerate(candidates, start=1):
        mean_rmse_log, best_iter_mean, fold_df = score_params_tss_lgbm(X, y, tss, p)
        print(f"[{i:02d}/{N_ITER}] mean RMSE_log={mean_rmse_log:.5f} | best_iter_mean={best_iter_mean:.1f}")

        if mean_rmse_log < best["rmse_log"]:
            best = {"rmse_log": mean_rmse_log, "params": p, "folds": fold_df, "best_iter_mean": best_iter_mean}

    folds   = best["folds"]
    summary = {
        "model":          model_label,
        "R2_log_mean":    float(folds["r2_log"].mean()),
        "RMSE_log_mean":  float(folds["rmse_log"].mean()),
        "MAE_log_mean":   float(folds["mae_log"].mean()),
        "RMSE_mean":      float(folds["rmse"].mean()),
        "MAE_mean":       float(folds["mae"].mean()),
        "best_iter_mean": float(folds["best_iter"].mean()),
        "n_splits":       int(N_SPLITS),
        "n_obs":          int(len(X)),
    }

    print("\nBest params:", best["params"])
    print("Best mean RMSE_log:", best["rmse_log"])
    return best["params"], summary, folds


# -----------------------------------------------------------------------------
# 9) RUN BOTH SPECIFICATIONS
# -----------------------------------------------------------------------------
params_std, summary_std, folds_std = tune_timeseries_lgbm(df, DROP_STANDARD, "STANDARD (no windows)")
params_ext, summary_ext, folds_ext = tune_timeseries_lgbm(df, DROP_EXTENDED, "EXTENDED (with windows)")

final_table = pd.DataFrame([summary_std, summary_ext])
print("\nFINAL RESULTS (LightGBM):")
print(final_table.to_string(index=False))


# -----------------------------------------------------------------------------
# 10) SAVE OUTPUTS
# -----------------------------------------------------------------------------
final_table.to_csv("final_comparable_table_tss_lgbm.csv", index=False)
folds_std.to_csv("folds_standard_tss_lgbm.csv", index=False)
folds_ext.to_csv("folds_extended_tss_lgbm.csv", index=False)
pd.Series(params_std).to_csv("best_params_standard_tss_lgbm.csv")
pd.Series(params_ext).to_csv("best_params_extended_tss_lgbm.csv")
