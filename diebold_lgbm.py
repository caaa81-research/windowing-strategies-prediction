# =============================================================================
# Windowing Strategies and Box Office Prediction
# Diebold-Mariano test for LightGBM (STANDARD vs EXTENDED)
#
# Trains both model specifications with the best tuned parameters,
# generates out-of-sample predictions via forward-chaining TimeSeriesSplit,
# and applies the Diebold-Mariano test using squared error loss on the
# log1p scale with a HAC variance estimator.
# =============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MultiLabelBinarizer
from scipy import stats


# -----------------------------------------------------------------------------
# 0) CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH  = "DB.xlsx"
TARGET     = "boxOfficeWorldwide"
TIME_COL   = "startYear"

RANDOM_STATE          = 42
N_JOBS                = -1
N_SPLITS              = 5
EARLY_STOPPING_ROUNDS = 200

MIN_FREQ_GENRES   = 25
MIN_FREQ_COUNTRY  = 25
MIN_FREQ_LANGUAGE = 25
TOP_K_DIRECTORS   = 10
TOP_K_WRITERS     = 10
TOP_K_ACTORS      = 10

COMMON_DROP = [
    "tconst", "originalTitle", TARGET,
    "boxOfficeDomestic", "boxOfficeInternational",
    "nextlix_original", "productionMethod", "distributor", "production",
    "RL.Theatrical", "RL.Premiere", "RL.Digital", "RL.Physical", "RL.TV",
]
WINDOW_VARS   = ["R.Dig-theatr", "R.Phys-theatr", "R.TV-theatr"]
DROP_STANDARD = COMMON_DROP + WINDOW_VARS
DROP_EXTENDED = COMMON_DROP

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
# 1) FEATURE ENGINEERING HELPERS
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
# 2) DIEBOLD-MARIANO TEST
# -----------------------------------------------------------------------------
def dm_test(y_true, y_pred_1, y_pred_2, h=1, loss="se", alternative="two-sided", nw_lag=None):
    """
    Harvey, Leybourne & Newbold (1997) corrected Diebold-Mariano test.

    Parameters
    ----------
    y_true, y_pred_1, y_pred_2 : array-like
    h           : forecast horizon (default 1)
    loss        : 'se' (squared error) or 'ae' (absolute error)
    alternative : 'two-sided', 'less', or 'greater'
    nw_lag      : Newey-West lag truncation (default max(h-1, 0))

    Returns
    -------
    dict with test statistic, p-value, and mean loss differential.
    Positive mean_loss_diff => model 1 has higher loss (model 2 is better).
    """
    y_true   = np.asarray(y_true,   dtype=float)
    y_pred_1 = np.asarray(y_pred_1, dtype=float)
    y_pred_2 = np.asarray(y_pred_2, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred_1) & np.isfinite(y_pred_2)
    y_true, y_pred_1, y_pred_2 = y_true[mask], y_pred_1[mask], y_pred_2[mask]
    T = len(y_true)
    if T < 5:
        raise ValueError(f"Too few aligned observations for DM test (T={T}).")

    e1 = y_true - y_pred_1
    e2 = y_true - y_pred_2
    d  = (e1 ** 2 - e2 ** 2) if loss == "se" else (np.abs(e1) - np.abs(e2))

    dbar   = d.mean()
    nw_lag = max(h - 1, 0) if nw_lag is None else nw_lag

    def autocov(x, k):
        mu = x.mean()
        return np.mean((x[k:] - mu) * (x[:-k] - mu)) if k > 0 else np.mean((x - mu) ** 2)

    var_d = autocov(d, 0)
    for k in range(1, nw_lag + 1):
        var_d += 2.0 * (1.0 - k / (nw_lag + 1.0)) * autocov(d, k)

    var_dbar = var_d / T
    if var_dbar <= 0 or not np.isfinite(var_dbar):
        raise ValueError("Non-positive HAC variance; check data and nw_lag.")

    dm        = dbar / np.sqrt(var_dbar)
    factor    = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_harvey = dm * factor

    dfree = T - 1
    if alternative == "two-sided":
        p = 2 * stats.t.sf(np.abs(dm_harvey), df=dfree)
    elif alternative == "less":
        p = stats.t.cdf(dm_harvey, df=dfree)
    else:
        p = stats.t.sf(dm_harvey, df=dfree)

    return {
        "T": int(T), "h": int(h), "loss": loss, "nw_lag": int(nw_lag),
        "dm_stat": float(dm_harvey), "p_value": float(p),
        "mean_loss_diff": float(dbar),
    }


# -----------------------------------------------------------------------------
# 3) DATA PREPARATION
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# 4) OUT-OF-SAMPLE PREDICTIONS (forward-chaining)
# -----------------------------------------------------------------------------
def oos_predict_lgbm(df_in: pd.DataFrame, drop_cols: list, best_params: dict, label: str):
    df_sorted = df_in.sort_values([TIME_COL]).reset_index(drop=True)
    y = df_sorted[TARGET].fillna(0.0).astype(float).values

    X = df_sorted.drop(columns=drop_cols, errors="ignore")
    assert_no_datetime(X, label)

    pred_log_oos = np.full(len(df_sorted), np.nan)
    tss = TimeSeriesSplit(n_splits=N_SPLITS)

    core_params = {
        "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
        "seed": RANDOM_STATE, "verbosity": -1,
        "num_threads": 0 if N_JOBS == -1 else int(N_JOBS),
    }
    train_params = core_params.copy()
    train_params.update({k: v for k, v in best_params.items() if k != "num_boost_round"})
    num_boost_round = int(best_params["num_boost_round"])

    for fold, (tr, te) in enumerate(tss.split(X), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr_log   = np.log1p(y[tr])
        y_te_log   = np.log1p(y[te])

        prep   = build_preprocessor(X_tr)
        X_tr_t = prep.fit_transform(X_tr)
        X_te_t = prep.transform(X_te)

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
        pred_log_oos[te] = bst.predict(X_te_t, num_iteration=best_iter)

        rmse_fold = np.sqrt(np.mean((y_te_log - pred_log_oos[te]) ** 2))
        print(f"[{label}] fold {fold}/{N_SPLITS} | best_iter={best_iter} | RMSE_log={rmse_fold:.5f}")

    return df_sorted, np.log1p(y), pred_log_oos


# -----------------------------------------------------------------------------
# 5) RUN
# -----------------------------------------------------------------------------
df_raw = pd.read_excel(DATA_PATH, engine="openpyxl")
for col in [TARGET, TIME_COL]:
    if col not in df_raw.columns:
        raise ValueError(f"Missing required column: {col}")

df_feat = prepare_df(df_raw)

df_sorted, y_true_log, pred_std_log = oos_predict_lgbm(
    df_feat, DROP_STANDARD, BEST_PARAMS_STANDARD, label="STANDARD"
)
_, _, pred_ext_log = oos_predict_lgbm(
    df_feat, DROP_EXTENDED, BEST_PARAMS_EXTENDED, label="EXTENDED"
)

result = dm_test(
    y_true=y_true_log, y_pred_1=pred_std_log, y_pred_2=pred_ext_log,
    h=1, loss="se", alternative="two-sided",
)

print("\nDiebold-Mariano test (LightGBM | squared error on log1p scale):")
print(pd.Series(result).to_string())

direction = "EXTENDED is more accurate (lower loss) than STANDARD." \
    if result["mean_loss_diff"] > 0 else \
    "STANDARD is more accurate (lower loss) than EXTENDED."
print(f"\nInterpretation: {direction}")
