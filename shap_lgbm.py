# =============================================================================
# Windowing Strategies and Box Office Prediction
# SHAP explainability analysis for LightGBM (EXTENDED specification)
#
# Trains the EXTENDED LightGBM model on the full dataset and computes
# SHAP values on a random subsample. Exports summary (beeswarm + bar)
# and dependence plots for the theatrical-to-digital window variable.
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
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
RANDOM_STATE     = 42
N_JOBS           = -1
SHAP_SAMPLE_SIZE = 2000
OUTPUT_DIR       = "shap_lgbm_extended"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
DROP_EXTENDED = COMMON_DROP

# Best hyperparameters for EXTENDED specification
BEST_PARAMS_EXTENDED = {
    "num_leaves": 31, "num_boost_round": 8000, "min_data_in_leaf": 20,
    "max_depth": -1, "learning_rate": 0.08, "lambda_l2": 1.0, "lambda_l1": 0.0,
    "feature_fraction": 0.7, "bagging_freq": 0, "bagging_fraction": 0.7,
}


# -----------------------------------------------------------------------------
# 1) HELPERS
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


def assert_no_datetime(X: pd.DataFrame):
    dt_cols = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    if dt_cols:
        raise ValueError(f"Datetime columns still present in X: {dt_cols}")


def get_feature_names_from_ct(ct):
    output_features = []
    for name, transformer, cols in ct.transformers_:
        if name == "remainder":
            continue
        if transformer == "passthrough":
            output_features.extend(list(cols))
            continue
        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                try:
                    feats = last_step.get_feature_names_out(cols)
                except Exception:
                    feats = last_step.get_feature_names_out()
                output_features.extend(list(feats))
            else:
                output_features.extend(list(cols))
        else:
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    feats = transformer.get_feature_names_out(cols)
                except Exception:
                    feats = transformer.get_feature_names_out()
                output_features.extend(list(feats))
            else:
                output_features.extend(list(cols))
    return output_features


# -----------------------------------------------------------------------------
# 2) LOAD AND PREPARE DATA
# -----------------------------------------------------------------------------
df_raw = pd.read_excel(DATA_PATH, engine="openpyxl")
for col in [TARGET, TIME_COL]:
    if col not in df_raw.columns:
        raise ValueError(f"Missing required column: {col}")

df = prepare_df(df_raw)
df = df.sort_values(TIME_COL).reset_index(drop=True)

y     = df[TARGET].fillna(0.0).astype(float).values
y_log = np.log1p(y)

X = df.drop(columns=DROP_EXTENDED, errors="ignore")
assert_no_datetime(X)


# -----------------------------------------------------------------------------
# 3) PREPROCESSING
# -----------------------------------------------------------------------------
preprocessor  = build_preprocessor(X)
X_t           = preprocessor.fit_transform(X)
feature_names = get_feature_names_from_ct(preprocessor)

print(f"X shape (original):    {X.shape}")
print(f"X shape (transformed): {X_t.shape}")
print(f"Number of features:    {len(feature_names)}")

X_t_dense = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
X_model   = pd.DataFrame(X_t_dense, columns=feature_names, index=X.index)


# -----------------------------------------------------------------------------
# 4) TRAIN LIGHTGBM
# -----------------------------------------------------------------------------
core_params = {
    "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
    "seed": RANDOM_STATE, "verbosity": -1,
    "num_threads": 0 if N_JOBS == -1 else int(N_JOBS),
    **{k: v for k, v in BEST_PARAMS_EXTENDED.items() if k != "num_boost_round"},
}

dtrain = lgb.Dataset(X_model, label=y_log, free_raw_data=False)
model  = lgb.train(params=core_params, train_set=dtrain,
                   num_boost_round=int(BEST_PARAMS_EXTENDED["num_boost_round"]))


# -----------------------------------------------------------------------------
# 5) SHAP SUBSAMPLE
# -----------------------------------------------------------------------------
X_shap = (X_model.sample(SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE)
          if len(X_model) > SHAP_SAMPLE_SIZE else X_model.copy())

print(f"SHAP sample shape: {X_shap.shape}")


# -----------------------------------------------------------------------------
# 6) SHAP VALUES
# -----------------------------------------------------------------------------
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

if isinstance(shap_values, list):
    shap_values = shap_values[0]


# -----------------------------------------------------------------------------
# 7) GLOBAL IMPORTANCE TABLE
# -----------------------------------------------------------------------------
shap_importance = (
    pd.DataFrame({
        "feature":       X_shap.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    })
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)

shap_importance.to_csv(os.path.join(OUTPUT_DIR, "shap_importance_lgbm_extended.csv"), index=False)

print("\nTop 20 features by mean |SHAP|:")
print(shap_importance.head(20).to_string(index=False))


# -----------------------------------------------------------------------------
# 8) BEESWARM SUMMARY PLOT
# -----------------------------------------------------------------------------
plt.figure()
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_beeswarm_lgbm_extended.png"),
            dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------------
# 9) BAR SUMMARY PLOT
# -----------------------------------------------------------------------------
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar_lgbm_extended.png"),
            dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------------
# 10) DEPENDENCE PLOT (theatrical-to-digital window)
# -----------------------------------------------------------------------------
target_feature = "R.Dig-theatr"

if target_feature in X_shap.columns:
    plt.figure()
    shap.dependence_plot(target_feature, shap_values, X_shap,
                         show=False, interaction_index=None)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_R_Dig_theatr.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nDependence plot saved for: {target_feature}")
else:
    print(f"\nFeature '{target_feature}' not found after preprocessing.")

print(f"\nDone. Outputs saved in: {OUTPUT_DIR}/")
