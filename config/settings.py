"""
Configuración central del proyecto. Modifica este archivo
para cambiar hiperparámetros, splits y columnas sin tocar
el código de los scripts.
"""

# ── Data ──────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Columnas ──────────────────────────────────────────────
TARGET    = "loan_status"
DROP_COLS = ["loan_id"]
CAT_COLS  = ["education", "self_employed"]
NUM_COLS  = [
    "no_of_dependents", "income_annum", "loan_amount",
    "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
]

# ── Hiperparámetros modelos ──────────────────────────────
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth":    8,
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
}

XGBOOST_PARAMS = {
    "n_estimators":  100,
    "max_depth":     6,
    "learning_rate": 0.1,
    "random_state":  RANDOM_STATE,
    "eval_metric":   "logloss",
    "verbosity":     0,
}

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter":     1000,
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
}

SVM_PARAMS = {
    "kernel":      "rbf",
    "probability": True,
    "random_state": RANDOM_STATE,
}

# ── Optuna ───────────────────────────
RANDOM_FOREST_SEARCH_SPACE = {
    "n_estimators":      (50, 300),
    "max_depth":         (3, 15),
    "min_samples_split": (2, 10),
    "min_samples_leaf":  (1, 5),
    "max_features":      ["sqrt", "log2"],
}

XGBOOST_SEARCH_SPACE = {
    "n_estimators":  (50, 300),
    "max_depth":     (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample":     (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha":     (0.0, 1.0),
    "reg_lambda":    (0.5, 2.0),
}

LOGISTIC_REGRESSION_SEARCH_SPACE = {
    "C":       (0.01, 10.0),
    "penalty": ["l1", "l2"],
    "solver":  ["liblinear", "saga"],
}

SVM_SEARCH_SPACE = {
    "C":     (0.1, 10.0),
    "gamma": ["scale", "auto"],
    "kernel": ["rbf", "poly"],
}

# Número de trials por modelo
OPTUNA_N_TRIALS = 30