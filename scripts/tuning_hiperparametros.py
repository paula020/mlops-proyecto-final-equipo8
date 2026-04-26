"""
────────────────────────────────────────────────────────────
Búsqueda de hiperparámetros con Optuna para los 4 modelos.
Cada trial se loggea como un run en MLflow.

"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import optuna
from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

import mlflow
from config.mlflow_config import (
    setup_mlflow, MLflowConfig,
    log_credit_metrics, log_dataset_info,
    EXPERIMENTS,
)
from config.settings import (
    TARGET, DROP_COLS, CAT_COLS, NUM_COLS,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    RANDOM_FOREST_SEARCH_SPACE, XGBOOST_SEARCH_SPACE,
    LOGISTIC_REGRESSION_SEARCH_SPACE, SVM_SEARCH_SPACE,
    OPTUNA_N_TRIALS,
)

load_dotenv()
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS),
    ])


def load_data(data_path: str):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    for col in CAT_COLS + [TARGET]:
        df[col] = df[col].str.strip()
    df = df.drop(columns=DROP_COLS, errors="ignore")
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET])
    return df


# ── Objective functions por modelo ───────────────────────

def objective_rf(trial, X_train, y_train):
    sp = RANDOM_FOREST_SEARCH_SPACE
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", *sp["n_estimators"]),
        "max_depth":         trial.suggest_int("max_depth", *sp["max_depth"]),
        "min_samples_split": trial.suggest_int("min_samples_split", *sp["min_samples_split"]),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", *sp["min_samples_leaf"]),
        "max_features":      trial.suggest_categorical("max_features", sp["max_features"]),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    }
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(**params)),
    ])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def objective_xgb(trial, X_train, y_train):
    sp = XGBOOST_SEARCH_SPACE
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", *sp["n_estimators"]),
        "max_depth":        trial.suggest_int("max_depth", *sp["max_depth"]),
        "learning_rate":    trial.suggest_float("learning_rate", *sp["learning_rate"]),
        "subsample":        trial.suggest_float("subsample", *sp["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *sp["colsample_bytree"]),
        "reg_alpha":        trial.suggest_float("reg_alpha", *sp["reg_alpha"]),
        "reg_lambda":       trial.suggest_float("reg_lambda", *sp["reg_lambda"]),
        "random_state":     RANDOM_STATE,
        "eval_metric":      "logloss",
        "verbosity":        0,
    }
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", XGBClassifier(**params)),
    ])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def objective_lr(trial, X_train, y_train):
    sp = LOGISTIC_REGRESSION_SEARCH_SPACE
    penalty = trial.suggest_categorical("penalty", sp["penalty"])
    solver  = trial.suggest_categorical("solver", sp["solver"])

    # liblinear no soporta todos los solvers con l2 a gran escala
    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        raise optuna.TrialPruned()

    params = {
        "C":            trial.suggest_float("C", *sp["C"], log=True),
        "penalty":      penalty,
        "solver":       solver,
        "max_iter":     1000,
        "random_state": RANDOM_STATE,
    }
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LogisticRegression(**params)),
    ])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def objective_svm(trial, X_train, y_train):
    sp = SVM_SEARCH_SPACE
    params = {
        "C":            trial.suggest_float("C", *sp["C"], log=True),
        "gamma":        trial.suggest_categorical("gamma", sp["gamma"]),
        "kernel":       trial.suggest_categorical("kernel", sp["kernel"]),
        "probability":  True,
        "random_state": RANDOM_STATE,
    }
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", SVC(**params)),
    ])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ── Tuning + logging en MLflow ────────────────────────────

def tune_model(model_name, objective_fn, X_train, X_test, y_train, y_test, df):
    """Corre Optuna y loggea el mejor trial en MLflow."""

    logger.info(f"\n🔍 Tuning: {model_name} ({OPTUNA_N_TRIALS} trials)")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_tuning",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train),
        n_trials=OPTUNA_N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info(f"   Mejor ROC AUC CV: {best.value:.4f}")
    logger.info(f"  Params: {best.params}")

    # Loggear mejor trial en MLflow
    mlflow.set_tracking_uri(mlflow.get_tracking_uri())
    mlflow.set_experiment(EXPERIMENTS["hyperopt"])

    with mlflow.start_run(run_name=f"{model_name}_best"):
        mlflow.set_tags({
            "model_type": model_name,
            "phase":      "hyperopt",
            "project":    "credit_approval",
            "status":     "SUCCESS",
        })

        # Log info dataset
        log_dataset_info(df, dataset_name="loan_approval")

        # Log mejores params
        mlflow.log_params(best.params)
        mlflow.log_metric("cv_roc_auc_best", best.value)
        mlflow.log_metric("n_trials", OPTUNA_N_TRIALS)

        # Reentrenar con mejores params en todo el train set
        best_model = _build_best_model(model_name, best.params)
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", best_model),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = log_credit_metrics(y_test, y_pred, y_prob, prefix="test")

        mlflow.sklearn.log_model(pipeline, artifact_path=f"model_{model_name}")

    return best.params, test_metrics


def _build_best_model(model_name: str, params: dict):
    """Construye el modelo con los mejores hiperparámetros encontrados."""
    if model_name == "random_forest":
        return RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "xgboost":
        return XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0)
    elif model_name == "logistic_regression":
        return LogisticRegression(**params, max_iter=1000, random_state=RANDOM_STATE)
    elif model_name == "svm":
        return SVC(**params, probability=True, random_state=RANDOM_STATE)


# ── Main ──────────────────────────────────────────────────

def main():
    import os
    data_path = os.getenv("DATA_PATH", "data/loan_approval_dataset.csv")

    logger.info("=" * 55)
    logger.info("  Hyperparameter Tuning — Credit Approval MLOps")
    logger.info("=" * 55)

    setup_mlflow(MLflowConfig())

    df = load_data(data_path)
    X  = df[NUM_COLS + CAT_COLS]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    objectives = {
        "random_forest":       objective_rf,
        "xgboost":             objective_xgb,
        "logistic_regression": objective_lr,
        "svm":                 objective_svm,
    }

    results = {}
    for model_name, objective_fn in objectives.items():
        best_params, test_metrics = tune_model(
            model_name, objective_fn,
            X_train, X_test, y_train, y_test, df,
        )
        results[model_name] = test_metrics

    # Resumen final
    logger.info("\n" + "=" * 55)
    logger.info("  RESUMEN TUNING")
    logger.info("=" * 55)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("test_roc_auc", 0),
        reverse=True,
    )
    for name, metrics in sorted_results:
        logger.info(
            f"  {name:<25} "
            f"ROC AUC: {metrics.get('test_roc_auc', 0):.4f} | "
            f"F1: {metrics.get('test_f1', 0):.4f}"
        )
    logger.info("=" * 55)
    logger.info("   Revisa los runs en http://localhost:5000")


if __name__ == "__main__":
    main()