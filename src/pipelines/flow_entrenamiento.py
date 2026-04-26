"""
Flow de Prefect para entrenamiento del modelo de crédito.
Conecta preprocesamiento → entrenamiento → evaluación → MLflow.

"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
from prefect import flow, task, get_run_logger
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from dotenv import load_dotenv

from config.mlflow_config import (
    setup_mlflow, MLflowConfig, TRACKING_URI, EXPERIMENTS,
    log_credit_metrics, log_dataset_info,
)
from config.settings import (
    TARGET, NUM_COLS, CAT_COLS,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS,
    LOGISTIC_REGRESSION_PARAMS, SVM_PARAMS,DEFAULT_MODEL
)
from src.data.preprocesamiento import load_and_clean
from src.features.transformers import build_model_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

load_dotenv()

# ── Modelos disponibles ───────────────────────────────────
MODELS = {
    "random_forest":       RandomForestClassifier(**RANDOM_FOREST_PARAMS),
    "xgboost":             XGBClassifier(**XGBOOST_PARAMS),
    "logistic_regression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
    "svm":                 SVC(**SVM_PARAMS),
}

DATA_PATH_DEFAULT = os.getenv("DATA_PATH", "data/loan_approval_dataset.csv")


# ── Tasks ─────────────────────────────────────────────────

@task(name="Cargar y limpiar datos", retries=2, retry_delay_seconds=5)
def task_cargar_datos(data_path: str):
    logger = get_run_logger()
    logger.info(f" Cargando datos desde: {data_path}")
    df, le = load_and_clean(data_path)
    logger.info(f" Datos cargados: {df.shape}")
    return df, le


@task(name="Split train/test")
def task_split(df):
    logger = get_run_logger()
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f" Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


@task(name="Entrenar modelo")
def task_entrenar(model_name: str, X_train, y_train):
    logger = get_run_logger()
    logger.info(f" Entrenando: {model_name}")

    model    = MODELS[model_name]
    pipeline = build_model_pipeline(model)

    # Cross validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    logger.info(
        f" CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )

    # Entrenamiento final
    pipeline.fit(X_train, y_train)
    return pipeline, cv_scores


@task(name="Evaluar modelo")
def task_evaluar(model_name: str, pipeline, X_test, y_test):
    logger = get_run_logger()
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    logger.info(f" Evaluación completada para {model_name}")
    return y_pred, y_prob


@task(name="Loggear en MLflow")
def task_loggear_mlflow(
    model_name: str,
    pipeline,
    cv_scores,
    y_test,
    y_pred,
    y_prob,
    df,
):
    logger = get_run_logger()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTS["baseline"])

    with mlflow.start_run(run_name=f"{model_name}_flow", nested=True):
        mlflow.set_tags({
            "model_type": model_name,
            "phase":      "training_pipeline",
            "project":    "credit_approval",
            "status":     "SUCCESS",
        })

        # Dataset info
        log_dataset_info(df, dataset_name="loan_approval")

        # Params del modelo
        mlflow.log_params({
            f"model_{k}": v
            for k, v in pipeline.named_steps["model"].get_params().items()
        })

        # CV metrics
        mlflow.log_metrics({
            "cv_roc_auc_mean": cv_scores.mean(),
            "cv_roc_auc_std":  cv_scores.std(),
        })

        # Test metrics
        metrics = log_credit_metrics(y_test, y_pred, y_prob, prefix="test")

        # Log modelo
        mlflow.sklearn.log_model(pipeline, artifact_path=f"model_{model_name}")

        run_id = mlflow.active_run().info.run_id

    logger.info(
        f" {model_name} loggeado — "
        f"ROC AUC: {metrics.get('test_roc_auc', 0):.4f}"
    )
    return run_id, metrics


# ── Flow principal ────────────────────────────────────────

@flow(
    name="entrenamiento-aprobación-crédito",
    description="Pipeline de entrenamiento para aprobación de crédito",
    log_prints=True,
)
def training_flow(
    data_path: str = DATA_PATH_DEFAULT,
    model_name: str = DEFAULT_MODEL,
):
    logger = get_run_logger()
    logger.info(f" Iniciando training flow — modelo: {model_name}")

    if model_name not in MODELS:
        raise ValueError(
            f"Modelo '{model_name}' no disponible. "
            f"Opciones: {list(MODELS.keys())}"
        )

    # Setup MLflow
    setup_mlflow(MLflowConfig())

    # Ejecutar tasks
    df, _                            = task_cargar_datos(data_path)
    X_train, X_test, y_train, y_test = task_split(df)
    pipeline, cv_scores              = task_entrenar(model_name, X_train, y_train)
    y_pred, y_prob                   = task_evaluar(model_name, pipeline, X_test, y_test)
    run_id, metrics                  = task_loggear_mlflow(
                                            model_name, pipeline, cv_scores,
                                            y_test, y_pred, y_prob, df,
                                        )

    logger.info(f" Flow completado — Run ID: {run_id}")
    return run_id, metrics


if __name__ == "__main__":
    training_flow(model_name=DEFAULT_MODEL)