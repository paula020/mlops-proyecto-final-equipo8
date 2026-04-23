"""
Flow maestro que conecta todo el pipeline end-to-end:
ETL → Validación → Feature Engineering → Entrenamiento
→ Evaluación → MLflow Registry
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from dotenv import load_dotenv

from config.mlflow_config import (
    setup_mlflow, MLflowConfig, TRACKING_URI, EXPERIMENTS,
    log_credit_metrics, log_dataset_info,
)
from config.settings import (
    TARGET, NUM_COLS, CAT_COLS,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    DEFAULT_MODEL,
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS,
    LOGISTIC_REGRESSION_PARAMS, SVM_PARAMS,
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
)
from src.data.etl import run_etl
from src.features.transformers import build_model_pipeline

load_dotenv()

MODELS = {
    "random_forest":       RandomForestClassifier(**RANDOM_FOREST_PARAMS),
    "xgboost":             XGBClassifier(**XGBOOST_PARAMS),
    "logistic_regression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
    "svm":                 SVC(**SVM_PARAMS),
}


# ── Tasks ─────────────────────────────────────────────────

@task(name="ETL — Extracción y limpieza", retries=2, retry_delay_seconds=5)
def task_etl(data_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(" Iniciando ETL")
    df = run_etl(data_path=data_path, log_mlflow=False)
    logger.info(f" ETL completado — {df.shape[0]} filas")
    return df


@task(name="Validación de datos")
def task_validar(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()

    required_cols = NUM_COLS + CAT_COLS + [TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

    nulls = df[required_cols].isnull().sum().sum()
    if nulls > 0:
        raise ValueError(f"Se encontraron {nulls} valores nulos")

    for col in NUM_COLS:
        if (df[col] < 0).any():
            raise ValueError(f"Columna {col} tiene valores negativos")

    logger.info(f" Validación exitosa — {df.shape[0]} filas, sin nulos ni negativos")
    return df


@task(name="Feature Engineering — Split")
def task_features(df: pd.DataFrame) -> tuple:
    logger = get_run_logger()

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f" Split — Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


@task(name="Entrenamiento con CV")
def task_entrenar(model_name: str, X_train, y_train) -> tuple:
    logger = get_run_logger()
    logger.info(f"🔬 Entrenando: {model_name}")

    pipeline = build_model_pipeline(MODELS[model_name])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    logger.info(f" CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_train, y_train)
    return pipeline, cv_scores


@task(name="Evaluación del modelo")
def task_evaluar(model_name: str, pipeline, X_test, y_test) -> tuple:
    logger = get_run_logger()

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score, f1_score
    roc_auc = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)

    logger.info(f" ROC AUC: {roc_auc:.4f} | F1: {f1:.4f}")
    return y_pred, y_prob


@task(name="Logging y registro en MLflow")
def task_mlflow_y_registry(
    model_name: str,
    pipeline,
    cv_scores,
    y_test,
    y_pred,
    y_prob,
    df: pd.DataFrame,
) -> str:
    logger = get_run_logger()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTS["final"])

    with mlflow.start_run(
        run_name=f"{model_name}_pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}",
        nested=True,
    ):
        mlflow.set_tags({
            "model_type": model_name,
            "phase":      "model_pipeline",
            "project":    "Aprobación de créditos",
            "status":     "SUCCESS",
        })

        # Dataset info
        log_dataset_info(df, dataset_name="loan_approval")

        # Params
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
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=f"model_{model_name}",
        )

        run_id = mlflow.active_run().info.run_id

    logger.info(
        f" Run loggeado — "
        f"ROC AUC: {metrics.get('test_roc_auc', 0):.4f} | "
        f"Run ID: {run_id}"
    )
    return run_id


# ── Flow maestro ──────────────────────────────────────────

@flow(
    name="pipeline-principal-Aprobación-créditos",
    description="Pipeline end-to-end: ETL → features → entrenamiento → evaluación → MLflow",
    log_prints=True,
)
def main_pipeline(
    data_path: str = str(RAW_DATA_PATH),
    model_name: str = DEFAULT_MODEL,
):
    logger = get_run_logger()
    logger.info(f" Iniciando pipeline principal — modelo: {model_name}")

    # Setup MLflow
    setup_mlflow(MLflowConfig())

    # Pipeline end-to-end
    df                               = task_etl(data_path)
    df                               = task_validar(df)
    X_train, X_test, y_train, y_test = task_features(df)
    pipeline, cv_scores              = task_entrenar(model_name, X_train, y_train)
    y_pred, y_prob                   = task_evaluar(model_name, pipeline, X_test, y_test)
    run_id                           = task_mlflow_y_registry(
                                           model_name, pipeline, cv_scores,
                                           y_test, y_pred, y_prob, df,
                                       )

    logger.info(f"Pipeline completado — Run ID: {run_id}")
    return run_id


if __name__ == "__main__":
    main_pipeline()