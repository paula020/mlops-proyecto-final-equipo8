"""
────────────────────────────────────────────────────────────
Flow de Prefect para preprocesamiento del dataset de crédito.
Cada paso es una Task independiente — reintentable y loggeable.
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.model_selection import train_test_split
from prefect import flow, task, get_run_logger

import mlflow
from config.mlflow_config import setup_mlflow, MLflowConfig, TRACKING_URI
from config.settings import (
    TARGET, NUM_COLS, CAT_COLS,
    TEST_SIZE, RANDOM_STATE,
)
from src.data.preprocesamiento import load_and_clean
from src.features.transformers import build_preprocessor
from dotenv import load_dotenv

load_dotenv()


# ── Tasks ─────────────────────────────────────────────────

@task(name="Cargar dataset", retries=2, retry_delay_seconds=5)
def task_load_data(data_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f" Cargando dataset desde: {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

    df, _ = load_and_clean(data_path)
    logger.info(f" Dataset cargado: {df.shape}")
    return df


@task(name="Validar datos")
def task_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()

    # Validar columnas requeridas
    required_cols = NUM_COLS + CAT_COLS + [TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

    # Validar que no haya nulos
    nulls = df[required_cols].isnull().sum()
    cols_with_nulls = nulls[nulls > 0]
    if not cols_with_nulls.empty:
        raise ValueError(f"Columnas con nulos: {cols_with_nulls.to_dict()}")

    # Validar que no haya negativos después de limpieza
    for col in NUM_COLS:
        if (df[col] < 0).any():
            raise ValueError(f"Columna {col} aún tiene valores negativos")

    logger.info(f" Validación exitosa — {df.shape[0]} filas, sin nulos ni negativos")
    return df


@task(name="Split train/test")
def task_split_data(df: pd.DataFrame) -> tuple:
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
    logger.info(
        f"Balance train: {y_train.value_counts(normalize=True).round(3).to_dict()}"
    )
    return X_train, X_test, y_train, y_test


@task(name="Log info en MLflow")
def task_log_mlflow(df: pd.DataFrame, X_train, X_test) -> None:
    logger = get_run_logger()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("credit_approval/baseline")

    with mlflow.start_run(run_name="preprocessing_flow", nested=True):
        mlflow.log_params({
            "dataset_rows":  len(df),
            "dataset_cols":  len(df.columns),
            "train_rows":    len(X_train),
            "test_rows":     len(X_test),
            "test_size":     TEST_SIZE,
            "random_state":  RANDOM_STATE,
            "num_features":  len(NUM_COLS),
            "cat_features":  len(CAT_COLS),
        })
        mlflow.log_metric(
            "target_balance",
            round(df[TARGET].value_counts(normalize=True).iloc[0], 4),
        )

    logger.info(" Info loggeada en MLflow")


# ── Flow principal ────────────────────────────────────────
DATA_PATH_DEFAULT = os.getenv("DATA_PATH", "data/loan_approval_dataset.csv")

@flow(
    name="Preprocesamiento aprobación de créditos",
    description="Preprocesamiento del dataset de aprobación de crédito",
    log_prints=True,
)
def preprocessing_flow(data_path: str = DATA_PATH_DEFAULT):
    # Ejecutar tasks en secuencia
    df                              = task_load_data(data_path)
    df                              = task_validate_data(df)
    X_train, X_test, y_train, y_test = task_split_data(df)
    task_log_mlflow(df, X_train, X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessing_flow()