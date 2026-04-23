"""
────────────────────────────────────────────────────────────
Configuración central de MLflow para el proyecto.
Crea experimentos, tags estándar y utilidades de logging.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────
TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Aprobación creditos")

EXPERIMENTS = {
    "baseline":    f"{EXPERIMENT_NAME}/baseline",
    "feature_eng": f"{EXPERIMENT_NAME}/feature_engineering",
    "hyperopt":    f"{EXPERIMENT_NAME}/hyperparameter_tuning",
    "final":       f"{EXPERIMENT_NAME}/final_models",
}

DEFAULT_TAGS = {
    "project": "Aprobación créditos",
    "team":    "mlops",
    "task":    "binary_classification",
}


# ── Config dataclass ──────────────────────────────────────
@dataclass
class MLflowConfig:
    tracking_uri:    str  = TRACKING_URI
    experiment_name: str  = EXPERIMENT_NAME
    tags:            dict = field(default_factory=lambda: DEFAULT_TAGS.copy())


# ── Setup de experimentos ─────────────────────────────────
def setup_mlflow(config: MLflowConfig = None) -> MlflowClient:
    """
    Conecta con el tracking server y crea todos los
    experimentos si no existen.
    """
    if config is None:
        config = MLflowConfig()

    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient(config.tracking_uri)
    logger.info(f"MLflow Tracking URI: {config.tracking_uri}")

    for key, exp_name in EXPERIMENTS.items():
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            exp_id = client.create_experiment(exp_name)
            logger.info(f" Experimento creado: '{exp_name}' (id={exp_id})")
        else:
            logger.info(f" Ya existe: '{exp_name}' (id={exp.experiment_id})")

    return client


# ── Context manager para runs ─────────────────────────────
class MLflowRun:
    """
    Inicia un run con autologging y tags estándar.

    Uso:
        with MLflowRun("baseline", run_name="rf_v1") as run:
            model.fit(X_train, y_train)
    """
    def __init__(
        self,
        experiment_key: str = "baseline",
        run_name: Optional[str] = None,
        tags: Optional[dict] = None,
    ):
        exp_name = EXPERIMENTS.get(experiment_key, EXPERIMENT_NAME)
        mlflow.set_experiment(exp_name)
        self.run_name   = run_name
        self.extra_tags = {**DEFAULT_TAGS, **(tags or {})}
        self.run        = None

    def __enter__(self):
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True,
            silent=True,
        )
        mlflow.xgboost.autolog(silent=True)

        self.run = mlflow.start_run(
            run_name=self.run_name,
            tags=self.extra_tags,
        )
        logger.info(f" Run iniciado: {self.run.info.run_id}")
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc_val))
            logger.error(f" Run fallido: {exc_val}")
        else:
            mlflow.set_tag("status", "SUCCESS")
        mlflow.end_run()
        return False


# ── Helpers de logging ────────────────────────────────────
def log_credit_metrics(y_true, y_pred, y_prob=None, prefix: str = "") -> dict:
    """Loggea métricas estándar para clasificación de crédito."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        matthews_corrcoef,
    )
    p = f"{prefix}_" if prefix else ""

    metrics = {
        f"{p}accuracy":  accuracy_score(y_true, y_pred),
        f"{p}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{p}recall":    recall_score(y_true, y_pred, zero_division=0),
        f"{p}f1":        f1_score(y_true, y_pred, zero_division=0),
        f"{p}mcc":       matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        metrics[f"{p}roc_auc"]       = roc_auc_score(y_true, y_prob)
        metrics[f"{p}avg_precision"] = average_precision_score(y_true, y_prob)

    mlflow.log_metrics(metrics)
    return metrics


def log_dataset_info(df, dataset_name: str = "credit_data") -> None:
    """Loggea metadatos del dataset como parámetros del run."""
    mlflow.log_params({
        "dataset_name":   dataset_name,
        "dataset_rows":   len(df),
        "dataset_cols":   len(df.columns),
        "target_balance": round(df.iloc[:, -1].value_counts(normalize=True).iloc[0], 4),
    })