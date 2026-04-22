"""
Experimentos iniciales: compara Random Forest, XGBoost,
SVM y Logistic Regression registrando todo en MLflow.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

import mlflow
from config.mlflow_config import (
    setup_mlflow, MLflowRun, MLflowConfig,
    log_credit_metrics, log_dataset_info,
)

from config.settings import (
    TARGET, DROP_COLS, CAT_COLS, NUM_COLS,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS,
    LOGISTIC_REGRESSION_PARAMS, SVM_PARAMS,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Modelos a experimentar ────────────────────────────────
MODELS = {
    "random_forest":       RandomForestClassifier(**RANDOM_FOREST_PARAMS),
    "xgboost":             XGBClassifier(**XGBOOST_PARAMS),
    "logistic_regression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
    "svm":                 SVC(**SVM_PARAMS),
}


def build_preprocessor() -> ColumnTransformer:
    """Preprocesamiento básico: escala numéricas, encode categóricas."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS),
    ])


def load_data(data_path: str):
    """Carga y prepara el dataset."""
    df = pd.read_csv(data_path)

    # Limpiar espacios en columnas string
    df.columns = df.columns.str.strip()
    for col in CAT_COLS + [TARGET]:
        df[col] = df[col].str.strip()

    # Drop columnas irrelevantes
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Encode target: Approved=1, Rejected=0
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET])

    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    logger.info(f"Balance del target:\n{df[TARGET].value_counts(normalize=True).round(3)}")

    return df


def run_experiment(model_name: str, model, X_train, X_test, y_train, y_test, df):
    """Corre un experimento completo para un modelo y lo loggea en MLflow."""

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

    with MLflowRun(
        experiment_key="baseline",
        run_name=model_name,
        tags={"model_type": model_name, "phase": "baseline"},
    ) as run:

        # Info del dataset
        log_dataset_info(df, dataset_name="loan_approval")

        # Log parámetros del modelo manualmente (autolog no cubre pipelines siempre)
        mlflow.log_params({
            f"model_{k}": v
            for k, v in model.get_params().items()
        })

        # Cross-validation (5 folds estratificados)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=["accuracy", "f1", "roc_auc", "precision", "recall"],
            return_train_score=True,
            n_jobs=-1,
        )

        # Log métricas de CV
        for metric in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
            mlflow.log_metrics({
                f"cv_train_{metric}_mean": cv_results[f"train_{metric}"].mean(),
                f"cv_val_{metric}_mean":   cv_results[f"test_{metric}"].mean(),
                f"cv_val_{metric}_std":    cv_results[f"test_{metric}"].std(),
            })

        # Entrenamiento final y evaluación en test
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = log_credit_metrics(y_test, y_pred, y_prob, prefix="test")

        logger.info(
            f"  ✅ {model_name} — "
            f"ROC AUC: {test_metrics.get('test_roc_auc', 0):.4f} | "
            f"F1: {test_metrics.get('test_f1', 0):.4f}"
        )

        return test_metrics


def main():
    import os
    data_path = os.getenv("DATA_PATH", "data/raw/credit_data.csv")

    logger.info("=" * 55)
    logger.info("  Experimentos Iniciales — Credit Approval MLOps")
    logger.info("=" * 55)

    # Setup MLflow
    setup_mlflow(MLflowConfig())

    # Cargar datos
    df = load_data(data_path)
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Correr experimentos
    results = {}
    for model_name, model in MODELS.items():
        logger.info(f"\n🔬 Entrenando: {model_name}")
        results[model_name] = run_experiment(
            model_name, model,
            X_train, X_test, y_train, y_test, df,
        )

    # Resumen final
    logger.info("\n" + "=" * 55)
    logger.info("  RESUMEN DE EXPERIMENTOS")
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
    logger.info("  👉 Compara los runs en http://127.0.0.1:5000")


if __name__ == "__main__":
    main()