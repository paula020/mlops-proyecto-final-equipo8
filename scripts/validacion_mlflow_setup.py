"""
Corre un run de prueba end-to-end para confirmar que
toda la configuración de MLflow funciona correctamente.

"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config.mlflow_config import (
    setup_mlflow, MLflowRun, MLflowConfig,
    log_credit_metrics, log_dataset_info,
    TRACKING_URI,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_smoke_test():
    logger.info("=" * 50)
    logger.info("  MLflow Validacion — Aprobación de créditos MLOps")
    logger.info("=" * 50)

    # 1. Setup de experimentos
    config = MLflowConfig()
    client = setup_mlflow(config)

    # 2. Datos sintéticos (simula dataset de aprobación)
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6,
        n_classes=2, random_state=42,
    )
    feature_names = [
        "income", "debt_ratio", "age", "num_loans",
        "credit_score", "employment_years", "loan_amount",
        "num_dependents", "past_defaults", "savings_ratio",
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["approved"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_names], df["approved"],
        test_size=0.2, random_state=42, stratify=df["approved"],
    )

    # 3. Run de prueba
    with MLflowRun(
        experiment_key="baseline",
        run_name="smoke_test_rf",
        tags={"smoke_test": "true"},
    ) as run:

        log_dataset_info(df, dataset_name="synthetic_credit")

        model = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = log_credit_metrics(y_test, y_pred, y_prob, prefix="test")

        run_id = run.info.run_id

    # 4. Verificar que se guardó
    finished_run = client.get_run(run_id)
    assert finished_run.info.status == "FINISHED"

    logger.info("\n" + "=" * 50)
    logger.info("   VALIDACIÓN EXITOSA")
    logger.info(f"  Run ID  : {run_id}")
    logger.info(f"  ROC AUC : {metrics.get('test_roc_auc', 0):.4f}")
    logger.info(f"  UI      : {TRACKING_URI}")
    logger.info("=" * 50)
    logger.info(" Revisa el run en http://127.0.0.1:5000")


if __name__ == "__main__":
    run_smoke_test()