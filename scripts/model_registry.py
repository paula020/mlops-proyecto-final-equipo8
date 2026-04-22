"""
────────────────────────────────────────────────────────────
Registra los mejores modelos del tuning en MLflow Model
Registry, los versiona con tags y gestiona el flujo
Staging → Production.

"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from config.mlflow_config import (
    setup_mlflow, MLflowConfig, TRACKING_URI, EXPERIMENTS,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Nombre del modelo registrado ──────────────────────────
REGISTERED_MODEL_NAME = "Clasificador aprobación crédito"


def get_best_runs(client: MlflowClient, n: int = 4) -> list:
    """
    Obtiene los mejores runs del experimento hyperopt
    ordenados por test_roc_auc descendente.
    """
    experiment = client.get_experiment_by_name(EXPERIMENTS["hyperopt"])
    if experiment is None:
        raise ValueError(
            f"Experimento '{EXPERIMENTS['hyperopt']}' no encontrado. "
            "Corre primero tuning_hiperparametros.py"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.status = 'SUCCESS'",
        order_by=["metrics.test_roc_auc DESC"],
        max_results=n,
    )

    if not runs:
        raise ValueError("No se encontraron runs exitosos en el experimento hyperopt.")

    logger.info(f"🔍 Mejores {len(runs)} runs encontrados:")
    for r in runs:
        logger.info(
            f"   {r.data.tags.get('model_type', 'unknown'):<25} "
            f"ROC AUC: {r.data.metrics.get('test_roc_auc', 0):.4f}"
        )
    return runs


def register_model(client: MlflowClient, run: mlflow.entities.Run) -> str:
    """
    Registra un run en el Model Registry y retorna la versión.
    """
    model_type = run.data.tags.get("model_type", "unknown")
    run_id     = run.info.run_id
    artifact_path = f"model_{model_type}"

    model_uri = f"runs:/{run_id}/{artifact_path}"

    logger.info(f"\n Registrando: {model_type}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )
    version = result.version
    logger.info(f"    Versión {version} registrada")
    return version


def add_version_tags(client: MlflowClient, version: str, run: mlflow.entities.Run) -> None:
    """Agrega tags descriptivos a la versión del modelo."""
    model_type = run.data.tags.get("model_type", "unknown")
    roc_auc    = run.data.metrics.get("test_roc_auc", 0)
    f1         = run.data.metrics.get("test_f1", 0)

    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "model_type",   model_type)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "test_roc_auc", f"{roc_auc:.4f}")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "test_f1",      f"{f1:.4f}")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "registered_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "run_id",       run.info.run_id)

    logger.info(f"     Tags agregados a versión {version}")


def promote_to_staging(client: MlflowClient, version: str, run: mlflow.entities.Run) -> None:
    """Mueve una versión al stage Staging."""
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    model_type = run.data.tags.get("model_type", "unknown")
    logger.info(f"    Versión {version} ({model_type}) → Staging")


def promote_to_production(client: MlflowClient, version: str, run: mlflow.entities.Run) -> None:
    """Mueve la mejor versión a Production y archiva las anteriores."""
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,  # archiva versiones previas en Production
    )
    model_type = run.data.tags.get("model_type", "unknown")
    roc_auc    = run.data.metrics.get("test_roc_auc", 0)
    logger.info(f"    Versión {version} ({model_type}) → Production (ROC AUC: {roc_auc:.4f})")


def create_model_card(client: MlflowClient, best_run: mlflow.entities.Run) -> None:
    """
    Crea y loggea una model card como artefacto del mejor run.
    """
    model_type = best_run.data.tags.get("model_type", "unknown")
    metrics    = best_run.data.metrics
    params     = best_run.data.params

    card = f"""# Model Card — Credit Approval Classifier
## Modelo en Producción

| Campo           | Valor                          |
|-----------------|-------------------------------|
| Algoritmo       | {model_type}                  |
| Run ID          | {best_run.info.run_id}        |
| Fecha registro  | {datetime.now().strftime("%Y-%m-%d")} |
| Versión         | Production                    |

## Métricas de Evaluación

| Métrica         | Valor                         |
|-----------------|-------------------------------|
| ROC AUC         | {metrics.get("test_roc_auc", 0):.4f} |
| F1 Score        | {metrics.get("test_f1", 0):.4f}      |
| Accuracy        | {metrics.get("test_accuracy", 0):.4f}|
| Precision       | {metrics.get("test_precision", 0):.4f}|
| Recall          | {metrics.get("test_recall", 0):.4f}  |

## Datos de Entrenamiento

| Campo           | Valor                         |
|-----------------|-------------------------------|
| Dataset         | loan_approval_dataset         |
| Filas           | {int(metrics.get("dataset_rows", 0))} |
| Features        | no_of_dependents, income_annum, loan_amount, loan_term,|
|                 | cibil_score, residential_assets_value,                 |
|                 | commercial_assets_value, luxury_assets_value,          |
|                 | bank_asset_value, education, self_employed             |
| Target          | loan_status (Approved / Rejected)                      |
| Split           | 80% train / 20% test                                   |

## Consideraciones de Uso

- Modelo entrenado para clasificación binaria de aprobación de crédito.
- Reentrenar si ROC AUC en producción cae por debajo de 0.95.
"""

    # Guardar como archivo y loggear como artefacto
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(card)
        tmp_path = f.name

    with mlflow.start_run(run_id=best_run.info.run_id):
        mlflow.log_artifact(tmp_path, artifact_path="model_card")

    os.unlink(tmp_path)
    logger.info(f"    Model card creada y loggeada como artefacto")


def main():
    logger.info("=" * 55)
    logger.info("  Model Registry — Credit Approval MLOps")
    logger.info("=" * 55)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = setup_mlflow(MLflowConfig())

    # 1. Obtener mejores runs del tuning
    best_runs = get_best_runs(client, n=4)

    # 2. Registrar, taggear y promover a Staging
    versions = []
    for run in best_runs:
        version = register_model(client, run)
        add_version_tags(client, version, run)
        promote_to_staging(client, version, run)
        versions.append((version, run))

    # 3. Promover el mejor a Production (primero de la lista = mayor ROC AUC)
    best_version, best_run = versions[0]
    promote_to_production(client, best_version, best_run)

    # 4. Crear model card del modelo en producción
    create_model_card(client, best_run)

    # Resumen
    logger.info("\n" + "=" * 55)
    logger.info("  RESUMEN MODEL REGISTRY")
    logger.info("=" * 55)
    for version, run in versions:
        model_type = run.data.tags.get("model_type", "unknown")
        roc_auc    = run.data.metrics.get("test_roc_auc", 0)
        stage      = "Production " if version == best_version else "Staging "
        logger.info(f"  v{version} {model_type:<25} ROC AUC: {roc_auc:.4f} → {stage}")
    logger.info("=" * 55)
    logger.info("   Revisa el registry en http://127.0.0.1:5000/#/models")


if __name__ == "__main__":
    main()