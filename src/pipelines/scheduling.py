"""
Configura el scheduling automático del flow de entrenamiento.
Despliega el flow con ejecución semanal automática.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule
from dotenv import load_dotenv

from src.pipelines.flow_entrenamiento import training_flow
from config.settings import DEFAULT_MODEL, CRON_ENTRENAMIENTO, TIMEZONE

load_dotenv()

# ── Schedule semanal ──────────────────────────────────────


def deploy_training_flow():
    """
    Despliega el flow de entrenamiento con schedule semanal.
    Mantén este script corriendo para que Prefect ejecute
    el flow automáticamente.
    """

    print("=" * 55)
    print("  Scheduling — Aprobación de crédito MLOps")
    print("=" * 55)
    print(f"  Modelo    : {DEFAULT_MODEL}")
    print(f"  Schedule  : Cada día a las 6:00 PM")
    print(f"  Cron      : {CRON_ENTRENAMIENTO}")
    print("=" * 55)
    print("   Revisa los deploys en http://127.0.0.1:4200/deployments")
    print("  Ctrl+C para detener")
    print("=" * 55)

    # Crear deployment con schedule
    deployment = training_flow.to_deployment(
        name="entrenamiento-semanal-aprobación-crédito",
        schedule=CronSchedule(cron=CRON_ENTRENAMIENTO, timezone=TIMEZONE),
        parameters={"model_name": DEFAULT_MODEL},
        tags=["credit-approval", "training", "scheduled"],
        description="Reentrenamiento semanal automático — cada día 6PM Bogotá",
    )

    # Levantar el servidor de deployments
    serve(deployment)


if __name__ == "__main__":
    deploy_training_flow()