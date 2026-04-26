"""
API REST para predicción de aprobación de crédito.
Fase 4 — MLOps Proyecto Final Equipo 8

Endpoints:
  GET  /           → health check
  GET  /health     → estado del modelo
  POST /predict    → predicción individual
  POST /predict/batch → predicciones en lote
  GET  /model/info → información del modelo cargado
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CreditApplication,
    BatchCreditApplication,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

load_dotenv()

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config desde env ──────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "credit_approval_model")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Production")
MODEL_URI           = os.getenv(
    "MODEL_URI",
    f"models:/{MODEL_NAME}/{MODEL_STAGE}",
)

# ── Estado global del modelo ──────────────────────────────
model_state = {
    "pipeline":    None,
    "model_uri":   MODEL_URI,
    "model_name":  MODEL_NAME,
    "model_stage": MODEL_STAGE,
    "loaded":      False,
    "error":       None,
}


# ── Lifespan (carga del modelo al arrancar) ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar y libera recursos al cerrar."""
    logger.info("=" * 55)
    logger.info("  Iniciando API — Aprobación de Crédito MLOps")
    logger.info("=" * 55)

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"  MLflow URI : {MLFLOW_TRACKING_URI}")
        logger.info(f"  Model URI  : {MODEL_URI}")

        model_state["pipeline"] = mlflow.sklearn.load_model(MODEL_URI)
        model_state["loaded"]   = True
        logger.info("  Modelo cargado exitosamente ✓")

    except Exception as e:
        model_state["loaded"] = False
        model_state["error"]  = str(e)
        logger.warning(f"  No se pudo cargar el modelo: {e}")
        logger.warning("  La API arranca en modo degradado — /predict no disponible")

    yield  # ── App corriendo ──

    logger.info("  Cerrando API...")
    model_state["pipeline"] = None
    model_state["loaded"]   = False


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Credit Approval API",
    description=(
        "API de predicción de aprobación de crédito. "
        "Proyecto MLOps — Equipo 8."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────
def _require_model():
    """Lanza 503 si el modelo no está cargado."""
    if not model_state["loaded"] or model_state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error":   "Modelo no disponible",
                "mensaje": "El modelo no pudo cargarse al iniciar la API.",
                "causa":   model_state.get("error", "Desconocida"),
            },
        )


def _application_to_df(application: CreditApplication) -> pd.DataFrame:
    """Convierte un CreditApplication en DataFrame con las columnas correctas."""
    return pd.DataFrame([application.model_dump()])


def _predict(df: pd.DataFrame) -> tuple[list[int], list[float]]:
    """Ejecuta predicción y retorna etiquetas y probabilidades."""
    pipeline  = model_state["pipeline"]
    labels    = pipeline.predict(df).tolist()
    proba     = pipeline.predict_proba(df)[:, 1].tolist()
    return labels, proba


# ── Endpoints ─────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "api":     "Credit Approval API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    """Estado general de la API y del modelo."""
    return HealthResponse(
        status   = "ok" if model_state["loaded"] else "degradado",
        modelo_cargado = model_state["loaded"],
        model_uri      = model_state["model_uri"],
        error          = model_state.get("error"),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Modelo"])
def model_info():
    """Información del modelo actualmente cargado."""
    _require_model()
    pipeline  = model_state["pipeline"]
    estimator = pipeline.named_steps.get("model", pipeline)

    return ModelInfoResponse(
        model_name   = model_state["model_name"],
        model_stage  = model_state["model_stage"],
        model_uri    = model_state["model_uri"],
        model_type   = type(estimator).__name__,
        n_features   = getattr(estimator, "n_features_in_", None),
        params       = {
            k: str(v)
            for k, v in estimator.get_params().items()
        },
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(application: CreditApplication):
    """
    Predicción individual de aprobación de crédito.

    Retorna:
    - `aprobado`: True si el crédito sería aprobado
    - `probabilidad`: probabilidad de aprobación (0-1)
    - `etiqueta`: "Approved" o "Rejected"
    """
    _require_model()

    try:
        df              = _application_to_df(application)
        labels, probas  = _predict(df)
        label           = labels[0]
        proba           = probas[0]

        logger.info(
            f"  Predicción — label: {label} | "
            f"proba: {proba:.4f} | "
            f"cibil: {application.cibil_score}"
        )

        return PredictionResponse(
            aprobado     = bool(label == 1),
            probabilidad = round(proba, 4),
            etiqueta     = "Approved" if label == 1 else "Rejected",
            score        = application.cibil_score,
        )

    except Exception as e:
        logger.error(f"  Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error de predicción: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predicción"])
def predict_batch(batch: BatchCreditApplication):
    """
    Predicciones en lote (máximo 100 aplicaciones).

    Retorna una lista de predicciones con su probabilidad.
    """
    _require_model()

    if len(batch.applications) == 0:
        raise HTTPException(status_code=422, detail="La lista de aplicaciones está vacía.")

    if len(batch.applications) > 100:
        raise HTTPException(
            status_code=422,
            detail="Máximo 100 aplicaciones por request.",
        )

    try:
        df             = pd.DataFrame([a.model_dump() for a in batch.applications])
        labels, probas = _predict(df)

        predicciones = [
            PredictionResponse(
                aprobado     = bool(lbl == 1),
                probabilidad = round(p, 4),
                etiqueta     = "Approved" if lbl == 1 else "Rejected",
                score        = batch.applications[i].cibil_score,
            )
            for i, (lbl, p) in enumerate(zip(labels, probas))
        ]

        aprobados  = sum(1 for p in predicciones if p.aprobado)
        rechazados = len(predicciones) - aprobados

        logger.info(
            f"  Batch — {len(predicciones)} predicciones | "
            f"Aprobados: {aprobados} | Rechazados: {rechazados}"
        )

        return BatchPredictionResponse(
            total       = len(predicciones),
            aprobados   = aprobados,
            rechazados  = rechazados,
            predicciones= predicciones,
        )

    except Exception as e:
        logger.error(f"  Error en batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error de predicción batch: {str(e)}")


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )