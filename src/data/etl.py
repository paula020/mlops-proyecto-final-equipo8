"""
ETL para el dataset de aprobación de crédito.
Extract  → lee el CSV raw
Transform → limpia, valida y transforma
Load     → guarda el dataset procesado listo para entrenamiento
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import mlflow
from dotenv import load_dotenv

from config.settings import (
    TARGET, DROP_COLS, CAT_COLS, NUM_COLS,RAW_DATA_PATH,PROCESSED_DATA_DIR,PROCESSED_DATA_PATH
)
from config.mlflow_config import TRACKING_URI
from src.data.preprocesamiento import clean_dataframe, encode_target

load_dotenv()

logger = logging.getLogger(__name__)



# ── Extract ───────────────────────────────────────────────

def extract(data_path: str) -> pd.DataFrame:
    """Lee el CSV raw y retorna el dataframe original."""
    logger.info(f" EXTRACT — Leyendo: {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f" {df.shape[0]} filas, {df.shape[1]} columnas extraídas")
    return df


# ── Transform ─────────────────────────────────────────────

def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aplica todas las transformaciones al dataframe raw.
    Retorna el dataframe transformado y un reporte de calidad.
    """
    logger.info(" TRANSFORM — Aplicando transformaciones")
    report = {}

    # 1. Shape original
    report["rows_raw"]    = len(df)
    report["cols_raw"]    = len(df.columns)

    # 2. Drop columnas irrelevantes
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # 3. Limpieza
    df = clean_dataframe(df)

    # 4. Validar y reportar negativos
    for col in NUM_COLS:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            report[f"negative_{col}"] = int(n_neg)

    # 5. Encode target
    df, le = encode_target(df)

    # 6. Reporte de calidad
    report["rows_processed"]  = len(df)
    report["cols_processed"]  = len(df.columns)
    report["nulls_total"]     = int(df.isnull().sum().sum())
    report["duplicates"]      = int(df.duplicated().sum())
    report["target_balance"]  = round(
        df[TARGET].value_counts(normalize=True).iloc[0], 4
    )
    report["timestamp"]       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f" Transformación completada — {df.shape}")
    logger.info(f"   Nulos totales  : {report['nulls_total']}")
    logger.info(f"   Duplicados     : {report['duplicates']}")
    logger.info(f"   Balance target : {report['target_balance']}")

    return df, report


# ── Load ──────────────────────────────────────────────────

def load(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH) -> None:
    """Guarda el dataset procesado en data/processed/."""
    logger.info(f" LOAD — Guardando en: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f" Dataset procesado guardado — {df.shape[0]} filas")


# ── Log MLflow ────────────────────────────────────────────

def log_etl_mlflow(report: dict) -> None:
    """Loggea el reporte de calidad del ETL en MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("credit_approval/baseline")

    with mlflow.start_run(run_name="etl_pipeline", nested=True):
        mlflow.set_tags({
            "phase":   "etl",
            "project": "credit_approval",
            "status":  "SUCCESS",
        })
        mlflow.log_params({
            "raw_rows":       report["rows_raw"],
            "raw_cols":       report["cols_raw"],
            "processed_rows": report["rows_processed"],
            "processed_cols": report["cols_processed"],
            "timestamp":      report["timestamp"],
        })
        mlflow.log_metrics({
            "nulls_total":    report["nulls_total"],
            "duplicates":     report["duplicates"],
            "target_balance": report["target_balance"],
        })
    logger.info(" Reporte ETL loggeado en MLflow")


# ── Pipeline ETL completo ─────────────────────────────────

def run_etl(
    data_path: str = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    log_mlflow: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline ETL completo:
    Extract → Transform → Load → Log
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 55)
    logger.info("  ETL Pipeline — Credit Approval MLOps")
    logger.info("=" * 55)

    # Extract
    df_raw = extract(data_path)

    # Transform
    df_processed, report = transform(df_raw)

    # Load
    load(df_processed, output_path)

    # Log
    if log_mlflow:
        log_etl_mlflow(report)

    logger.info("=" * 55)
    logger.info("   ETL COMPLETADO")
    logger.info(f"  Raw      : {report['rows_raw']} filas")
    logger.info(f"  Procesado: {report['rows_processed']} filas")
    logger.info(f"  Output   : {output_path}")
    logger.info("=" * 55)

    return df_processed


if __name__ == "__main__":
    run_etl()