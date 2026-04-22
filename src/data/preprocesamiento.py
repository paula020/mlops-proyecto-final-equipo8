"""
────────────────────────────────────────────────────────────
Carga y limpieza del dato crudo.
- Lectura del CSV
- Tratamiento de valores negativos
- Encoding del target
"""

import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow

from config.settings import (
    TARGET, DROP_COLS, CAT_COLS,NEGATIVE_VALUE_COLS
)

logger = logging.getLogger(__name__)



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica:
    - Strip de espacios en columnas string
    - Trata valores negativos en residential_assets_value
    """
    df = df.copy()

    # Strip espacios
    df.columns = df.columns.str.strip()
    for col in CAT_COLS + [TARGET]:
        df[col] = df[col].str.strip()

    # Tratar valores negativos — reemplaza con 0
    for col in NEGATIVE_VALUE_COLS:
        if col in df.columns:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                logger.info(
                    f"  {col}: {n_negative} valores negativos "
                    f"reemplazados por 0"
                )
                mlflow.log_metric(f"negative_values_{col}", int(n_negative))
                df[col] = df[col].clip(lower=0)

    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode target: Approved=1, Rejected=0."""
    le = LabelEncoder()
    df = df.copy()
    df[TARGET] = le.fit_transform(df[TARGET])
    logger.info(
        f"Target clases: {dict(zip(le.classes_, le.transform(le.classes_)))}"
    )
    return df, le


def load_and_clean(data_path: str) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Carga el CSV, limpia y encodea el target.
    Retorna el dataframe listo y el label encoder.
    """
    df = pd.read_csv(data_path)
    logger.info(f" Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Drop columnas irrelevantes
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Limpieza
    df = clean_dataframe(df)

    # Encode target
    df, le = encode_target(df)

    logger.info(f" Dataset limpio: {df.shape[0]} filas, {df.shape[1]} columnas")
    logger.info(
        f"Balance target: "
        f"{df[TARGET].value_counts(normalize=True).round(3).to_dict()}"
    )

    return df, le