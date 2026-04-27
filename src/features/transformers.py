"""
────────────────────────────────────────────────────────────
Transformaciones y feature engineering reproducibles.
- Estandarización de numéricas
- Encoding de categóricas
- Pipeline completo preprocesamiento + modelo
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import CAT_COLS, DROP_COLS, NUM_COLS, TARGET


def split_features_target(
    df: pd.DataFrame, target_column: str = TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features y target, eliminando columnas de ID."""
    X = df.drop(columns=[target_column] + [c for c in DROP_COLS if c in df.columns])
    y = df[target_column]
    return X, y


def build_preprocessor(
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """
    Preprocesamiento estándar:
    - Numéricas: StandardScaler
    - Categóricas: OneHotEncoder (drop first)
    """
    num_cols = numerical_features if numerical_features is not None else NUM_COLS
    cat_cols = categorical_features if categorical_features is not None else CAT_COLS
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def build_model_pipeline(model) -> Pipeline:
    """
    Pipeline completo: preprocesamiento + modelo.
    Listo para fit/predict.
    """
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
