"""
────────────────────────────────────────────────────────────
Transformaciones y feature engineering reproducibles.
- Estandarización de numéricas
- Encoding de categóricas
- Pipeline completo preprocesamiento + modelo
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config.settings import NUM_COLS, CAT_COLS


def build_preprocessor() -> ColumnTransformer:
    """
    Preprocesamiento estándar:
    - Numéricas: StandardScaler
    - Categóricas: OneHotEncoder (drop first)
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS),
        ],
        remainder="drop",
    )


def build_model_pipeline(model) -> Pipeline:
    """
    Pipeline completo: preprocesamiento + modelo.
    Listo para fit/predict.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])