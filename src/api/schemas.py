"""
Schemas Pydantic para la API de aprobación de crédito.
Validan los inputs y tipan los outputs de cada endpoint.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Input ─────────────────────────────────────────────────

class CreditApplication(BaseModel):
    """
    Datos de una solicitud de crédito.
    Corresponde exactamente a las columnas que usa el pipeline entrenado:
    NUM_COLS + CAT_COLS de config/settings.py
    """

    # Numéricas (NUM_COLS)
    no_of_dependents:          int   = Field(..., ge=0, le=10,         description="Número de dependientes (0-10)")
    income_annum:              float = Field(..., gt=0,                description="Ingreso anual en pesos")
    loan_amount:               float = Field(..., gt=0,                description="Monto del préstamo solicitado")
    loan_term:                 int   = Field(..., ge=1, le=30,         description="Plazo del préstamo en años (1-30)")
    cibil_score:               int   = Field(..., ge=300, le=900,      description="Score crediticio CIBIL (300-900)")
    residential_assets_value:  float = Field(..., ge=0,                description="Valor de activos residenciales")
    commercial_assets_value:   float = Field(..., ge=0,                description="Valor de activos comerciales")
    luxury_assets_value:       float = Field(..., ge=0,                description="Valor de activos de lujo")
    bank_asset_value:          float = Field(..., ge=0,                description="Valor de activos bancarios")

    # Categóricas (CAT_COLS)
    education:      str = Field(..., description="Nivel educativo: 'Graduate' o 'Not Graduate'")
    self_employed:  str = Field(..., description="Trabaja por cuenta propia: 'Yes' o 'No'")

    @field_validator("education")
    @classmethod
    def validate_education(cls, v: str) -> str:
        valid = {"Graduate", "Not Graduate"}
        v = v.strip()
        if v not in valid:
            raise ValueError(f"'education' debe ser uno de: {valid}. Recibido: '{v}'")
        return v

    @field_validator("self_employed")
    @classmethod
    def validate_self_employed(cls, v: str) -> str:
        valid = {"Yes", "No"}
        v = v.strip()
        if v not in valid:
            raise ValueError(f"'self_employed' debe ser uno de: {valid}. Recibido: '{v}'")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "no_of_dependents":         2,
                "income_annum":             5800000,
                "loan_amount":              12000000,
                "loan_term":                10,
                "cibil_score":              720,
                "residential_assets_value": 8000000,
                "commercial_assets_value":  2000000,
                "luxury_assets_value":      500000,
                "bank_asset_value":         3000000,
                "education":               "Graduate",
                "self_employed":           "No",
            }
        }
    }


class BatchCreditApplication(BaseModel):
    """Lista de solicitudes para predicción en lote."""
    applications: list[CreditApplication] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de solicitudes (máx. 100)",
    )


# ── Output ────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Resultado de una predicción individual."""
    aprobado:     bool  = Field(..., description="True si el crédito sería aprobado")
    probabilidad: float = Field(..., description="Probabilidad de aprobación (0.0 - 1.0)")
    etiqueta:     str   = Field(..., description="'Approved' o 'Rejected'")
    score:        int   = Field(..., description="CIBIL score del solicitante")

    model_config = {
        "json_schema_extra": {
            "example": {
                "aprobado":     True,
                "probabilidad": 0.8732,
                "etiqueta":     "Approved",
                "score":        720,
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Resultado de predicción en lote."""
    total:        int                    = Field(..., description="Total de solicitudes procesadas")
    aprobados:    int                    = Field(..., description="Cantidad de créditos aprobados")
    rechazados:   int                    = Field(..., description="Cantidad de créditos rechazados")
    predicciones: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Estado de salud de la API."""
    status:          str           = Field(..., description="'ok' o 'degradado'")
    modelo_cargado:  bool          = Field(..., description="True si el modelo está disponible")
    model_uri:       str           = Field(..., description="URI del modelo en MLflow")
    error:           Optional[str] = Field(None, description="Mensaje de error si el modelo no cargó")


class ModelInfoResponse(BaseModel):
    """Información del modelo cargado."""
    model_name:  str            = Field(..., description="Nombre en el MLflow Registry")
    model_stage: str            = Field(..., description="Stage: Production, Staging, etc.")
    model_uri:   str            = Field(..., description="URI completa del modelo")
    model_type:  str            = Field(..., description="Tipo de estimador (XGBClassifier, etc.)")
    n_features:  Optional[int]  = Field(None, description="Número de features del modelo")
    params:      dict           = Field(..., description="Hiperparámetros del modelo")