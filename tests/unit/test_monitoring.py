"""
Tests unitarios para src/monitoring/drift_detector.py
Fase 6 — Equipo 8
"""

import pandas as pd
import pytest

from src.monitoring.drift_detector import detect_drift, detect_prediction_drift

# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture()
def reference_df() -> pd.DataFrame:
    """Dataset de referencia pequeño con distribución conocida."""
    return pd.DataFrame(
        {
            "income_annum": [500000, 600000, 700000, 800000, 900000, 1000000] * 5,
            "loan_amount": [200000, 250000, 300000, 350000, 400000, 450000] * 5,
            "cibil_score": [650, 680, 700, 720, 740, 760] * 5,
            "no_of_dependents": [0, 1, 2, 3, 0, 1] * 5,
            "loan_term": [5, 10, 15, 20, 10, 5] * 5,
            "residential_assets_value": [100000, 200000, 300000, 400000, 500000, 600000] * 5,
            "commercial_assets_value": [50000, 100000, 150000, 200000, 250000, 300000] * 5,
            "luxury_assets_value": [10000, 20000, 30000, 40000, 50000, 60000] * 5,
            "bank_asset_value": [80000, 90000, 100000, 110000, 120000, 130000] * 5,
            "education": ["Graduate", "Not Graduate"] * 15,
            "self_employed": ["No", "Yes", "No", "No", "Yes", "No"] * 5,
            "loan_status": ["Approved", "Rejected", "Approved", "Approved", "Rejected", "Approved"]
            * 5,
        }
    )


# ── detect_drift ──────────────────────────────────────────


def test_detect_drift_estable_cuando_referencia_igual_actual(
    reference_df: pd.DataFrame,
) -> None:
    """Si referencia y actual son idénticos no debe detectarse drift."""
    result = detect_drift(reference_df, reference_df.copy())

    assert result["status"] == "estable"
    assert result["summary"]["features_with_drift"] == 0
    assert result["summary"]["drift_ratio"] == 0.0


def test_detect_drift_detecta_cambio_en_feature_numerica(
    reference_df: pd.DataFrame,
) -> None:
    """Una feature numérica con valores multiplicados por 10 debe detectar drift (PSI alto)."""
    current = reference_df.copy()
    current["cibil_score"] = current["cibil_score"] * 10  # drift extremo

    result = detect_drift(reference_df, current)

    drifted = [
        row for row in result["numeric_features"] if row["feature"] == "cibil_score"
    ]
    assert len(drifted) == 1
    assert drifted[0]["drift_detected"] is True


def test_detect_drift_detecta_cambio_en_feature_categorica(
    reference_df: pd.DataFrame,
) -> None:
    """Cambiar self_employed a 100% 'Yes' debe superar el umbral TVD."""
    current = reference_df.copy()
    current["self_employed"] = "Yes"  # distribución completamente distinta

    result = detect_drift(reference_df, current)

    drifted = [
        row for row in result["categorical_features"] if row["feature"] == "self_employed"
    ]
    assert len(drifted) == 1
    assert drifted[0]["drift_detected"] is True


def test_detect_drift_estructura_reporte(reference_df: pd.DataFrame) -> None:
    """El reporte siempre debe contener las claves esperadas."""
    result = detect_drift(reference_df, reference_df.copy())

    assert "generated_at_utc" in result
    assert "status" in result
    assert "summary" in result
    assert "numeric_features" in result
    assert "categorical_features" in result

    summary = result["summary"]
    assert "total_features_evaluated" in summary
    assert "features_with_drift" in summary
    assert "drift_ratio" in summary


# ── detect_prediction_drift ───────────────────────────────


def test_prediction_drift_estable_cuando_distribucion_igual(
    reference_df: pd.DataFrame,
) -> None:
    """Sin cambio en loan_status el drift debe ser estable."""
    result = detect_prediction_drift(reference_df, reference_df.copy())

    assert result["status"] == "estable"
    assert result["score"] < 0.10


def test_prediction_drift_detectado_cuando_ratio_cambia(
    reference_df: pd.DataFrame,
) -> None:
    """Cambiar todos los loan_status a Rejected debe superar el umbral TVD."""
    current = reference_df.copy()
    current["loan_status"] = "Rejected"  # 100% rechazos vs ~33% original

    result = detect_prediction_drift(reference_df, current)

    assert result["status"] == "drift_detectado"
    assert result["drift_detected"] is True
    assert result["score"] >= 0.10


def test_prediction_drift_no_disponible_sin_columna_target(
    reference_df: pd.DataFrame,
) -> None:
    """Si current_df no tiene loan_status, debe retornar no_disponible sin lanzar error."""
    current = reference_df.drop(columns=["loan_status"])

    result = detect_prediction_drift(reference_df, current)

    assert result["status"] == "no_disponible"
    assert result["score"] is None
    assert "motivo" in result
