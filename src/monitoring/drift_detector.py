from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config.settings import CAT_COLS, NUM_COLS

NUMERIC_DRIFT_THRESHOLD = 0.20
CATEGORICAL_DRIFT_THRESHOLD = 0.15
EPSILON = 1e-6


def _distribution(values: pd.Series) -> pd.Series:
    total = values.sum()
    if total == 0:
        return values.astype(float)
    return values.astype(float) / float(total)


def _calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calcula Population Stability Index para una feature numerica."""
    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)

    if reference.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    bin_edges = np.unique(np.quantile(reference, quantiles))

    # Si no hay variacion real en la variable, no hay drift numerico util.
    if len(bin_edges) < 3:
        return 0.0

    reference_bins = pd.cut(reference, bin_edges, include_lowest=True)
    current_bins = pd.cut(current, bin_edges, include_lowest=True)

    ref_dist = _distribution(reference_bins.value_counts(sort=False))
    cur_dist = _distribution(current_bins.value_counts(sort=False))

    psi = np.sum((cur_dist - ref_dist) * np.log((cur_dist + EPSILON) / (ref_dist + EPSILON)))
    return float(psi)


def _calculate_categorical_distance(reference: pd.Series, current: pd.Series) -> float:
    """Calcula distancia de variacion total entre distribuciones categoricas."""
    reference = reference.astype(str).str.strip().fillna("<NA>")
    current = current.astype(str).str.strip().fillna("<NA>")

    ref_dist = reference.value_counts(normalize=True)
    cur_dist = current.value_counts(normalize=True)

    all_categories = sorted(set(ref_dist.index) | set(cur_dist.index))
    ref_prob = ref_dist.reindex(all_categories, fill_value=0.0)
    cur_prob = cur_dist.reindex(all_categories, fill_value=0.0)

    tvd = 0.5 * np.abs(ref_prob - cur_prob).sum()
    return float(tvd)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Compara referencia vs datos actuales y retorna reporte de drift."""
    numeric_cols = numeric_cols or NUM_COLS
    categorical_cols = categorical_cols or CAT_COLS

    numeric_results: list[dict[str, Any]] = []
    categorical_results: list[dict[str, Any]] = []
    missing_columns: list[str] = []

    for col in numeric_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            missing_columns.append(col)
            continue

        score = _calculate_psi(reference_df[col], current_df[col])
        numeric_results.append(
            {
                "feature": col,
                "method": "PSI",
                "score": round(score, 6),
                "threshold": NUMERIC_DRIFT_THRESHOLD,
                "drift_detected": bool(score >= NUMERIC_DRIFT_THRESHOLD),
            }
        )

    for col in categorical_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            missing_columns.append(col)
            continue

        score = _calculate_categorical_distance(reference_df[col], current_df[col])
        categorical_results.append(
            {
                "feature": col,
                "method": "TVD",
                "score": round(score, 6),
                "threshold": CATEGORICAL_DRIFT_THRESHOLD,
                "drift_detected": bool(score >= CATEGORICAL_DRIFT_THRESHOLD),
            }
        )

    total_features = len(numeric_results) + len(categorical_results)
    drifted_features = sum(
        1 for row in numeric_results + categorical_results if row["drift_detected"]
    )

    overall_status = "drift_detectado" if drifted_features > 0 else "estable"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "summary": {
            "total_features_evaluated": total_features,
            "features_with_drift": drifted_features,
            "drift_ratio": round((drifted_features / total_features), 4) if total_features else 0.0,
            "missing_columns": sorted(set(missing_columns)),
        },
        "numeric_features": numeric_results,
        "categorical_features": categorical_results,
    }


PREDICTION_DRIFT_THRESHOLD = 0.10


def detect_prediction_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str = "loan_status",
) -> dict[str, Any]:
    """
    Compara la distribución de predicciones (Approved/Rejected) entre
    referencia y datos actuales usando TVD.

    Si el target_col no existe en current_df, el resultado queda como
    'no_disponible' — útil cuando no se tienen etiquetas reales en producción.
    """
    if target_col not in reference_df.columns:
        return {
            "status": "no_disponible",
            "motivo": f"Columna '{target_col}' no encontrada en referencia.",
            "score": None,
            "threshold": PREDICTION_DRIFT_THRESHOLD,
        }

    ref_dist = (
        reference_df[target_col].astype(str).str.strip().fillna("<NA>").value_counts(normalize=True)
    )

    if target_col not in current_df.columns:
        return {
            "status": "no_disponible",
            "motivo": (
                f"Columna '{target_col}' no encontrada en datos actuales. "
                "Proporciona predicciones reales del modelo para habilitar este análisis."
            ),
            "reference_distribution": ref_dist.to_dict(),
            "score": None,
            "threshold": PREDICTION_DRIFT_THRESHOLD,
        }

    cur_dist = (
        current_df[target_col].astype(str).str.strip().fillna("<NA>").value_counts(normalize=True)
    )

    all_classes = sorted(set(ref_dist.index) | set(cur_dist.index))
    ref_prob = ref_dist.reindex(all_classes, fill_value=0.0)
    cur_prob = cur_dist.reindex(all_classes, fill_value=0.0)

    tvd = float(0.5 * np.abs(ref_prob - cur_prob).sum())
    drift_detected = tvd >= PREDICTION_DRIFT_THRESHOLD

    return {
        "status": "drift_detectado" if drift_detected else "estable",
        "method": "TVD",
        "target_col": target_col,
        "score": round(tvd, 6),
        "threshold": PREDICTION_DRIFT_THRESHOLD,
        "drift_detected": drift_detected,
        "reference_distribution": ref_prob.to_dict(),
        "current_distribution": cur_prob.to_dict(),
    }
