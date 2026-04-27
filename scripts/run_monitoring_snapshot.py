from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente: python scripts/run_monitoring_snapshot.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import CAT_COLS, NUM_COLS, TARGET
from src.monitoring.drift_detector import detect_drift, detect_prediction_drift
from src.monitoring.report import save_monitoring_report


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _build_current_from_reference(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Crea un escenario local de ejemplo cuando no hay data productiva.

    Simula un escenario de drift moderado:
    - loan_amount sube 20% (cambio económico)
    - cibil_score baja 25 pts (peor perfil crediticio)
    - self_employed pasa a 100% Yes (sesgo de selección)
    - loan_status: ratio Approved baja de ~62% a ~40% (deterioro de cartera)
    """
    import numpy as np

    rng = np.random.default_rng(42)
    current = reference_df.sample(frac=0.30, random_state=42).copy()

    if "loan_amount" in current.columns:
        current["loan_amount"] = current["loan_amount"] * 1.20

    if "cibil_score" in current.columns:
        current["cibil_score"] = (current["cibil_score"] - 25).clip(lower=300)

    if "self_employed" in current.columns:
        current["self_employed"] = "Yes"

    # Simular cambio en distribución de predicciones (más rechazos)
    if TARGET in current.columns:
        n = len(current)
        simulated = rng.choice(["Approved", "Rejected"], size=n, p=[0.40, 0.60])
        current[TARGET] = simulated

    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot local de monitoreo de drift")
    parser.add_argument(
        "--reference-path",
        default="data/loan_approval_dataset.csv",
        help="CSV de referencia (normalmente entrenamiento)",
    )
    parser.add_argument(
        "--current-path",
        default=None,
        help="CSV actual de producción. Si no se entrega, se simula uno.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/monitoring",
        help="Carpeta de salida para reportes",
    )

    args = parser.parse_args()

    reference_df = _load_csv(args.reference_path)

    if args.current_path:
        current_df = _load_csv(args.current_path)
    else:
        current_df = _build_current_from_reference(reference_df)

    report = detect_drift(
        reference_df=reference_df,
        current_df=current_df,
        numeric_cols=NUM_COLS,
        categorical_cols=CAT_COLS,
    )

    prediction_drift = detect_prediction_drift(
        reference_df=reference_df,
        current_df=current_df,
        target_col=TARGET,
    )
    report["prediction_drift"] = prediction_drift

    json_path, html_path = save_monitoring_report(report, Path(args.output_dir))

    print("=" * 45)
    print("  Monitoring Snapshot — Equipo 8")
    print("=" * 45)
    print(f"Estado features     : {report['status']}")
    print(f"Features evaluadas  : {report['summary']['total_features_evaluated']}")
    print(f"Features con drift  : {report['summary']['features_with_drift']}")
    print(f"Drift ratio         : {report['summary']['drift_ratio']}")
    print("-" * 45)
    pred = report["prediction_drift"]
    print(f"Prediction drift    : {pred.get('status', 'N/A')}")
    if pred.get("score") is not None:
        print(f"  TVD score         : {pred['score']} (umbral: {pred['threshold']})")
        if pred.get("reference_distribution"):
            print(f"  Referencia        : {pred['reference_distribution']}")
        if pred.get("current_distribution"):
            print(f"  Actual            : {pred['current_distribution']}")
    else:
        print(f"  Motivo            : {pred.get('motivo', '')}")
    print("=" * 45)
    print(f"Reporte JSON: {json_path}")
    print(f"Reporte HTML: {html_path}")


if __name__ == "__main__":
    main()
