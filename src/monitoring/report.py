"""Generacion de reportes de monitoreo en JSON y HTML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _rows_to_html_table(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"<h3>{title}</h3><p>Sin datos para esta seccion.</p>"

    df = pd.DataFrame(rows)
    return f"<h3>{title}</h3>{df.to_html(index=False)}"


def save_monitoring_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Guarda reporte de drift como JSON y HTML."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "drift_report.json"
    html_path = output_path / "drift_report.html"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    summary = report.get("summary", {})
    html = f"""
<!doctype html>
<html lang=\"es\">
<head>
  <meta charset=\"utf-8\" />
  <title>Reporte de Monitoreo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
    h1, h2, h3 {{ margin-bottom: 0.6rem; }}
    .badge {{ display: inline-block; padding: 0.25rem 0.6rem; border-radius: 0.5rem; font-weight: 700; }}
    .estable {{ background: #dcfce7; color: #166534; }}
    .drift {{ background: #fee2e2; color: #991b1b; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 0.5rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Reporte de Monitoreo de Drift</h1>
  <p><strong>Fecha UTC:</strong> {report.get("generated_at_utc", "N/A")}</p>
  <p>
    <strong>Estado general:</strong>
    <span class=\"badge {'drift' if report.get('status') == 'drift_detectado' else 'estable'}\">{report.get('status', 'N/A')}</span>
  </p>

  <h2>Resumen</h2>
  <ul>
    <li>Features evaluadas: {summary.get('total_features_evaluated', 0)}</li>
    <li>Features con drift: {summary.get('features_with_drift', 0)}</li>
    <li>Proporción con drift: {summary.get('drift_ratio', 0.0)}</li>
    <li>Columnas faltantes: {', '.join(summary.get('missing_columns', [])) or 'Ninguna'}</li>
  </ul>

  {_rows_to_html_table(report.get('numeric_features', []), 'Drift en features numéricas')}
  {_rows_to_html_table(report.get('categorical_features', []), 'Drift en features categóricas')}
</body>
</html>
""".strip()

    html_path.write_text(html, encoding="utf-8")

    return json_path, html_path
