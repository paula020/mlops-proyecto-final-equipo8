# Propuesta Fase 5: Monitoreo Diseño Inicial

- monitoreo de drift en datos de entrada,
- monitoreo de drift en distribucion de predicciones,
- reporte local reproducible (JSON y HTML).


## Riesgos a monitorear

1. Data Drift cambios en distribucion de features numericas y categoricas
2. Prediction Drift: cambios en proporcion de solicitudes aprobadas/rechazadas
3. Riesgo operativo: degradacion silenciosa del modelo por cambios del negocio

## Diseno propuesto

### 1) Drift de features numericas

- Metodo es PSI (Population Stability Index)
- Umbral inicial es PSI >= 0.20 indica drift relevante
- Features : income_annum, loan_amount, cibil_score, bank_asset_value

### 2) Drift de features categoricas

- Metodo: TVD (Total Variation Distance)
- Umbral inicial: TVD >= 0.15 indica drift relevante
- Features clave: education, self_employed

### 3) Drift de predicciones

- Metodo es TVD sobre la columna `loan_status` (Approved / Rejected)
- Umbral inicial es TVD >= 0.10 indica cambio relevante en la distribucion de salidas
- Implementado en `detect_prediction_drift()` dentro de `drift_detector.py`
- Cuando no hay etiquetas reales en produccion, la funcion retorna `status: no_disponible`


## Artefactos

- src/monitoring/drift_detector.py: PSI (numericas), TVD (categoricas y predicciones)
- src/monitoring/report.py: exporta reporte JSON + HTML con tabla por feature
- scripts/run_monitoring_snapshot.py: CLI con modo simulacion y modo real

## Ejecucion 
Caso 1: con simulacion (sin datos de produccion reales)

```bash
python scripts/run_monitoring_snapshot.py
```

Caso 2: con datos actuales reales

```bash
python scripts/run_monitoring_snapshot.py \
  --reference-path data/loan_approval_dataset.csv \
  --current-path data/current_batch.csv \
  --output-dir artifacts/monitoring
```

## Salidas esperadas

- artifacts/monitoring/drift_report.json
- artifacts/monitoring/drift_report.html

## Criterio de decision

- Si hay drift en mas del 30% de features evaluadas, abrir incidente de datos
- Si drift impacta features criticas por dos corridas consecutivas, programar retraining.
- Si no hay drift, mantener modelo y continuar monitoreo

