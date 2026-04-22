# mlops-proyecto-final-equipo8
Elaborado por
ATEHORTUA ARANGO MATEO
AVILA PINTO CRISTIAN CAMILO
BARRERA CAMARGO PAULA CAROLINA


Docente
David Palacio Jimenez

Programa Académico
Especialización en Ciencia de Datos e Inteligencia Artificial
Facultad de Ingeniería
Universidad de Medellín

Año
2026

## Resumen del Proyecto

Este proyecto desarrolla una solución de machine learning end-to-end para predecir la aprobación o rechazo de solicitudes de crédito a partir de variables financieras del solicitante. El trabajo se construye siguiendo prácticas de MLOps: análisis exploratorio, calidad de datos, preprocesamiento, feature engineering, baseline de modelos, experiment tracking, orquestación, despliegue y monitoreo.

## Problema de Negocio

En un escenario hipotético de una entidad financiera, evaluar solicitudes de crédito de forma manual puede ser lento, inconsistente y costoso. Un modelo predictivo puede apoyar una etapa inicial de preevaluación para:

- priorizar solicitudes con alta probabilidad de aprobación,
- identificar perfiles con alto riesgo de rechazo,
- reducir tiempos de análisis,
- mejorar consistencia en decisiones preliminares.

El objetivo  es servir como herramienta de apoyo para acelerar el proceso de originación de crédito.

## Dataset Seleccionado

Dataset elegido: `loan_approval_dataset.csv`

Motivos de la selección:

- plantea un problema claro de clasificación binaria (`Approved` vs `Rejected`),
- es adecuado para comparar modelos baseline y luego desplegar una API de predicción,
- presenta un caso realista de calidad de datos que permite justificar reglas de limpieza y validación.

Decision final: se eligió este dataset sobre otras alternativas porque permite construir un proyecto completo de MLOps y mejor trazabilidad de métricas de negocio.

## Objetivo del Modelo

Construir un modelo de clasificación que prediga el estado de una solicitud de crédito (`Approved` o `Rejected`) usando variables como ingreso anual, monto del préstamo, plazo, puntaje crediticio y valor de activos.

## Métricas de Éxito

Las métricas principales del proyecto serán:

- `F1-score`: métrica principal de equilibrio entre precisión y recall.
- `Recall` para la clase Rejected: importante para detectar solicitudes con mayor riesgo de no aprobación.
- `ROC-AUC`: mide la capacidad general de discriminación del modelo.

Metas iniciales para el MVP:

- `F1-score >= 0.85`
- `Recall (Rejected) >= 0.80`
- `ROC-AUC >= 0.90`


## Alcance del Proyecto

### MVP

El MVP incluirá:

- selección y análisis del dataset,
- EDA ,
- reglas de calidad de datos,
- preprocesamiento reproducible,
- feature engineering,
- entrenamiento de modelos baseline,
- tracking de experimentos con MLflow,
- pipeline de entrenamiento orquestado,
- despliegue básico como API,
- propuesta de monitoreo.


## Timeline y Responsables

| Fase / Tarea | Responsable principal | Estado esperado |
|---|---|---|
| Selección del dataset y definición del problema | Equipo | Completado |
| EDA, calidad de datos, preprocesamiento y feature engineering | Paula Carolina Barrera Camargo | En desarrollo |
| Baseline y tests de datos | Paula Carolina Barrera Camargo | En desarrollo |
| Experiment tracking con MLflow | Mateo Atehortua Arango | Pendiente |
| Pipeline de entrenamiento y orquestación | Mateo Atehortua Arango | Pendiente |
| API de despliegue y validación de inputs | Cristian Camilo Avila Pinto | Pendiente |
| Propuesta de monitoreo y documentación de operación | pendiente | Pendiente |
| Integración final, pruebas end-to-end y revisión README | Equipo | Pendiente |

## Estructura del Repositorio

```text
mlops-proyecto-final-equipo8/
├── README.md
├── pyproject.toml
├── .github/
├── configs/
├── data/
├── docs/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── api/
│   ├── monitoring/
│   └── pipeline/
└── tests/
```

## para ejecutar

### 1. Preparar el entorno local

se debe instalar
- Python 3.13+
- Git
- `uv` (gestor de paquetes moderno)

### 2. Clonar el repositorio

```bash
git clone https://github.com/paula020/mlops-proyecto-final-equipo8.git
cd mlops-proyecto-final-equipo8
```

### 3. Configurar el entorno virtual y dependencias

```bash
uv venv
source .venv/bin/activate  
# PowerShell:
.venv\Scripts\Activate.ps1

uv sync
```

### 4. Ejecutar los tests unitarios

Valida que todos los módulos funcionan correctamente:

```bash
uv run pytest -q
```

Resultado esperado: 6 passed

### 5. Ejecutar EDA (Exploratory Data Analysis)

```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

El notebook genera:
- Análisis univariado y bivariado
- Detección de calidad de datos
- Correlaciones y distribuciones
- Recomendaciones de preprocesamiento

### 6. Ejecutar Baseline de Modelos

```bash
uv run jupyter notebook notebooks/02_baseline.ipynb
```

El notebook genera:
- Carga y limpieza de datos
- Feature engineering
- Entrenamiento de Regresión Logística y Random Forest
- Comparación de métricas (accuracy, F1, recall, ROC-AUC)
- Conclusiones y candidato para MLflow

### 7. Verificar estructura de módulos

Para validar que los módulos están importando correctamente:

```bash
uv run python -c "from src.data import load_dataset; from src.features import add_financial_features; from src.models import train_baseline_models; print('✓ Todos los módulos importan correctamente')"
```

## Hallazgos iniciales (Fase 1)

### Calidad del Dataset

- **Filas**: 4,269 solicitudes
- **Columnas**: 13 variables originales
- **Duplicados**: 0 (sin duplicación)
- **Nulos**: 0 (dataset completo)
- **Anomalías detectadas**: 28 valores negativos en `residential_assets_value` (corregidos en preprocesamiento)

### Variables creadas en Feature Engineering

- `total_assets`: suma de todos los tipos de activos (corregida para negativos)
- `loan_to_income_ratio`: monto del préstamo / ingreso anual
- `loan_to_assets_ratio`: monto del préstamo / total de activos
- `net_worth_proxy`: patrimonio neto (activos - préstamo)

### Desempeño del Baseline

| Modelo | Accuracy | F1 | Recall | ROC-AUC |
|---|---|---|---|---|
| Regresión Logística | 0.918 | 0.933 | 0.921 | 0.973 |
| **Random Forest** | **0.999** | **0.999** | **1.000** | **1.000** |

**Conclusión**: Random Forest es el candidato seleccionado para la siguiente fase (Experiment Tracking con MLflow).

### 8. Levantar MLflow
```bash
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artefactos --host 127.0.0.1 --port 5000
```
UI disponible en: http://127.0.0.1:5000

### 9. Correr experimentos

```bash
# Validar configuración MLflow
python scripts/validate_mlflow_setup.py

# Experimentos baseline
python scripts/experimentos_iniciales.py

# Tuning de hiperparámetros
python scripts/hyperparameter_tuning.py
```