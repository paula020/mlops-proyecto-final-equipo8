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
| EDA, calidad de datos, preprocesamiento y feature engineering | Paula Carolina Barrera Camargo | Completado |
| Baseline y tests de datos | Paula Carolina Barrera Camargo | Completado |
| Experiment tracking con MLflow | Mateo Atehortua Arango | Completado |
| Pipeline de entrenamiento y orquestación | Mateo Atehortua Arango | Completado |
| API de despliegue y validación de inputs | Cristian Camilo Avila Pinto | Completado|
| Propuesta de monitoreo y documentación de operación | Paula Carolina Barrera Camargo | Completado |
| Integración final, pruebas end-to-end y revisión README | Equipo | Completado|

## Estructura del Repositorio

```text
mlops-proyecto-final-equipo8/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
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
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
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
# Linux/Mac:
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Windows PowerShell (usar uv run):
uv run mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```
UI disponible en: http://localhost:5000

### 9. Levantar Prefect
```bash
prefect server start
```
UI disponible en: http://127.0.0.1:4200

> Se requiere 3 terminales corriendo en paralelo:
> 1. MLflow server
> 2. Prefect server  
> 3. Ejecución de scripts

### 10. Correr experimentos

```bash
# Validar configuración MLflow
python scripts/validate_mlflow_setup.py

# Experimentos baseline
python scripts/experimentos_iniciales.py

# Tuning de hiperparámetros
python scripts/hyperparameter_tuning.py
```

---

## Fase 4: Deployment — Containerización y API REST

La Fase 4 empaqueta la API de predicción y MLflow en contenedores Docker para que el sistema sea reproducible y portable.

### Archivos creados en esta fase

| Archivo | Ubicación | Descripción |
|---|---|---|
| `Dockerfile` | raíz del proyecto | Build multi-stage con `uv` — construye la imagen de la API |
| `docker-compose.yml` | raíz del proyecto | Orquesta la API y MLflow en la misma red |
| `.dockerignore` | raíz del proyecto | Excluye `.venv`, notebooks y logs del build |
| `main.py` | `src/api/` | App FastAPI con todos los endpoints |
| `schemas.py` | `src/api/` | Validación de inputs y outputs con Pydantic |

### Prerequisitos

- Docker Desktop instalado y corriendo (ícono verde en la barra de tareas)
- El dataset en `data/raw/loan_approval_dataset.csv`

### Paso 1 — Registrar el modelo en MLflow

Antes de levantar Docker, el modelo debe estar entrenado y registrado. Con MLflow corriendo localmente:

```bash
# Windows PowerShell:
uv run mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

En otra terminal, entrenar y registrar el modelo:

```bash
uv run python train_simple.py
```

Luego abrir http://localhost:5000 → **Models** → `credit_approval_model` → cambiar Stage a **Production**.

Una vez promovido, detener el MLflow local (Ctrl+C) — Docker levantará el suyo propio con los mismos datos.

### Paso 2 — Levantar todo con Docker Compose

```bash
# Primera vez (construye la imagen, tarda 2-3 minutos):
docker compose up --build

# Veces siguientes (sin rebuild):
docker compose up -d

# Si se modificó código, siempre rebuild:
docker compose up --build

# Bajar todos los contenedores:
docker compose down
```

> **Nota Windows PowerShell:** el operador `\` para continuar líneas no funciona. Usar todo en una sola línea o con el backtick `` ` ``.

### Paso 3 — Verificar que los servicios están activos

```bash
docker compose ps
```

Resultado esperado:

```
NAME                  STATUS          PORTS
credit-approval-api   Up (healthy)    0.0.0.0:8000->8000/tcp
mlflow-server         Up              0.0.0.0:5000->5000/tcp
```

Ver logs en tiempo real:

```bash
docker compose logs -f api      # solo la API
docker compose logs -f mlflow   # solo MLflow
docker compose logs -f          # ambos
```

### Paso 4 — Probar la API

**Opción A — Swagger UI (recomendado, desde el navegador):**

Abrir http://localhost:8000/docs → clic en `POST /predict` → `Try it out` → `Execute`

**Opción B — PowerShell:**

```powershell
# Health check
Invoke-RestMethod -Uri http://localhost:8000/health

# Predicción individual
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body '{
  "no_of_dependents": 2,
  "income_annum": 5800000,
  "loan_amount": 12000000,
  "loan_term": 10,
  "cibil_score": 720,
  "residential_assets_value": 8000000,
  "commercial_assets_value": 2000000,
  "luxury_assets_value": 500000,
  "bank_asset_value": 3000000,
  "education": "Graduate",
  "self_employed": "No"
}'
```

**Opción C — Python:**

```bash
uv run python - <<'EOF'
import requests

print(requests.get("http://localhost:8000/health").json())

payload = {
    "no_of_dependents": 2, "income_annum": 5800000,
    "loan_amount": 12000000, "loan_term": 10, "cibil_score": 720,
    "residential_assets_value": 8000000, "commercial_assets_value": 2000000,
    "luxury_assets_value": 500000, "bank_asset_value": 3000000,
    "education": "Graduate", "self_employed": "No"
}
print(requests.post("http://localhost:8000/predict", json=payload).json())
EOF
```

### Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Info básica de la API |
| GET | `/health` | Estado del modelo cargado |
| GET | `/model/info` | Parámetros del modelo activo |
| POST | `/predict` | Predicción individual |
| POST | `/predict/batch` | Predicciones en lote (máx. 100) |

### Respuesta esperada de `/predict`

```json
{
  "aprobado": true,
  "probabilidad": 0.8732,
  "etiqueta": "Approved",
  "score": 720
}
```

### Modo degradado

Si el modelo no está en stage `Production` al arrancar, la API levanta igual pero:
- `/health` retorna `"status": "degradado"`
- `/predict` y `/predict/batch` retornan error `503`
- `/` y `/model/info` siguen funcionando

Para solucionarlo: promover el modelo en http://localhost:5000 → Models → Production, luego `docker compose restart api`.

### URLs de los servicios con Docker

| Servicio | URL |
|---|---|
| API de predicción | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## Fase 5: Monitoreo (Diseño inicial)

| Archivo | Responsabilidad |
|---|---|
| `src/monitoring/drift_detector.py` | PSI para features numéricas, TVD para categóricas y predicciones |
| `src/monitoring/report.py` | Exporta reporte JSON + HTML legible |
| `scripts/run_monitoring_snapshot.py` | CLI ejecutable con modo simulación y modo real |
| `docs/monitoring_proposal.md` | Propuesta: riesgos, umbrales y criterios de decisión |

### Modo simulación 

Simula un batch con drift moderado: `loan_amount` +20%, `cibil_score` -25 pts,
`self_employed` 100% Yes, y deterioro de cartera (más rechazos).

```bash
python scripts/run_monitoring_snapshot.py
```

### Modo real 

```bash
python scripts/run_monitoring_snapshot.py \
  --reference-path data/loan_approval_dataset.csv \
  --current-path data/current_batch.csv \
  --output-dir artifacts/monitoring
```

### Salidas

- `artifacts/monitoring/drift_report.json` — reporte  por feature
- `artifacts/monitoring/drift_report.html` — reporte visual en HTML


| Indicador | Umbral | Acción recomendada |
|---|---|---|
| PSI feature numérica | >= 0.20 | Revisar distribución; posible retraining |
| TVD feature categórica | >= 0.15 | Investigar sesgo de selección |
| TVD prediction drift | >= 0.10 | Alerta de degradación del modelo |
| Drift ratio > 30% | > 0.30 | Abrir incidente de datos |
| Drift en features críticas (`cibil_score`, `income_annum`) 2 runs consecutivos | — | Programar retraining |

## Fase 6: Testing y Best Practices

### Testing

Se implementaron tests unitarios para los módulos principales del proyecto.
Todos los tests se ejecutan con:

```bash
uv run pytest tests/ -v
```

| Archivo de test | Qué valida |
|---|---|
| `tests/unit/test_preprocessing.py` | `split_features_target` y `build_preprocessor` |
| `tests/unit/test_feature_engineering.py` | Creación de features derivadas (`total_assets`, ratios, etc.) |
| `tests/unit/test_data_quality.py` | Reglas de calidad sobre el dataset |
| `tests/unit/test_baseline.py` | Entrenamiento de modelos baseline |
| `tests/unit/test_monitoring.py` | PSI, TVD y prediction drift del módulo de monitoreo |

### Code Quality

El proyecto usa ruff como linter (equivalente a flake8 + isort) y black como formatter, ambos configurados en `pyproject.toml`.

```bash
# Revisar errores de estilo e imports
uv run ruff check src/ tests/ scripts/

# Formatear automáticamente
uv run black src/ tests/ scripts/
```

### Pre-commit Hooks

Los hooks se instalan una sola vez y se ejecutan en cada `git commit`:

```bash
# Instalar 
uv run pre-commit install

# Ejecutar manualmente sobre todos los archivos
uv run pre-commit run --all-files
```

Los hooks configurados en `.pre-commit-config.yaml` ejecutan ruff, black y validaciones básicas de archivos YAML, conflictos de merge, espacios sobrantes

### Documentación

| Documento | Contenido |
|---|---|
| `README.md` | Guía del proyecto |
| `docs/monitoring_proposal.md` | Propuesta de monitoreo con umbrales y criterios |
| `docs/deployment_guide.md` | Guía paso a paso para deployment local y con Docker |
| `docs/FASE4_README.md` | Detalle técnico de la API y Docker |
| `http://localhost:8000/docs` | Documentación interactiva de la API Swagger UI, disponible con la API corriendo |
