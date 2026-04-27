# Guía de Deployment — Credit Approval API
**Proyecto MLOps · Equipo 8**
Especialización en Ciencia de Datos e IA — Universidad de Medellín


## 1. Deployment local (sin Docker)

### 1.1 Clonar y preparar entorno

```bash
git clone https://github.com/paula020/mlops-proyecto-final-equipo8.git
cd mlops-proyecto-final-equipo8

uv venv
# Linux/Mac:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

uv sync
```

### 1.2 Verificar que los módulos importan correctamente

```bash
uv run python -c "
from src.data import load_dataset
from src.features import add_financial_features
from src.models import train_baseline_models
"
```

### 1.3 Ejecutar tests unitarios

```bash
uv run pytest tests/ -v
```

Resultado esperado: todos los tests en `tests/unit/` pasan (preprocessing, feature engineering, monitoring).

### 1.4 Ejecutar snapshot de monitoreo 

```bash
# Modo simulación (no requiere datos de producción)
python scripts/run_monitoring_snapshot.py

# Modo real (con batch de producción)
python scripts/run_monitoring_snapshot.py \
  --reference-path data/loan_approval_dataset.csv \
  --current-path data/current_batch.csv \
  --output-dir artifacts/monitoring
```

Salidas generadas:
- `artifacts/monitoring/drift_report.json`
- `artifacts/monitoring/drift_report.html`

---

## 2. Deployment con Docker Compose 

Levanta dos servicios: MLflow tracking server (puerto 5000) y FastAPI (puerto 8000)



```bash
docker compose up --build
```

Esto construye la imagen de la API usando el `Dockerfile` multi-stage y descarga la imagen de MLflow.

### 2.2 Uso habitual

```bash
# Levantar en background
docker compose up -d

# Ver logs en tiempo real
docker compose logs -f api
docker compose logs -f mlflow

# Bajar todo
docker compose down
```

### 2.3 Verificar que la API está activa

```bash
# Health check
curl http://localhost:8000/health

# PowerShell
Invoke-RestMethod -Uri http://localhost:8000/health | ConvertTo-Json
```

Respuesta esperada:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "credit_approval_model",
  "model_stage": "Production"
}
```

---

## 3. Endpoints de la API

La documentación interactiva (Swagger UI) está disponible en:
**http://localhost:8000/docs**

### GET `/health`
Estado del sistema y del modelo cargado.

### GET `/model/info`
Información del modelo en producción: nombre, stage, URI.

### POST `/predict`
Predicción individual.

**Body :**
```json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 1200000,
  "loan_amount": 350000,
  "loan_term": 12,
  "cibil_score": 720,
  "residential_assets_value": 500000,
  "commercial_assets_value": 200000,
  "luxury_assets_value": 150000,
  "bank_asset_value": 180000
}
```

**Respuesta:**
```json
{
  "approved": true,
  "probability": 0.9919,
  "model_version": "file:///mlflow/mlruns/1/models/..."
}
```

### POST `/predict/batch`
Misma estructura pero con una lista de solicitudes.

---

## 4. Variables de entorno

El archivo `.env.example` documenta todas las variables disponibles. Las principales son:

| Variable | Default | Descripción |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | URI del tracking server |
| `MODEL_NAME` | `credit_approval_model` | Nombre del modelo en el registry |
| `MODEL_STAGE` | `Production` | Stage del modelo a cargar |
| `MODEL_URI` | `models:/credit_approval_model/Production` | URI directa (sobreescribe name+stage) |
| `DATA_PATH` | `data/loan_approval_dataset.csv` | Dataset de referencia |

---

## 5. Estructura de volúmenes Docker

```
./mlruns   →  /mlflow/mlruns   (artefactos y modelos MLflow, montado read-only en la API)
./data     →  /app/data        (dataset CSV, montado read-only)
```

---

## 6. Registro y entrenamiento de modelos

### 6.1 Levantar MLflow server 

```bash
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

UI en: http://127.0.0.1:5000

### 6.2 Entrenar y registrar modelo

```bash
python train_simple.py
```

Registra la versión del modelo en el MLflow registry. Luego promover a Production desde la UI o con:

```bash
python scripts/model_registry.py
```

---

## 7. Code quality y hooks

El proyecto usa ruff (linter) y black (formatter), configurados en `pyproject.toml`.



Para ejecutarlos manualmente sobre todos los archivos:

```bash
uv run pre-commit run --all-files
```

Para solo lint y format:

```bash
uv run ruff check src/ tests/ scripts/
uv run black src/ tests/ scripts/
```

---

