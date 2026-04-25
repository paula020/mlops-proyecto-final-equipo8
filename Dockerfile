# ─────────────────────────────────────────────────────────
#  Dockerfile — Credit Approval API
#  Proyecto MLOps - Equipo 8
#
#  Build:  docker build -t credit-approval-api .
#  Run:    docker run -p 8000:8000 --env-file .env credit-approval-api
# ─────────────────────────────────────────────────────────

# ── Stage 1: base con Python ──────────────────────────────
FROM python:3.13-slim AS base

# Evita que Python genere .pyc y bufferea stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
    && rm -rf /var/lib/apt/lists/*


# ── Stage 2: instalación de dependencias ─────────────────
FROM base AS builder

# Instalar uv (gestor de paquetes del proyecto)
RUN pip install --no-cache-dir uv

# Copiar archivos de dependencias
COPY pyproject.toml .
COPY uv.lock        .

# Instalar dependencias en un entorno virtual
RUN uv sync --frozen --no-dev


# ── Stage 3: imagen final ─────────────────────────────────
FROM base AS final

WORKDIR /app

# Copiar entorno virtual desde el builder
COPY --from=builder /app/.venv /app/.venv

# Asegurar que el venv esté en el PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copiar el código fuente del proyecto
COPY src/       ./src/
COPY config/    ./config/

# Copiar datos si existen (el CSV del dataset)
# En producción esto vendría de un volumen o S3
COPY data/      ./data/

# Variables de entorno por defecto (sobreescribibles con --env-file)
ENV MLFLOW_TRACKING_URI="http://mlflow:5000" \
    MODEL_NAME="credit_approval_model"       \
    MODEL_STAGE="Production"                 \
    DATA_PATH="data/raw/loan_approval_dataset.csv"

# Puerto expuesto
EXPOSE 8000

# Health check — Docker verifica que la API responda
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de arranque
CMD ["uvicorn", "src.api.main:app", \
     "--host", "127.0.0.1",           \
     "--port", "8000",              \
     "--log-level", "info"]