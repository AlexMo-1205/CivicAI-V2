# ── Stage 1 : builder ────────────────────────────────────────────────────────
# Installe les dépendances dans un environnement isolé
FROM python:3.12-slim AS builder

WORKDIR /app

# Installe uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copie les fichiers de dépendances
COPY pyproject.toml .
COPY uv.lock* .

# Installe les dépendances dans /app/.venv
RUN uv sync --frozen --no-dev


# ── Stage 2 : runtime ─────────────────────────────────────────────────────────
# Image finale légère sans les outils de build
FROM python:3.12-slim

WORKDIR /app

# Copie le venv depuis le builder
COPY --from=builder /app/.venv /app/.venv

# Copie le code et les données
COPY agent.py .
COPY api.py .
COPY static/ ./static/
COPY docs/ ./docs/
COPY ingest.py .

# Variables d'environnement
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose le port FastAPI
EXPOSE 8000

# Génère la base vectorielle puis lance le serveur
CMD ["sh", "-c", "python ingest.py && uvicorn api:app --host 0.0.0.0 --port 8000"]
