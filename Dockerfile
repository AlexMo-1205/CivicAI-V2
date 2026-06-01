# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
COPY uv.lock* .
COPY README.md .
COPY src/ ./src/

# Install into /app/.venv. Use the CPU torch index declared in pyproject.toml
# (no nvidia-* / triton wheels — bge-m3 and the reranker run on CPU).
# Purge the uv cache in the SAME layer so it isn't frozen into the intermediate
# image; only /app/.venv is copied into the runtime stage.
RUN uv sync --frozen --no-dev \
    && rm -rf /root/.cache/uv


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY static/ ./static/
COPY docs/ ./docs/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app/src"

# Cache bge-m3 (~2 GB) + bge-reranker-v2-m3 (~1 GB) under /root/.cache/huggingface.
# docker-compose mounts ./.hf_cache there so the ~3 GB download happens once
# on first start and is reused across container restarts.
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000

# Build the vector store, then boot the API.
CMD ["sh", "-c", "python scripts/ingest.py && uvicorn civicai.api.app:app --host 0.0.0.0 --port 8000"]
