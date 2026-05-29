# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
COPY uv.lock* .
COPY README.md .
COPY src/ ./src/

RUN uv sync --frozen --no-dev


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

EXPOSE 8000

# Build the vector store, then boot the API.
CMD ["sh", "-c", "python scripts/ingest.py && uvicorn civicai.api.app:app --host 0.0.0.0 --port 8000"]
