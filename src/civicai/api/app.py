"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from civicai.api.routes import router
from civicai.config import SETTINGS


def create_app() -> FastAPI:
    app = FastAPI(title="CivicAI API")
    app.mount("/static", StaticFiles(directory=str(SETTINGS.static_dir)), name="static")
    app.include_router(router)
    return app


app = create_app()
