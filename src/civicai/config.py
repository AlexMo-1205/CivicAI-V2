"""Centralized configuration and constants.

Every magic value used elsewhere in the project lives here. Nothing else
hardcodes thresholds, model names, paths, or collection identifiers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# override=True so an empty shell var (e.g. `export ANTHROPIC_API_KEY=`)
# doesn't shadow a real value in .env.
load_dotenv(override=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value


@dataclass(frozen=True)
class Settings:
    # LLM
    anthropic_api_key: str | None
    model: str
    max_tokens: int

    # Embeddings + vector store
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    collection_name: str
    db_dir: Path

    # RAG routing
    similarity_threshold: float
    default_n_results: int

    # External search
    tavily_api_key: str | None
    tavily_max_results: int

    # Docs ingestion
    docs_dir: Path

    # API / static
    static_dir: Path
    static_index: Path


SETTINGS = Settings(
    anthropic_api_key=_env("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-5",
    max_tokens=4096,

    embed_model="all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=50,
    collection_name="civicai",
    db_dir=PROJECT_ROOT / "chroma_db",

    similarity_threshold=0.5,
    default_n_results=5,

    tavily_api_key=_env("TAVILY_API_KEY"),
    tavily_max_results=5,

    docs_dir=PROJECT_ROOT / "docs",

    static_dir=PROJECT_ROOT / "static",
    static_index=PROJECT_ROOT / "static" / "index.html",
)


def build_system_prompt(today: date | None = None) -> str:
    """Return the system prompt with today's date interpolated.

    The prompt itself is intentionally kept in French to preserve the exact
    behavior of the original agent (tone, language of replies, routing wording).
    """
    today = today or date.today()
    return f"""Tu es CivicAI, un assistant spécialisé dans les démarches administratives en Thaïlande.
La date d'aujourd'hui est {today}.

Tu aides les citoyens et expatriés à comprendre :
- Les visas et conditions de séjour
- Les permis de travail
- La résidence permanente
- Toute démarche administrative en Thaïlande

Règles :
- Utilise TOUJOURS search_docs en premier pour chercher dans ta base de connaissance
- Si search_docs ne trouve rien de pertinent (score < 0.5), utilise web_search
- Réponds de manière claire, structurée et bienveillante
- Cite toujours tes sources
- Si tu n'es pas sûr, dis-le et recommande de consulter un professionnel
- Réponds dans la langue de l'utilisateur (français ou anglais)
"""
