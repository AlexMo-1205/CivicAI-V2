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
    embed_dim: int
    chunk_size: int
    chunk_overlap: int
    min_chunk_split_chars: int  # docs at/under this size stay as a single chunk
    collection_name: str
    db_dir: Path

    # Retrieval pipeline
    retrieve_top_k: int          # dense candidates pulled from Chroma
    rerank_top_n: int            # kept after cross-encoder rerank
    reranker_model: str

    # RAG routing
    similarity_threshold: float          # legacy dense-only threshold, kept for back-compat
    rerank_routing_threshold: float      # top reranked (sigmoid) score below which we fall back to web_search
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

    # bge-m3: multilingual (100+ languages), 1024-dim dense vectors.
    # Vectors are L2-normalized and the ChromaDB collection uses cosine space.
    embed_model="BAAI/bge-m3",
    embed_dim=1024,
    chunk_size=500,
    chunk_overlap=50,
    # Short procedural docs (steps + costs + penalties) must stay in one chunk —
    # splitting orphans the corrective answer from its context.
    # ~700 tokens at the chars/4 heuristic.
    min_chunk_split_chars=2800,
    # Collection name includes the dim so a future model change can't accidentally
    # mix two embedding spaces in the same collection.
    collection_name="civicai_bge_m3_1024",
    db_dir=PROJECT_ROOT / "chroma_db",

    retrieve_top_k=40,                   # wide net for the dense recall pass
    # rerank_top_n bumped 6 -> 8: two adversarial ground-truth facts (penalty
    # numbers, condo-vs-land distinction) sit at ranks 7-8 after reranking on
    # a 21-doc corpus, so top_n<8 starves the LLM of the corrective context.
    rerank_top_n=8,
    reranker_model="BAAI/bge-reranker-v2-m3",

    similarity_threshold=0.5,
    # PLACEHOLDER — Phase 3 (RAGAS sweep) replaces this with the eval-selected value.
    rerank_routing_threshold=0.5,
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
