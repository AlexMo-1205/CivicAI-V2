"""Sentence-Transformer embedding model (lazy singleton)."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from civicai.config import SETTINGS


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(SETTINGS.embed_model)
