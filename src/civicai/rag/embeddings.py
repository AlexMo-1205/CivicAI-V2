"""Embedding-provider abstraction + sentence-transformers adapter.

The Protocol decouples retrieval / ingestion from the concrete embedder so that
swapping models (or moving to a hosted service) is one adapter + one config flip.
The default is BAAI/bge-m3 — a strong multilingual model (100+ languages) that
matches our French queries against French/Thai administrative documents.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol, runtime_checkable

from civicai.config import SETTINGS


@runtime_checkable
class EmbeddingProvider(Protocol):
    """What every embedder must expose."""

    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddings:
    """Local backend running any sentence-transformers model.

    Vectors are L2-normalized so we can use cosine distance in ChromaDB and
    interpret scores in roughly [0, 1] for relevant matches.
    """

    def __init__(self, model_name: str, dim: int):
        self.name = model_name
        self.dim = dim
        self._model = None  # lazy load

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> EmbeddingProvider:
    """Return the configured provider as a process-wide singleton."""
    return SentenceTransformerEmbeddings(SETTINGS.embed_model, SETTINGS.embed_dim)
