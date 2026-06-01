"""Unit tests for the embedding-provider adapter.

These intentionally avoid loading the real bge-m3 weights (~2 GB) — the
sentence-transformers model is stubbed so the test runs in milliseconds.
"""
from unittest.mock import MagicMock

import numpy as np

from civicai.config import SETTINGS
from civicai.rag.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddings,
    get_embeddings,
)


def _stub_model(dim: int):
    """Fake SentenceTransformer that returns `dim`-shaped zero vectors."""
    model = MagicMock()

    def encode(texts, **_kwargs):
        if isinstance(texts, str):
            return np.zeros(dim, dtype=float)
        return np.zeros((len(texts), dim), dtype=float)

    model.encode.side_effect = encode
    return model


def test_provider_obeys_protocol():
    provider = SentenceTransformerEmbeddings("any/model", dim=8)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.name == "any/model"
    assert provider.dim == 8


def test_embed_query_dim_matches_config():
    provider = SentenceTransformerEmbeddings(SETTINGS.embed_model, SETTINGS.embed_dim)
    provider._model = _stub_model(SETTINGS.embed_dim)
    vector = provider.embed_query("test query")
    assert len(vector) == SETTINGS.embed_dim


def test_embed_documents_dim_matches_config():
    provider = SentenceTransformerEmbeddings(SETTINGS.embed_model, SETTINGS.embed_dim)
    provider._model = _stub_model(SETTINGS.embed_dim)
    vectors = provider.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == SETTINGS.embed_dim for v in vectors)


def test_get_embeddings_returns_singleton():
    a = get_embeddings()
    b = get_embeddings()
    assert a is b
    assert a.name == SETTINGS.embed_model
    assert a.dim == SETTINGS.embed_dim
