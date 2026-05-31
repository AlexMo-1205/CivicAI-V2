"""Unit test: similarity-threshold routing in search_docs + dispatcher behavior."""
from unittest.mock import MagicMock

from civicai.tools import dispatcher
from civicai.tools import search_docs as sd_mod
from civicai.tools import web_search as ws_mod


def _fake_query_results(distances):
    n = len(distances)
    return {
        "documents": [[f"doc-{i}" for i in range(n)]],
        "metadatas": [[{"source": f"s{i}.txt"} for i in range(n)]],
        "distances": [distances],
    }


def _fake_embeddings():
    """Stub EmbeddingProvider — returns a deterministic 1-dim vector."""
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.0]
    embeddings.embed_documents.return_value = [[0.0]]
    return embeddings


def test_below_threshold_triggers_fallback_message(monkeypatch):
    fake_collection = MagicMock()
    # Distances 0.7, 0.8 -> scores 0.3, 0.2 -> avg 0.25 (< 0.5)
    fake_collection.query.return_value = _fake_query_results([0.7, 0.8])
    monkeypatch.setattr(sd_mod, "get_collection", lambda: fake_collection)
    monkeypatch.setattr(sd_mod, "get_embeddings", _fake_embeddings)

    out = sd_mod.search_docs("anything")
    assert "web_search" in out
    assert "Aucun document pertinent" in out


def test_above_threshold_returns_formatted_docs(monkeypatch):
    fake_collection = MagicMock()
    # Distance 0.1 -> score 0.9 (>= 0.5)
    fake_collection.query.return_value = _fake_query_results([0.1])
    monkeypatch.setattr(sd_mod, "get_collection", lambda: fake_collection)
    monkeypatch.setattr(sd_mod, "get_embeddings", _fake_embeddings)

    out = sd_mod.search_docs("visa")
    assert "doc-0" in out
    assert "Source: s0.txt" in out
    assert "web_search" not in out


def test_web_search_uses_tavily_client(monkeypatch):
    fake_tavily = MagicMock()
    fake_tavily.search.return_value = {
        "results": [{"title": "T1", "url": "https://x", "content": "snip"}],
    }
    monkeypatch.setattr(ws_mod, "get_tavily", lambda: fake_tavily)

    out = ws_mod.web_search("thailand visa 2026")
    fake_tavily.search.assert_called_once()
    assert "T1" in out and "https://x" in out


def test_dispatcher_unknown_tool():
    assert dispatcher.run_tool("nope", {}) == "Outil inconnu : nope"


def test_dispatcher_routes_to_handler(monkeypatch):
    monkeypatch.setitem(dispatcher.HANDLERS, "search_docs",
                        lambda **kw: f"stub:{kw['query']}")
    assert dispatcher.run_tool("search_docs", {"query": "abc"}) == "stub:abc"
