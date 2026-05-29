"""Unit test: search_docs falls back to web_search below the 0.5 score threshold."""
import importlib
from unittest.mock import MagicMock


def _load_agent():
    if "agent" in importlib.sys.modules:
        del importlib.sys.modules["agent"]
    return importlib.import_module("agent")


def _query_returning(distances):
    n = len(distances)
    return {
        "documents": [[f"doc-{i}" for i in range(n)]],
        "metadatas": [[{"source": f"s{i}.txt"} for i in range(n)]],
        "distances": [distances],
    }


def _patched_encode():
    enc = MagicMock()
    enc.tolist.return_value = [0.0]
    return enc


def test_below_threshold_triggers_fallback_message(stub_external_modules, monkeypatch):
    agent = _load_agent()

    fake_collection = MagicMock()
    # Distances 0.7, 0.8 -> scores 0.3, 0.2 -> avg 0.25 (< 0.5)
    fake_collection.query.return_value = _query_returning([0.7, 0.8])
    monkeypatch.setattr(agent, "collection", fake_collection)
    monkeypatch.setattr(agent, "embedder", MagicMock(encode=lambda q: _patched_encode()))

    out = agent.run_tool("search_docs", {"query": "anything"})
    assert "web_search" in out
    assert "Aucun document pertinent" in out


def test_above_threshold_returns_formatted_docs(stub_external_modules, monkeypatch):
    agent = _load_agent()

    fake_collection = MagicMock()
    # Distance 0.1 -> score 0.9 (>= 0.5)
    fake_collection.query.return_value = _query_returning([0.1])
    monkeypatch.setattr(agent, "collection", fake_collection)
    monkeypatch.setattr(agent, "embedder", MagicMock(encode=lambda q: _patched_encode()))

    out = agent.run_tool("search_docs", {"query": "visa"})
    assert "doc-0" in out
    assert "Source: s0.txt" in out
    assert "web_search" not in out


def test_web_search_uses_tavily_client(stub_external_modules, monkeypatch):
    agent = _load_agent()

    fake_tavily = MagicMock()
    fake_tavily.search.return_value = {
        "results": [
            {"title": "T1", "url": "https://x", "content": "snippet"},
        ]
    }
    monkeypatch.setattr(agent, "tavily", fake_tavily)

    out = agent.run_tool("web_search", {"query": "thailand visa 2026"})
    fake_tavily.search.assert_called_once()
    assert "T1" in out and "https://x" in out
