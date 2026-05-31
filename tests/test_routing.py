"""Routing tests: web_search fallback fires on top reranked score below threshold."""
from unittest.mock import MagicMock

from civicai.config import SETTINGS
from civicai.rag.retrieval import Candidate
from civicai.tools import dispatcher
from civicai.tools import search_docs as sd_mod
from civicai.tools import web_search as ws_mod


def _candidates_with_rerank_scores(scores):
    """Stub Candidate list pre-scored by the reranker (already sorted desc)."""
    out = []
    for i, s in enumerate(sorted(scores, reverse=True)):
        c = Candidate(
            text=f"doc-{i}",
            source=f"s{i}.txt",
            chunk_id=i,
            dense_score=0.5,
            rerank_score=float(s),
        )
        out.append(c)
    return out


def _patch_pipeline(monkeypatch, reranked: list[Candidate]):
    """Stub the whole retrieve -> rerank pipeline to return `reranked` as-is."""
    # Retrieval returns the same set unranked; reranker just hands it back sorted.
    monkeypatch.setattr(sd_mod, "retrieve", lambda query, k: reranked)
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = reranked
    monkeypatch.setattr(sd_mod, "get_reranker", lambda: fake_reranker)
    return fake_reranker


def test_below_threshold_triggers_fallback_message(monkeypatch):
    # Top reranked score 0.30 -> below the 0.5 threshold (placeholder)
    reranked = _candidates_with_rerank_scores([0.30, 0.25, 0.10])
    _patch_pipeline(monkeypatch, reranked)

    out = sd_mod.search_docs("anything")
    assert "Aucun document pertinent" in out
    assert "web_search" in out
    # The fallback message reports the top score, not an average
    assert "0.3" in out


def test_above_threshold_returns_formatted_docs(monkeypatch):
    reranked = _candidates_with_rerank_scores([0.92, 0.81, 0.66])
    _patch_pipeline(monkeypatch, reranked)

    out = sd_mod.search_docs("visa")
    assert "doc-0" in out
    assert "Source: s0.txt" in out
    assert "web_search" not in out


def test_threshold_uses_top_score_not_average(monkeypatch):
    """A single very high score should NOT be drowned out by lower neighbors."""
    # Mean of these is well below 0.5, but the top is 0.95 — should NOT fall back.
    reranked = _candidates_with_rerank_scores([0.95, 0.20, 0.10, 0.05])
    _patch_pipeline(monkeypatch, reranked)

    out = sd_mod.search_docs("specific question")
    assert "web_search" not in out
    assert "doc-0" in out


def test_empty_retrieval_falls_back(monkeypatch):
    monkeypatch.setattr(sd_mod, "retrieve", lambda query, k: [])
    out = sd_mod.search_docs("anything")
    assert "Aucun document pertinent" in out


def test_search_docs_passes_config_top_k_and_top_n(monkeypatch):
    captured = {}

    def fake_retrieve(query, k):
        captured["k"] = k
        return _candidates_with_rerank_scores([0.9])

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = _candidates_with_rerank_scores([0.9])

    monkeypatch.setattr(sd_mod, "retrieve", fake_retrieve)
    monkeypatch.setattr(sd_mod, "get_reranker", lambda: fake_reranker)

    sd_mod.search_docs("q")
    assert captured["k"] == SETTINGS.retrieve_top_k
    _, kwargs = fake_reranker.rerank.call_args[:2], fake_reranker.rerank.call_args.kwargs
    assert kwargs["top_n"] == SETTINGS.rerank_top_n


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
    monkeypatch.setitem(
        dispatcher.HANDLERS, "search_docs", lambda **kw: f"stub:{kw['query']}"
    )
    assert dispatcher.run_tool("search_docs", {"query": "abc"}) == "stub:abc"
