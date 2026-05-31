"""Reranker unit tests (cross-encoder is mocked — no model load, no network)."""
import math
from unittest.mock import MagicMock

from civicai.config import SETTINGS
from civicai.rag.reranker import (
    BGECrossEncoderReranker,
    Reranker,
    _sigmoid,
    get_reranker,
)
from civicai.rag.retrieval import Candidate


def _stub_ce(scores):
    """Fake CrossEncoder whose predict() returns the given raw logits."""
    model = MagicMock()
    model.predict.return_value = list(scores)
    return model


def _candidates(texts):
    return [
        Candidate(text=t, source=f"s{i}.txt", chunk_id=i, dense_score=0.5)
        for i, t in enumerate(texts)
    ]


def test_protocol_conformance():
    r = BGECrossEncoderReranker("any/model")
    assert isinstance(r, Reranker)
    assert r.name == "any/model"


def test_sigmoid_bounds():
    assert _sigmoid(-100) < 0.001
    assert _sigmoid(100) > 0.999
    assert math.isclose(_sigmoid(0.0), 0.5)


def test_rerank_orders_by_score_descending():
    r = BGECrossEncoderReranker("stub")
    # Logits: doc-1 gets highest, then doc-0, then doc-2
    r._model = _stub_ce([1.0, 3.0, -2.0])

    cands = _candidates(["doc-0", "doc-1", "doc-2"])
    out = r.rerank("any", cands, top_n=3)

    assert [c.text for c in out] == ["doc-1", "doc-0", "doc-2"]
    # All scores must be in (0, 1) after sigmoid
    for c in out:
        assert 0.0 < c.rerank_score < 1.0
    # Strictly descending
    assert out[0].rerank_score > out[1].rerank_score > out[2].rerank_score


def test_rerank_truncates_to_top_n():
    r = BGECrossEncoderReranker("stub")
    r._model = _stub_ce([0.1, 0.2, 0.3, 0.4, 0.5])
    cands = _candidates([f"d{i}" for i in range(5)])

    out = r.rerank("any", cands, top_n=2)
    assert len(out) == 2
    assert out[0].text == "d4"   # highest logit
    assert out[1].text == "d3"


def test_rerank_empty_candidates_returns_empty():
    r = BGECrossEncoderReranker("stub")
    r._model = _stub_ce([])
    assert r.rerank("any", [], top_n=5) == []


def test_get_reranker_returns_singleton():
    a = get_reranker()
    b = get_reranker()
    assert a is b
    assert a.name == SETTINGS.reranker_model


def test_candidate_score_property_prefers_rerank():
    c = Candidate(text="t", source="s", chunk_id=0, dense_score=0.2)
    assert c.score == 0.2
    c.rerank_score = 0.9
    assert c.score == 0.9
