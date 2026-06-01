"""Stage 2 of the retrieval pipeline: cross-encoder reranking.

bge-reranker-v2-m3 is a multilingual cross-encoder. It outputs unbounded
logits, which we squash through sigmoid so scores land in (0, 1) and the
routing threshold becomes interpretable across queries.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol, runtime_checkable

from civicai.config import SETTINGS
from civicai.rag.retrieval import Candidate


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: list[Candidate], top_n: int
    ) -> list[Candidate]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class BGECrossEncoderReranker:
    """Cross-encoder backend (default: BAAI/bge-reranker-v2-m3)."""

    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None  # lazy load

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name)
        return self._model

    def rerank(
        self, query: str, candidates: list[Candidate], top_n: int
    ) -> list[Candidate]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        raw_scores = self._load().predict(pairs)

        for c, raw in zip(candidates, raw_scores):
            c.rerank_score = _sigmoid(float(raw))

        candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return candidates[:top_n]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Return the configured reranker as a process-wide singleton."""
    return BGECrossEncoderReranker(SETTINGS.reranker_model)
