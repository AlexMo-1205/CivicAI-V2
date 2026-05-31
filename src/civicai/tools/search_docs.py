"""Local RAG retrieval tool.

Pipeline: dense `retrieve` (top_k) -> cross-encoder `rerank` (top_n) -> route.

Routing rule (the one the eval harness tunes): if the TOP reranked
sigmoid-normalized score is below `SETTINGS.rerank_routing_threshold`,
return a fallback message so Claude calls `web_search` next.
"""
from __future__ import annotations

from civicai.config import SETTINGS
from civicai.rag.reranker import get_reranker
from civicai.rag.retrieval import Candidate, retrieve


def _format(candidates: list[Candidate]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c.source} | Score: {round(c.score, 3)}]\n{c.text}"
        for c in candidates
    )


def _fallback_message(top_score: float) -> str:
    return (
        f"Aucun document pertinent trouvé (score top: {round(top_score, 3)}). "
        "Utilise web_search pour répondre à cette question."
    )


def search_docs(query: str, n_results: int | None = None) -> str:
    """Tool entry point. `n_results` is accepted for back-compat but the
    retrieve-then-rerank pipeline is driven by config (`retrieve_top_k` and
    `rerank_top_n`)."""
    candidates = retrieve(query, k=SETTINGS.retrieve_top_k)

    if not candidates:
        return _fallback_message(0.0)

    reranked = get_reranker().rerank(
        query, candidates, top_n=SETTINGS.rerank_top_n
    )

    top_score = reranked[0].score  # already a sigmoid-normalized rerank score
    if top_score < SETTINGS.rerank_routing_threshold:
        return _fallback_message(top_score)

    return _format(reranked)
