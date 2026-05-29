"""Local RAG retrieval tool.

Owns the similarity-threshold routing rule: if the average score across the
top-N results is below `SETTINGS.similarity_threshold`, return a fallback
message that prompts Claude to call `web_search` next.
"""
from __future__ import annotations

from civicai.config import SETTINGS
from civicai.rag.embeddings import get_embedder
from civicai.rag.vectorstore import get_collection


def search_docs(query: str, n_results: int | None = None) -> str:
    n_results = n_results or SETTINGS.default_n_results

    embedder = get_embedder()
    collection = get_collection()

    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    formatted: list[str] = []
    scores: list[float] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1 - dist, 3)
        scores.append(score)
        formatted.append(f"[Source: {meta['source']} | Score: {score}]\n{doc}")

    average_score = sum(scores) / len(scores)

    if average_score < SETTINGS.similarity_threshold:
        return (
            f"Aucun document pertinent trouvé (score moyen: {average_score}). "
            "Utilise web_search pour répondre à cette question."
        )

    return "\n\n---\n\n".join(formatted)
