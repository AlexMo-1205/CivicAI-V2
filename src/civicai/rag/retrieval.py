"""Stage 1 of the retrieval pipeline: dense recall from ChromaDB.

Returns a list of `Candidate` objects that the reranker then re-scores.
Splitting retrieval and reranking into separate stages keeps the pipeline
readable (retrieve -> rerank -> decide) and trivially testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from civicai.config import SETTINGS
from civicai.rag.embeddings import get_embeddings
from civicai.rag.vectorstore import get_collection


@dataclass
class Candidate:
    text: str
    source: str
    chunk_id: int
    dense_score: float
    rerank_score: Optional[float] = None

    @property
    def score(self) -> float:
        """The score used for routing decisions: rerank if present, else dense."""
        return self.rerank_score if self.rerank_score is not None else self.dense_score


def retrieve(query: str, k: Optional[int] = None) -> list[Candidate]:
    """Dense top-k recall pass against ChromaDB (cosine space)."""
    k = k or SETTINGS.retrieve_top_k

    embeddings = get_embeddings()
    collection = get_collection()

    query_vector = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    candidates: list[Candidate] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append(
            Candidate(
                text=doc,
                source=meta["source"],
                chunk_id=int(meta["chunk_id"]),
                dense_score=round(1.0 - float(dist), 4),
            )
        )
    return candidates
