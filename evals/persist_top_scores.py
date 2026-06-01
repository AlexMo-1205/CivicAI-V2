"""Persist top reranker scores per dataset item (one-time, no LLM calls).

Writes evals/runs/p3b_top_scores.jsonl with rows: {id, category, top_score}.
After this, the P3c sweep can run purely on cached data with no model loads.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    from civicai.config import SETTINGS
    from civicai.rag.retrieval import retrieve
    from civicai.rag.reranker import get_reranker

    dataset = Path("evals/dataset.jsonl")
    out = Path("evals/runs/p3b_top_scores.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in dataset.read_text().splitlines() if l.strip()]
    rr = get_reranker()
    rows = []
    for i, item in enumerate(items, 1):
        cands = retrieve(item["question"], k=SETTINGS.retrieve_top_k)
        ranked = rr.rerank(item["question"], cands, top_n=1)
        top = float(ranked[0].rerank_score) if ranked else float("nan")
        rows.append({"id": item["id"], "category": item["category"], "top_score": top})
        print(f"  [{i:>3}/{len(items)}] {item['id']:<12} {item['category']:<12} top={top:.3f}",
              flush=True)

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
