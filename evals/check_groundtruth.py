#!/usr/bin/env python3
"""
Throwaway diagnostic — run BEFORE the full RAGAS eval.

For each adversarial item and the two flagged fallback items (fall-04, fall-06),
this prints the ground truth next to the top retrieved chunks, so you can eyeball
whether the corrective / expected fact is actually present in the retrieved context.

Why it matters:
- Adversarial: the agent can only score "faithful" if the corrected fact is in the
  retrieved chunks. If it's not in the KB, the item is secretly a fallback case.
- fall-04 / fall-06: confirm the KB genuinely lacks the specific list, so routing-to-web
  is the right behavior and the ground truth is defensible.

This talks to ChromaDB directly to avoid depending on refactored module names.
Match the CONFIG block below to your config.py, then:  uv run python check_groundtruth.py
Delete it once you've reviewed — it's not part of the test suite.
"""

import json
import re
import sys
from pathlib import Path

# ---- CONFIG: match these to civicai/config.py ------------------------------
DATASET_PATH = "evals/dataset.jsonl"
CHROMA_PATH = "./chroma_db"          # persist dir
COLLECTION_NAME = "civicai_bge_m3_1024"          # your collection name
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RETRIEVE_TOP_K = 40                  # how many to pull from Chroma
SHOW_TOP_N = 8                       # how many to print per question
RUN_RERANKER = True                  # set False for a faster, retrieval-only pass
IDS_TO_CHECK = ["fall-04", "fall-06"]  # plus everything with category == "adversarial"
# ----------------------------------------------------------------------------


def load_items():
    path = Path(DATASET_PATH)
    if not path.exists():
        sys.exit(f"Dataset not found at {DATASET_PATH} — run this from the repo root.")
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [
        it for it in items
        if it.get("category") == "adversarial" or it.get("id") in IDS_TO_CHECK
    ]
    return selected


def key_facts(ground_truth: str):
    """Pull number-like tokens from the ground truth (fees, amounts, durations).
    Crude hint, not a verdict — most useful for the fee-correction adversarials."""
    # matches: 2 000, 2,000, 750, 50 000 THB, 90 jours, 6 mois, etc.
    return set(re.findall(r"\d[\d\s.,]*\d|\d", ground_truth))


def contains_fact(chunk: str, facts) -> list:
    norm = re.sub(r"\s+", " ", chunk)
    hits = []
    for f in facts:
        f_norm = re.sub(r"\s+", " ", f).strip()
        if f_norm and f_norm in norm:
            hits.append(f_norm)
    return hits


def main():
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        sys.exit(f"Missing dep ({e}). Run inside the project env: uv run python check_groundtruth.py")

    items = load_items()
    if not items:
        sys.exit("No matching items found — check category flags / IDS_TO_CHECK.")

    print(f"Loaded {len(items)} item(s) to verify "
          f"({sum(1 for i in items if i.get('category') == 'adversarial')} adversarial, "
          f"{sum(1 for i in items if i.get('id') in IDS_TO_CHECK)} flagged fallback)\n")

    print(f"Loading embedder {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    reranker = None
    if RUN_RERANKER:
        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker {RERANK_MODEL} ...")
            reranker = CrossEncoder(RERANK_MODEL)
        except Exception as e:
            print(f"  (reranker unavailable, retrieval-only: {e})")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    def sigmoid(x):
        import math
        return 1 / (1 + math.exp(-x))

    for it in items:
        qid = it.get("id", "?")
        cat = it.get("category", "?")
        question = it.get("question", "")
        gt = it.get("ground_truth", "")
        facts = key_facts(gt)

        print("=" * 88)
        print(f"[{qid}] category={cat}")
        print(f"Q : {question}")
        print(f"GT: {gt}")
        if facts:
            print(f"key facts to find in context: {sorted(facts)}")
        print("-" * 88)

        q_emb = embedder.encode(question, normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=[q_emb], n_results=RETRIEVE_TOP_K)
        docs = res["documents"][0]
        dists = res.get("distances", [[None] * len(docs)])[0]

        ranked = list(zip(docs, dists))
        if reranker is not None:
            scores = reranker.predict([(question, d) for d in docs])
            ranked = sorted(
                [(d, dist, sigmoid(float(s))) for d, dist, s in zip(docs, dists, scores)],
                key=lambda x: x[2], reverse=True,
            )
        else:
            ranked = [(d, dist, None) for d, dist in ranked]

        any_fact_in_top = False
        for rank, row in enumerate(ranked[:SHOW_TOP_N], 1):
            doc, dist, rr = row
            sim = (1 - dist) if isinstance(dist, (int, float)) else None
            hits = contains_fact(doc, facts) if facts else []
            if hits:
                any_fact_in_top = True
            flags = []
            if sim is not None:
                flags.append(f"sim~{sim:.3f}")
            if rr is not None:
                flags.append(f"rerank={rr:.3f}")
            if hits:
                flags.append(f"FACT✓ {hits}")
            snippet = re.sub(r"\s+", " ", doc).strip()[:240]
            print(f"  #{rank} [{' | '.join(flags)}]")
            print(f"      {snippet}…")

        print("-" * 88)
        if facts:
            verdict = "PRESENT in top results" if any_fact_in_top else "NOT FOUND in top results"
            print(f"  → key fact(s) {verdict}.")
            if cat == "adversarial" and not any_fact_in_top:
                print("    ⚠ corrective fact missing — this scores as unfaithful even when the")
                print("      answer is right. Either the fact isn't in the KB (reclassify as")
                print("      fallback) or retrieval needs tuning before the eval is trustworthy.")
            if qid in IDS_TO_CHECK and not any_fact_in_top:
                print("    ⚠ KB lacks the specific fact — routing-to-web is correct IF you can")
                print("      write a confident web-derived ground truth; otherwise cut the item.")
        else:
            print("  → no numeric key fact extracted; eyeball the chunks above against the GT.")
        print()

    print("=" * 88)
    print("Review done. The number-matching is a hint, not proof — read the snippets for")
    print("non-numeric facts (rules, eligibility). Fix/reclassify items, then proceed to P3b.")


if __name__ == "__main__":
    main()
