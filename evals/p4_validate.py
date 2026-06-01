"""P4 prompt-tightening — targeted validation.

Regenerates AND re-scores ONLY a small, fixed subset of items, into a SEPARATE
results file so the P3b headline (evals/runs/p3b_results.jsonl) is untouched.

Cost: ~10 items × (1 ask() + 1 RAGAS evaluate). Small. Run directly:

    uv run python evals/p4_validate.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


# Hard-fail before any provider import.
load_dotenv(override=True)
if not os.getenv("ANTHROPIC_API_KEY"):
    sys.stderr.write("FATAL: ANTHROPIC_API_KEY missing. Aborting.\n")
    sys.exit(2)


VALIDATION_IDS = [
    # 2 known hallucination cases
    "multi-11", "wp-02",
    # 6 high-faithfulness locals (regression check — must NOT drop)
    "ltr-01", "ylw-01", "tax-03", "reen-02", "dl-01", "mar-02",
    # 2 adversarials (must still contradict the false premise)
    "adv-02", "adv-03",
]

DATASET = Path("evals/dataset.jsonl")
OLD_RESULTS = Path("evals/runs/p3b_results.jsonl")
OUT_RESULTS = Path("evals/runs/p4_validation_results.jsonl")


# ---------------------------------------------------------------------------
# civicai pipeline + capture (mirrors evals/runner.py)
# ---------------------------------------------------------------------------
from civicai.agent.graph import ask
from civicai.config import SETTINGS
from civicai.rag.reranker import get_reranker
from civicai.rag.retrieval import retrieve
from civicai.tools import dispatcher as disp_mod
from civicai.tools.search_docs import _fallback_message, _format


class _Capture:
    def __init__(self):
        self.chunks: list = []


CAP = _Capture()


def _instrumented_search_docs(query: str, n_results=None) -> str:
    cands = retrieve(query, k=SETTINGS.retrieve_top_k)
    reranked = get_reranker().rerank(query, cands, top_n=SETTINGS.rerank_top_n)
    CAP.chunks = list(reranked)
    if not reranked:
        return _fallback_message(0.0)
    top = reranked[0].score
    if top < SETTINGS.rerank_routing_threshold:
        return _fallback_message(top)
    return _format(reranked)


disp_mod.HANDLERS["search_docs"] = _instrumented_search_docs


# ---------------------------------------------------------------------------
# Lazy bge-m3 wrapper (reused from scorer.py via duplication so we don't import
# scorer's heavy module path)
# ---------------------------------------------------------------------------
from langchain_core.embeddings import Embeddings


class LazyBGEEmbeddings(Embeddings):
    MODEL = "BAAI/bge-m3"

    def __init__(self):
        self._m = None

    def _load(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer

            self._m = SentenceTransformer(self.MODEL)
        return self._m

    def embed_documents(self, texts):
        return self._load().encode(
            list(texts), show_progress_bar=False, normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return self._load().encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    items_all = {it["id"]: it for it in _load_jsonl(DATASET)}
    old_all = {r["id"]: r for r in _load_jsonl(OLD_RESULTS)}

    missing = [i for i in VALIDATION_IDS if i not in items_all]
    if missing:
        sys.stderr.write(f"FATAL: ids missing from dataset: {missing}\n")
        sys.exit(3)

    print("=" * 70)
    print(f">>> P4 VALIDATION — {len(VALIDATION_IDS)} items")
    print(f">>> output: {OUT_RESULTS} (separate; p3b headline untouched)")
    print("=" * 70)

    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    judge = ChatAnthropic(
        model=SETTINGS.model,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=8192,
        temperature=0.0,
    )
    jw = LangchainLLMWrapper(judge)
    ew = LangchainEmbeddingsWrapper(LazyBGEEmbeddings())
    for m in (faithfulness, answer_relevancy, context_precision, context_recall):
        m.llm = jw
        if hasattr(m, "embeddings"):
            m.embeddings = ew
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    rc = RunConfig(max_workers=2, max_retries=10, max_wait=60, timeout=240)

    # Wipe + start fresh output
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if OUT_RESULTS.exists():
        OUT_RESULTS.unlink()

    rows = []
    for i, _id in enumerate(VALIDATION_IDS, 1):
        item = items_all[_id]
        print(f"\n[{i}/{len(VALIDATION_IDS)}] {_id}  category={item['category']}")
        print(f"  Q: {item['question'][:80]}")

        # Phase 1: regenerate with tightened prompt
        CAP.chunks = []
        t0 = time.time()
        try:
            answer = ask(item["question"])
        except Exception as exc:  # noqa: BLE001
            print(f"  pipeline ERROR: {exc}")
            continue
        chunks = [c.text for c in CAP.chunks] or ["[no local context]"]
        print(f"  generated in {time.time()-t0:.1f}s  ({len(answer)} chars, {len(chunks)} chunks)")

        # Phase 2: score
        ds = EvaluationDataset.from_list([{
            "user_input": item["question"],
            "response": answer,
            "retrieved_contexts": chunks,
            "reference": item["ground_truth"],
        }])
        t1 = time.time()
        try:
            res = evaluate(ds, metrics=metrics, llm=jw, embeddings=ew,
                           run_config=rc, show_progress=False)
            df = res.to_pandas()
            scores = {c: float(df[c].iloc[0]) if c in df.columns and df[c].iloc[0] == df[c].iloc[0] else float("nan")
                      for c in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
        except Exception as exc:  # noqa: BLE001
            print(f"  scoring ERROR: {exc}")
            scores = {c: float("nan") for c in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
        print(f"  scored in {time.time()-t1:.1f}s")

        # Build row + before/after comparison
        old = old_all.get(_id, {})
        row = {
            "id": _id,
            "category": item["category"],
            "old_faithfulness": float(old.get("faithfulness", float("nan"))) if old else float("nan"),
            "new_faithfulness": scores["faithfulness"],
            "old_answer_relevancy": float(old.get("answer_relevancy", float("nan"))) if old else float("nan"),
            "new_answer_relevancy": scores["answer_relevancy"],
            "old_context_precision": float(old.get("context_precision", float("nan"))) if old else float("nan"),
            "new_context_precision": scores["context_precision"],
            "old_context_recall": float(old.get("context_recall", float("nan"))) if old else float("nan"),
            "new_context_recall": scores["context_recall"],
            "response": answer,
        }
        with OUT_RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        rows.append(row)

        def _fmt(v):
            return f"{v:.3f}" if v == v else "  nan"
        print(f"  faith     {_fmt(row['old_faithfulness'])} -> {_fmt(row['new_faithfulness'])}")
        print(f"  rel       {_fmt(row['old_answer_relevancy'])} -> {_fmt(row['new_answer_relevancy'])}")

        del ds, res
        gc.collect()
        time.sleep(1.5)

    # Summary
    print()
    print("=" * 70)
    print("Before / After (faithfulness | answer_relevancy)")
    print("=" * 70)
    print(f"{'id':<10} {'cat':<12} {'old_f':>7} {'new_f':>7}   {'Δf':>7}   {'old_r':>7} {'new_r':>7}   {'Δr':>7}")
    for r in rows:
        df_ = (r["new_faithfulness"] - r["old_faithfulness"]) if r["old_faithfulness"] == r["old_faithfulness"] else float("nan")
        dr_ = (r["new_answer_relevancy"] - r["old_answer_relevancy"]) if r["old_answer_relevancy"] == r["old_answer_relevancy"] else float("nan")
        def s(v): return f"{v:+.3f}" if v == v else "    nan"
        def f(v): return f"{v:.3f}" if v == v else "  nan"
        print(f"{r['id']:<10} {r['category']:<12} {f(r['old_faithfulness']):>7} {f(r['new_faithfulness']):>7}   {s(df_):>7}   {f(r['old_answer_relevancy']):>7} {f(r['new_answer_relevancy']):>7}   {s(dr_):>7}")
    print(f"\nResults file: {OUT_RESULTS}")


if __name__ == "__main__":
    main()
