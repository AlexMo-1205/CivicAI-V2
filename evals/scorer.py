"""P3b — Standalone RAGAS scorer (decoupled from the pipeline).

Reads frozen records from `evals/runs/p3b_records.jsonl`, scores each item
with RAGAS (Claude judge + bge-m3 for embedding metrics), and appends the
result to `evals/runs/p3b_results.jsonl` after EACH item — fully resumable.

Why a separate file from runner.py:
The pipeline path imports civicai.rag.embeddings, civicai.rag.reranker,
civicai.rag.retrieval — which load bge-m3 (~2 GB) and bge-reranker-v2-m3
(~1 GB) into memory eagerly. During scoring, those models are not needed
(answer_relevancy needs ONE embedder; nothing needs the reranker), so we
import none of them here. bge-m3 is loaded LAZILY on the first metric call.

NOT a pytest target. Run directly:

    uv run python evals/scorer.py                     # score all unscored records
    uv run python evals/scorer.py --limit 5           # next 5 unscored
    uv run python evals/scorer.py --rps-sleep 2.0     # slower if rate-limited
    uv run python evals/scorer.py --batch 1           # 1 record per evaluate() call

Exit codes:
    0  ok (all scoring attempts wrote a row — errors are isolated per item)
    2  missing ANTHROPIC_API_KEY
    3  no records on disk
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Hard-fail if Anthropic key missing — never silently fall back to OpenAI.
# ---------------------------------------------------------------------------
def _require_anthropic_key() -> None:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.stderr.write(
            "FATAL: ANTHROPIC_API_KEY not set. Refusing to silently fall back "
            "to the RAGAS default LLM (OpenAI). Aborting.\n"
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# 2. Run files
# ---------------------------------------------------------------------------
RUNS_DIR = Path("evals/runs")
RECORDS_PATH = RUNS_DIR / "p3b_records.jsonl"
RESULTS_PATH = RUNS_DIR / "p3b_results.jsonl"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5"

METRIC_COLS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _done_ids(path: Path) -> set[str]:
    return {row["id"] for row in _load_jsonl(path)}


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# 3. Lazy bge-m3 embedder for RAGAS (no civicai imports)
# ---------------------------------------------------------------------------
from langchain_core.embeddings import Embeddings


class LazyBGEEmbeddings(Embeddings):
    """bge-m3 wrapped as a langchain Embeddings. Loads weights on first call,
    NOT at import time. Keeps the standalone scorer light when only metadata
    operations happen."""

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            sys.stderr.write(f"[scorer] lazy-loading embedder {self.MODEL_NAME}...\n")
            sys.stderr.flush()
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def embed_documents(self, texts):
        return self._load().encode(
            list(texts), show_progress_bar=False, normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return self._load().encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# 4. RAGAS wiring (explicit — no defaults)
# ---------------------------------------------------------------------------
def _wire_metrics(judge_wrap, emb_wrap):
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
        metric.llm = judge_wrap
        if hasattr(metric, "embeddings"):
            metric.embeddings = emb_wrap
    return [faithfulness, answer_relevancy, context_precision, context_recall]


def _run_config():
    from ragas.run_config import RunConfig

    # max_workers=2 keeps Anthropic concurrency well under tier-1 50 req/min
    # max_retries+max_wait absorb transient 429s
    return RunConfig(max_workers=2, max_retries=10, max_wait=60, timeout=180)


# ---------------------------------------------------------------------------
# 5. Score one record (full isolation; one evaluate() per call)
# ---------------------------------------------------------------------------
def score_record(record: dict, metrics, judge_wrap, emb_wrap, run_config) -> dict:
    from ragas import EvaluationDataset, evaluate

    ds = EvaluationDataset.from_list(
        [
            {
                "user_input": record["user_input"],
                "response": record["response"],
                "retrieved_contexts": record["retrieved_contexts"],
                "reference": record["reference"],
            }
        ]
    )
    res = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=judge_wrap,
        embeddings=emb_wrap,
        run_config=run_config,
        show_progress=False,
    )
    df = res.to_pandas()
    out = {"id": record["id"], "category": record["category"], "error": None}
    for col in METRIC_COLS:
        if col in df.columns:
            val = df[col].iloc[0]
            out[col] = float(val) if val == val else float("nan")
        else:
            out[col] = float("nan")
    # Drop large intermediates before next loop iteration.
    del ds, res, df
    return out


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rps-sleep", type=float, default=2.0)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    _require_anthropic_key()

    records = _load_jsonl(RECORDS_PATH)
    if not records:
        sys.stderr.write(f"FATAL: no records at {RECORDS_PATH}\n")
        sys.exit(3)

    already = _done_ids(RESULTS_PATH)
    pending = [r for r in records if r["id"] not in already]
    if args.limit:
        pending = pending[: args.limit]

    print()
    print("=" * 70, flush=True)
    print(f">>> SCORER (standalone, no civicai.rag imports)", flush=True)
    print(f">>> JUDGE LLM = {args.judge_model}  (ChatAnthropic + LangchainLLMWrapper)", flush=True)
    print(f">>> EMBEDDER  = {LazyBGEEmbeddings.MODEL_NAME}  (lazy-loaded on first call)", flush=True)
    print(f">>> records   = {len(records)} on disk", flush=True)
    print(f">>> already   = {len(already)} scored", flush=True)
    print(f">>> pending   = {len(pending)} this run", flush=True)
    print(f">>> rps-sleep = {args.rps_sleep}s between items", flush=True)
    print("=" * 70, flush=True)

    if not pending:
        print("\nNothing to score.", flush=True)
        return

    # Build judge + lazy embedder
    from langchain_anthropic import ChatAnthropic
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    judge = ChatAnthropic(
        model=args.judge_model,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4096,
        temperature=0.0,
    )
    judge_wrap = LangchainLLMWrapper(judge)
    emb_wrap = LangchainEmbeddingsWrapper(LazyBGEEmbeddings())

    metrics = _wire_metrics(judge_wrap, emb_wrap)
    run_config = _run_config()

    errors: list[tuple[str, str]] = []
    t0 = time.time()
    for i, rec in enumerate(pending, 1):
        item_t0 = time.time()
        try:
            scored = score_record(rec, metrics, judge_wrap, emb_wrap, run_config)
        except Exception as exc:  # noqa: BLE001
            scored = {
                "id": rec["id"],
                "category": rec["category"],
                "error": f"{type(exc).__name__}: {exc}",
                **{c: float("nan") for c in METRIC_COLS},
            }
            errors.append((rec["id"], scored["error"]))
        _append_jsonl(RESULTS_PATH, scored)
        summary = " ".join(
            f"{c}={scored[c]:.3f}" if scored[c] == scored[c] else f"{c}=nan"
            for c in METRIC_COLS
        )
        flag = "ERR" if scored["error"] else "ok "
        dt = time.time() - item_t0
        print(
            f"  [{i:>3}/{len(pending)}] {rec['id']:<10} {rec['category']:<12} "
            f"{flag} ({dt:.1f}s)  {summary}",
            flush=True,
        )
        # Release transient memory between items
        gc.collect()
        time.sleep(args.rps_sleep)

    total_dt = time.time() - t0
    print()
    print("=" * 70, flush=True)
    print(f"Scored {len(pending)} item(s) in {total_dt:.1f}s  ({len(errors)} errored)", flush=True)
    if errors:
        print("Errors:", flush=True)
        for _id, msg in errors:
            print(f"  {_id}: {msg}", flush=True)
    print(f"Results file: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
