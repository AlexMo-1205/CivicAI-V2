"""P3b — RAGAS evaluation runner (incremental, resumable, rate-limit-safe).

Runs the FULL civicai pipeline once per dataset item, captures a frozen
record, then scores faithfulness / answer_relevancy / context_precision /
context_recall with an EXPLICIT Claude judge + bge-m3 embedder. No metric
is ever invoked with the RAGAS default LLM/embedder — if ANTHROPIC_API_KEY
is missing the script fails loudly rather than silently falling back to OpenAI.

Crash-safe: both pipeline records and scoring results are append-flushed to
JSONL after each item, so any kill / rate-limit / timeout keeps everything
done so far. Re-running auto-resumes — items already in the run files are
skipped. Set --fresh to start from scratch.

NOT a pytest target. Loads ~2GB of weights and calls the Anthropic API many
times. Run it directly:

    uv run python evals/runner.py                  # full eval (resumes)
    uv run python evals/runner.py --limit 5        # smoke test
    uv run python evals/runner.py --fresh          # wipe run files first
    uv run python evals/runner.py --rps-sleep 2.5  # slower if rate-limited

Files:
    evals/runs/p3b_records.jsonl   pipeline outputs (resume marker)
    evals/runs/p3b_results.jsonl   per-item RAGAS scores (resume marker)
    evals/runs/p3b_report_<ts>.md  rendered headline + per-category breakdown
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 1. HARD-FAIL early so RAGAS never silently falls back to OpenAI.
# ---------------------------------------------------------------------------
def _require_anthropic_key() -> None:
    from civicai.config import SETTINGS  # noqa: F401  (loads .env)
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.stderr.write(
            "FATAL: ANTHROPIC_API_KEY not set. Refusing to silently fall back "
            "to the RAGAS default LLM (OpenAI). Aborting.\n"
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# 2. civicai pipeline + capture shim
# ---------------------------------------------------------------------------
from civicai.agent.graph import ask
from civicai.config import SETTINGS
from civicai.rag.embeddings import get_embeddings
from civicai.rag.reranker import get_reranker
from civicai.rag.retrieval import retrieve
from civicai.tools import dispatcher as disp_mod
from civicai.tools.search_docs import _fallback_message, _format


class _RetrievalCapture:
    def __init__(self):
        self.last_chunks: list = []


CAPTURE = _RetrievalCapture()


def _instrumented_search_docs(query: str, n_results: Optional[int] = None) -> str:
    candidates = retrieve(query, k=SETTINGS.retrieve_top_k)
    reranked = get_reranker().rerank(query, candidates, top_n=SETTINGS.rerank_top_n)
    CAPTURE.last_chunks = list(reranked)

    if not reranked:
        return _fallback_message(0.0)

    top_score = reranked[0].score
    if top_score < SETTINGS.rerank_routing_threshold:
        return _fallback_message(top_score)
    return _format(reranked)


# ---------------------------------------------------------------------------
# 3. langchain Embeddings wrapper around our existing provider
# ---------------------------------------------------------------------------
from langchain_core.embeddings import Embeddings


class CivicAIEmbeddings(Embeddings):
    """Reuses civicai's already-loaded bge-m3 instance for RAGAS scoring."""

    def __init__(self, provider):
        self._p = provider

    def embed_documents(self, texts):
        return self._p.embed_documents(list(texts))

    def embed_query(self, text):
        return self._p.embed_query(text)


# ---------------------------------------------------------------------------
# 4. Incremental JSONL helpers (append + flush + fsync = crash-safe)
# ---------------------------------------------------------------------------
RUNS_DIR = Path("evals/runs")
RECORDS_PATH = RUNS_DIR / "p3b_records.jsonl"
RESULTS_PATH = RUNS_DIR / "p3b_results.jsonl"


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
# 5. RAGAS wiring (explicit; no defaults)
# ---------------------------------------------------------------------------
METRIC_COLS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


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


# ---------------------------------------------------------------------------
# 6. Pipeline pass (one item at a time, append after each)
# ---------------------------------------------------------------------------
def run_pipeline_item(item: dict) -> dict:
    CAPTURE.last_chunks = []
    answer = ask(item["question"])
    chunks = CAPTURE.last_chunks
    return {
        "id": item["id"],
        "category": item["category"],
        "user_input": item["question"],
        "response": answer,
        "retrieved_contexts": [c.text for c in chunks]
        or ["[no local context retrieved — fallback path]"],
        "reference": item["ground_truth"],
    }


def pipeline_pass(items: list[dict]) -> list[dict]:
    already = _done_ids(RECORDS_PATH)
    if already:
        print(f"[resume] pipeline: skipping {len(already)} already-recorded items")

    new_records: list[dict] = []
    errs: list[tuple[str, str]] = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        if item["id"] in already:
            continue
        try:
            rec = run_pipeline_item(item)
            _append_jsonl(RECORDS_PATH, rec)
            new_records.append(rec)
            print(
                f"  [{i:>3}/{len(items)}] pipe {rec['id']:<10} {rec['category']:<12} "
                f"ok (chunks={len(rec['retrieved_contexts'])})"
            )
        except Exception as exc:  # noqa: BLE001
            errs.append((item["id"], str(exc)))
            print(f"  [{i:>3}/{len(items)}] pipe {item['id']:<10} ERROR: {exc}")
    print(
        f"Pipeline pass: +{len(new_records)} new in "
        f"{time.time() - t0:.1f}s ({len(errs)} errored)"
    )
    return _load_jsonl(RECORDS_PATH)


# ---------------------------------------------------------------------------
# 7. Scoring pass — ONE item at a time, with per-item retry and isolation
# ---------------------------------------------------------------------------
def _ragas_run_config():
    """Bounded concurrency + backoff so we don't blow Anthropic rate limits."""
    from ragas.run_config import RunConfig

    # max_workers=2: RAGAS scores 4 metrics in parallel per single item.
    # max_retries / max_wait: ragas will retry transient errors up to 1 min.
    return RunConfig(max_workers=2, max_retries=10, max_wait=60, timeout=180)


def score_record(record: dict, metrics, judge_wrap, emb_wrap, run_config) -> dict:
    from ragas import EvaluationDataset, evaluate

    row = {
        "user_input": record["user_input"],
        "response": record["response"],
        "retrieved_contexts": record["retrieved_contexts"],
        "reference": record["reference"],
    }
    ds = EvaluationDataset.from_list([row])
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
    return out


def scoring_pass(records: list[dict], judge_wrap, emb_wrap, sleep_s: float) -> list[dict]:
    metrics = _wire_metrics(judge_wrap, emb_wrap)
    run_config = _ragas_run_config()
    already = _done_ids(RESULTS_PATH)
    if already:
        print(f"[resume] scoring: skipping {len(already)} already-scored items")

    new_results: list[dict] = []
    errs: list[tuple[str, str]] = []
    pending = [r for r in records if r["id"] not in already]
    t0 = time.time()
    for i, rec in enumerate(pending, 1):
        try:
            scored = score_record(rec, metrics, judge_wrap, emb_wrap, run_config)
        except Exception as exc:  # noqa: BLE001
            scored = {
                "id": rec["id"],
                "category": rec["category"],
                "error": str(exc),
                **{c: float("nan") for c in METRIC_COLS},
            }
            errs.append((rec["id"], str(exc)))
        _append_jsonl(RESULTS_PATH, scored)
        new_results.append(scored)
        summary = " ".join(
            f"{c}={scored[c]:.3f}" if scored[c] == scored[c] else f"{c}=nan"
            for c in METRIC_COLS
        )
        flag = "ERR" if scored["error"] else "ok "
        print(
            f"  [{i:>3}/{len(pending)}] score {rec['id']:<10} {rec['category']:<12} "
            f"{flag} {summary}"
        )
        time.sleep(sleep_s)
    print(
        f"Scoring pass: +{len(new_results)} new in "
        f"{time.time() - t0:.1f}s ({len(errs)} errored)"
    )
    return _load_jsonl(RESULTS_PATH)


# ---------------------------------------------------------------------------
# 8. Report rendering — authoritative count is from RESULTS_PATH
# ---------------------------------------------------------------------------
def _mean(values) -> float:
    clean = [float(v) for v in values if v == v]
    return statistics.fmean(clean) if clean else float("nan")


def render_report(results: list[dict], *, judge_model: str) -> str:
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return "# P3b — no results to render\n"

    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    err_count = int(df["error"].notna().sum()) if "error" in df else 0
    total = len(df)

    lines: list[str] = []
    lines.append("# P3b — RAGAS evaluation report\n")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Items scored: {total} (authoritative, from {RESULTS_PATH.name})")
    lines.append(f"- Items errored: {err_count}")
    lines.append(f"- **Judge LLM**: `{judge_model}` (Anthropic, via LangchainLLMWrapper)")
    lines.append(
        f"- **Embedder**: `{SETTINGS.embed_model}` "
        f"(LangchainEmbeddingsWrapper around civicai.rag.embeddings)"
    )
    lines.append(
        f"- Reranker: `{SETTINGS.reranker_model}` "
        f"(top_k={SETTINGS.retrieve_top_k}, top_n={SETTINGS.rerank_top_n})"
    )
    lines.append(
        f"- Routing threshold (PLACEHOLDER): "
        f"`{SETTINGS.rerank_routing_threshold}` — P3c sweep replaces this.\n"
    )

    def _block(label: str, sub: "pd.DataFrame", note: str) -> None:
        lines.append(f"## Category `{label}` (n={len(sub)})\n")
        if note:
            lines.append(note + "\n")
        lines.append("| Metric | Mean | Median |")
        lines.append("|---|---|---|")
        for m in metric_cols:
            vals = [v for v in sub[m] if v == v]
            med = statistics.median(vals) if vals else float("nan")
            lines.append(f"| {m} | {_mean(sub[m]):.3f} | {med:.3f} |")
        lines.append("")

    local_df = df[df["category"] == "local"]
    _block(
        "local",
        local_df,
        "**Headline — retrieval + generation quality on items the KB should answer. "
        "This is the number to optimize.**",
    )

    fb_df = df[df["category"] == "fallback"]
    _block(
        "fallback",
        fb_df,
        "**EXPECTED-LOW: the KB intentionally lacks this; routing is measured in P3c, "
        "not here.** Low faithfulness/precision/recall here reflect a successful fallback "
        "path where the answer comes from the web search, not from local retrieval.",
    )

    adv_df = df[df["category"] == "adversarial"]
    _block(
        "adversarial",
        adv_df,
        "**Reported separately — NOT folded into headline numbers.** A correct adversarial "
        "answer contradicts the user's premise, which scores atypically on answer_relevancy. "
        "Faithfulness against the corrective ground truth is the meaningful number here.",
    )

    # Per-item
    lines.append(f"## Per-item scores ({len(df)} items)\n")
    header = "| id | category | " + " | ".join(metric_cols) + " | error |"
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(metric_cols)))
    for _, row in df.iterrows():
        cells = [str(row["id"]), str(row["category"])]
        for m in metric_cols:
            v = row[m]
            cells.append(f"{v:.3f}" if v == v else "—")
        err = row.get("error") or ""
        cells.append(str(err)[:80])
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N items (smoke test).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe existing run files before starting (no resume).",
    )
    parser.add_argument(
        "--rps-sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between scored items (rate-limit safety).",
    )
    parser.add_argument("--judge-model", default=SETTINGS.model)
    args = parser.parse_args()

    _require_anthropic_key()

    if args.fresh:
        for p in (RECORDS_PATH, RESULTS_PATH):
            if p.exists():
                p.unlink()
                print(f"[fresh] removed {p}")

    from langchain_anthropic import ChatAnthropic
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    judge = ChatAnthropic(
        model=args.judge_model,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        # 4096 to avoid RAGAS' LLMDidNotFinishException on long statement lists.
        max_tokens=4096,
        temperature=0.0,
    )
    judge_wrap = LangchainLLMWrapper(judge)
    emb_wrap = LangchainEmbeddingsWrapper(CivicAIEmbeddings(get_embeddings()))

    print()
    print("=" * 70)
    print(f">>> JUDGE LLM  = {args.judge_model}  (ChatAnthropic + LangchainLLMWrapper)")
    print(f">>> EMBEDDER   = {SETTINGS.embed_model}  (LangchainEmbeddingsWrapper)")
    print(">>> RAGAS DEFAULTS DISABLED — every metric is explicitly wired.")
    print(f">>> RUN FILES  = {RECORDS_PATH}  +  {RESULTS_PATH}  (append+fsync per item)")
    print(f">>> rps-sleep  = {args.rps_sleep}s between items")
    print("=" * 70)

    disp_mod.HANDLERS["search_docs"] = _instrumented_search_docs

    # Load dataset
    items = [
        json.loads(line)
        for line in Path(args.dataset).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        items = items[: args.limit]
    print(f"\nLoaded {len(items)} item(s) from {args.dataset}\n")

    # Phase 1 — pipeline (resumable)
    print("=== Phase 1: pipeline pass ===")
    records = pipeline_pass(items)
    # Honor --limit at the scoring side too
    keep_ids = {it["id"] for it in items}
    records = [r for r in records if r["id"] in keep_ids]

    # Phase 2 — scoring (resumable)
    print("\n=== Phase 2: scoring pass (one item per evaluate() call) ===")
    results = scoring_pass(records, judge_wrap, emb_wrap, sleep_s=args.rps_sleep)
    results = [r for r in results if r["id"] in keep_ids]

    # Render report
    stamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = RUNS_DIR / f"p3b_report_{stamp}.md"
    md_path.write_text(
        render_report(results, judge_model=args.judge_model),
        encoding="utf-8",
    )

    scored = sum(1 for r in results if r.get("error") is None)
    errored = sum(1 for r in results if r.get("error") is not None)
    print()
    print(f"=== Done ===")
    print(f"  total in results file : {len(results)}")
    print(f"  scored ok             : {scored}")
    print(f"  errored               : {errored}")
    print(f"  report                : {md_path}")


if __name__ == "__main__":
    main()
