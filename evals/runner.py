"""P3b — RAGAS evaluation runner.

Runs the FULL civicai pipeline once per dataset item, captures a frozen
record (question, generated_answer, retrieved_contexts, ground_truth,
category), and scores faithfulness / answer_relevancy / context_precision /
context_recall with an EXPLICIT Claude judge + bge-m3 embedder. No metric is
ever invoked with the RAGAS default LLM/embedder — if ANTHROPIC_API_KEY is
missing the script fails loudly rather than silently falling back to OpenAI.

This is NOT a pytest target. It loads ~2 GB of model weights and calls the
Anthropic API many times. Run it directly:

    uv run python evals/runner.py [--limit N] [--judge-model claude-...]

Output: evals/runs/p3b_report_<ts>.md  +  p3b_report_<ts>.csv (raw scores)
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
    # Re-trigger civicai's load_dotenv(override=True) so .env populates first
    from civicai.config import SETTINGS  # noqa: F401  (side-effect)
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
    """Holds the chunks surfaced by the most recent search_docs call."""

    def __init__(self):
        self.last_chunks: list = []


CAPTURE = _RetrievalCapture()


def _instrumented_search_docs(query: str, n_results: Optional[int] = None) -> str:
    """Same logic as civicai.tools.search_docs.search_docs, but records the
    reranked chunks so the runner can use them as RAGAS retrieved_contexts."""
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
    """Adapter that lets RAGAS reuse the same loaded bge-m3 instance
    civicai uses, instead of loading the model twice."""

    def __init__(self, provider):
        self._p = provider

    def embed_documents(self, texts):
        return self._p.embed_documents(list(texts))

    def embed_query(self, text):
        return self._p.embed_query(text)


# ---------------------------------------------------------------------------
# 4. RAGAS wiring (explicit; no defaults)
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


# ---------------------------------------------------------------------------
# 5. Dataset I/O
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_pipeline(item: dict) -> dict:
    """Run civicai end-to-end on one item; freeze the record."""
    CAPTURE.last_chunks = []
    try:
        answer = ask(item["question"])
    except Exception as exc:  # noqa: BLE001
        # Re-raise so the runner can log per-item failures
        raise RuntimeError(f"ask() failed for {item['id']}: {exc}") from exc

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


# ---------------------------------------------------------------------------
# 6. Markdown report rendering
# ---------------------------------------------------------------------------
def _mean(values) -> float:
    clean = [float(v) for v in values if v == v]  # NaN filter
    return statistics.fmean(clean) if clean else float("nan")


METRIC_COLS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def render_report(df, *, items_count, errors, judge_model) -> str:
    # Only the four numeric score columns — RAGAS' DataFrame also carries the
    # text columns (user_input / response / etc.) which we don't average.
    metric_cols = [c for c in df.columns if c in METRIC_COLS]

    lines: list[str] = []
    lines.append("# P3b — RAGAS evaluation report\n")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Items scored: {items_count}")
    lines.append(f"- Errors: {len(errors)}")
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

    # Headline (local)
    local_df = df[df["category"] == "local"]
    lines.append(
        f"## Headline — category `local` (n={len(local_df)})\n"
    )
    lines.append(
        "Retrieval + generation quality on items the KB should answer. "
        "**This is the number to optimize.**\n"
    )
    lines.append("| Metric | Mean | Median |")
    lines.append("|---|---|---|")
    for m in metric_cols:
        vals = [v for v in local_df[m] if v == v]
        med = statistics.median(vals) if vals else float("nan")
        lines.append(f"| {m} | {_mean(local_df[m]):.3f} | {med:.3f} |")

    # Fallback (expected low)
    fb_df = df[df["category"] == "fallback"]
    lines.append(f"\n## Category `fallback` (n={len(fb_df)})\n")
    lines.append(
        "**EXPECTED-LOW: the KB intentionally lacks this; routing is measured "
        "in P3c, not here.** Low faithfulness/precision/recall here are not a "
        "bug — they reflect a successful fallback path where the answer comes "
        "from the web search, not from local retrieval.\n"
    )
    lines.append("| Metric | Mean |")
    lines.append("|---|---|")
    for m in metric_cols:
        lines.append(f"| {m} | {_mean(fb_df[m]):.3f} |")

    # Adversarial (separate)
    adv_df = df[df["category"] == "adversarial"]
    lines.append(f"\n## Category `adversarial` (n={len(adv_df)})\n")
    lines.append(
        "**Reported separately — NOT folded into headline numbers.** A correct "
        "adversarial answer contradicts the user's premise, which scores "
        "atypically on answer_relevancy. Faithfulness against the corrective "
        "ground truth is the meaningful number here.\n"
    )
    lines.append("| Metric | Mean |")
    lines.append("|---|---|")
    for m in metric_cols:
        lines.append(f"| {m} | {_mean(adv_df[m]):.3f} |")

    # Per-item
    lines.append(f"\n## Per-item scores ({len(df)} items)\n")
    header = "| id | category | " + " | ".join(metric_cols) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(metric_cols)))
    for _, row in df.iterrows():
        cells = [str(row["id"]), str(row["category"])]
        for m in metric_cols:
            v = row[m]
            cells.append(f"{v:.3f}" if v == v else "—")
        lines.append("| " + " | ".join(cells) + " |")

    if errors:
        lines.append("\n## Errors\n")
        for _id, msg in errors:
            lines.append(f"- `{_id}`: {msg}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument("--out", default=None, help="Override markdown path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N items (smoke test).",
    )
    parser.add_argument("--judge-model", default=SETTINGS.model)
    args = parser.parse_args()

    _require_anthropic_key()

    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
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
    print("=" * 70)

    metrics = _wire_metrics(judge_wrap, emb_wrap)

    # Capture chunks during search_docs tool calls
    disp_mod.HANDLERS["search_docs"] = _instrumented_search_docs

    # Load + run pipeline
    items = load_dataset(Path(args.dataset))
    if args.limit:
        items = items[: args.limit]
    print(f"\nLoaded {len(items)} item(s) from {args.dataset}\n")
    print("=== Running pipeline ===")
    records: list[dict] = []
    errors: list[tuple[str, str]] = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        try:
            rec = run_pipeline(item)
            records.append(rec)
            print(f"  [{i:>3}/{len(items)}] {rec['id']:<10} {rec['category']:<12} ok "
                  f"(chunks={len(rec['retrieved_contexts'])})")
        except Exception as exc:  # noqa: BLE001
            errors.append((item["id"], str(exc)))
            print(f"  [{i:>3}/{len(items)}] {item['id']:<10} ERROR: {exc}")
    print(f"Pipeline pass took {time.time() - t0:.1f}s ({len(errors)} errored)")

    if not records:
        print("No records to score; aborting.", file=sys.stderr)
        sys.exit(3)

    # Build frozen RAGAS dataset
    ragas_rows = [
        {
            "user_input": r["user_input"],
            "response": r["response"],
            "retrieved_contexts": r["retrieved_contexts"],
            "reference": r["reference"],
        }
        for r in records
    ]
    dataset = EvaluationDataset.from_list(ragas_rows)

    print("\n=== Scoring with RAGAS (frozen records) ===")
    t1 = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_wrap,
        embeddings=emb_wrap,
        show_progress=True,
    )
    print(f"Scoring took {time.time() - t1:.1f}s")

    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in records])
    df.insert(1, "category", [r["category"] for r in records])

    # Write report
    runs_dir = Path("evals/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = Path(args.out) if args.out else runs_dir / f"p3b_report_{stamp}.md"
    csv_path = md_path.with_suffix(".csv")

    df.to_csv(csv_path, index=False)
    md = render_report(
        df,
        items_count=len(items),
        errors=errors,
        judge_model=args.judge_model,
    )
    md_path.write_text(md, encoding="utf-8")

    print(f"\nReport:  {md_path}")
    print(f"Raw CSV: {csv_path}")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for _id, msg in errors:
            print(f"  {_id}: {msg}")


if __name__ == "__main__":
    main()
