"""P3c — routing-threshold sweep over cached P3b results.

ZERO API CALLS, no model loads. Reads three on-disk JSONL files and
computes, for each candidate threshold T, what the routing decision
would be and how the cached RAGAS metrics group up under that decision.

If a needed score isn't cached, the sweep STOPS and tells you — it never
falls back to live LLM calls.

Inputs:
    evals/dataset.jsonl                         (categories)
    evals/runs/p3b_top_scores.jsonl             (top reranker score per item)
    evals/runs/p3b_results.jsonl                (per-item RAGAS metrics)

Output:
    evals/runs/p3c_sweep_report_<ts>.md
    evals/runs/p3c_sweep.csv                    (raw sweep table)

Run:
    uv run python evals/sweep.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Config knobs (cost weights documented in the report)
# ---------------------------------------------------------------------------
# Cost weights: a fallback wrongly kept local produces an UNGROUNDED answer
# from local chunks (no relevant info on disk). A local wrongly sent to web
# usually still produces a correct answer, just with latency / cost overhead.
COST_LOCAL_TO_WEB = 1.0
COST_FALLBACK_TO_LOCAL = 3.0

# Sweep grid: coarse pass over 0.30 -> 0.80, fine pass over overlap window.
COARSE = [round(0.30 + 0.05 * i, 2) for i in range(11)]            # 0.30 .. 0.80
FINE = [round(0.50 + 0.01 * i, 2) for i in range(18)]              # 0.50 .. 0.67


# ---------------------------------------------------------------------------
# Load cached data
# ---------------------------------------------------------------------------
def _load(path: Path) -> list[dict]:
    if not path.exists():
        sys.stderr.write(f"FATAL: missing cache file {path}\n")
        sys.exit(2)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_inputs():
    items = _load(Path("evals/dataset.jsonl"))
    scores = _load(Path("evals/runs/p3b_top_scores.jsonl"))
    results = _load(Path("evals/runs/p3b_results.jsonl"))

    by_id_score = {r["id"]: r for r in scores}
    by_id_res = {r["id"]: r for r in results}

    # Sanity: every dataset item must have both a score and a result
    missing_score = [it["id"] for it in items if it["id"] not in by_id_score]
    missing_res = [it["id"] for it in items if it["id"] not in by_id_res]
    if missing_score or missing_res:
        sys.stderr.write(
            f"FATAL: cache incomplete. missing top_score for: {missing_score} | "
            f"missing RAGAS results for: {missing_res}\n"
            "STOPPING — this script does not call any LLM/judge.\n"
        )
        sys.exit(3)

    merged = []
    for it in items:
        s = by_id_score[it["id"]]
        r = by_id_res[it["id"]]
        merged.append({
            "id": it["id"],
            "category": it["category"],
            "top_score": s["top_score"],
            "faithfulness": r.get("faithfulness", float("nan")),
            "answer_relevancy": r.get("answer_relevancy", float("nan")),
            "context_precision": r.get("context_precision", float("nan")),
            "context_recall": r.get("context_recall", float("nan")),
        })
    return merged


# ---------------------------------------------------------------------------
# Routing math
# ---------------------------------------------------------------------------
def _route(top_score: float, threshold: float) -> str:
    return "web" if top_score < threshold else "local"


def _is_correct(category: str, decision: str) -> bool:
    expected = "web" if category == "fallback" else "local"  # adversarial -> local
    return decision == expected


def _safe_mean(values):
    clean = [v for v in values if v == v]
    return statistics.fmean(clean) if clean else float("nan")


def evaluate_threshold(items: list[dict], T: float) -> dict:
    """Compute routing + faithfulness aggregates at threshold T.

    Routing-accuracy is computed over local+fallback ONLY (adversarial excluded).
    Faithfulness MEAN excludes NaN (so multi-01 doesn't poison the average).
    """
    routing_items = [it for it in items if it["category"] in ("local", "fallback")]
    local_misrouted: list[str] = []     # local item sent to web
    fallback_missed: list[str] = []     # fallback item kept local
    correct = 0

    for it in routing_items:
        dec = _route(it["top_score"], T)
        if _is_correct(it["category"], dec):
            correct += 1
        else:
            if it["category"] == "local":
                local_misrouted.append(it["id"])
            else:
                fallback_missed.append(it["id"])

    n = len(routing_items)
    accuracy = correct / n if n else float("nan")
    error_cost = (
        len(local_misrouted) * COST_LOCAL_TO_WEB
        + len(fallback_missed) * COST_FALLBACK_TO_LOCAL
    )

    # Per-category cached-metric means over the same routing items
    local_items = [it for it in routing_items if it["category"] == "local"]
    fallback_items = [it for it in routing_items if it["category"] == "fallback"]

    return {
        "threshold": T,
        "n_routing": n,
        "routing_accuracy": accuracy,
        "local_to_web_count": len(local_misrouted),
        "local_to_web_ids": local_misrouted,
        "fallback_to_local_count": len(fallback_missed),
        "fallback_to_local_ids": fallback_missed,
        "error_cost": error_cost,
        # Cached metric means (do NOT depend on T — they describe the items
        # in each category; they are the same across T but reported for context)
        "local_faithfulness_mean": _safe_mean([it["faithfulness"] for it in local_items]),
        "local_answer_relevancy_mean": _safe_mean([it["answer_relevancy"] for it in local_items]),
        "local_context_precision_mean": _safe_mean([it["context_precision"] for it in local_items]),
        "local_context_recall_mean": _safe_mean([it["context_recall"] for it in local_items]),
        "fallback_faithfulness_mean": _safe_mean([it["faithfulness"] for it in fallback_items]),
    }


def select_best(results: list[dict]) -> dict:
    """Lowest error_cost wins; ties broken by higher routing_accuracy."""
    return min(results, key=lambda r: (r["error_cost"], -r["routing_accuracy"]))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(items: list[dict], coarse: list[dict], fine: list[dict],
                  best: dict) -> str:
    lines = []
    lines.append("# P3c — routing-threshold sweep\n")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Input cache: dataset (73 items), p3b_top_scores (73), p3b_results (73)")
    lines.append("- Sweep is pure-data: zero LLM calls, no model loads.")
    lines.append("- Routing accuracy computed over `local` + `fallback` only "
                 "(adversarial excluded — they are not a routing question).")
    lines.append("- multi-01 has NaN faithfulness from the P3b run (RAGAS statement "
                 "extractor returned 0 statements on a very long response); it is "
                 "still counted for routing but excluded from faithfulness means.")
    lines.append("")
    lines.append("## Cost model\n")
    lines.append(
        f"Routing errors are not equal in harm. We minimize a weighted error cost:\n\n"
        f"- `local → web` (false fallback): cost = **{COST_LOCAL_TO_WEB}**. "
        f"Latency / Tavily cost only — the web answer is usually still correct.\n"
        f"- `fallback → local` (missed fallback): cost = **{COST_FALLBACK_TO_LOCAL}**. "
        f"Produces an ungrounded answer from local chunks that don't contain the fact.\n"
    )

    def _row(r):
        return (
            f"| {r['threshold']:.2f} | "
            f"{r['routing_accuracy']*100:.1f}% | "
            f"{r['local_to_web_count']} | "
            f"{r['fallback_to_local_count']} | "
            f"{r['error_cost']:.1f} |"
        )

    def _table(rows, title):
        lines.append(f"\n## {title}\n")
        lines.append("| T | routing acc. | local→web (false fb) | fallback→local (missed fb) | weighted error cost |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            lines.append(_row(r))

    _table(coarse, "Coarse sweep (T = 0.30 → 0.80, step 0.05)")
    _table(fine, "Fine sweep (T = 0.50 → 0.67, step 0.01) — the overlap window")

    # Recommendation
    lines.append("\n## Recommended threshold\n")
    lines.append(
        f"**T = {best['threshold']:.2f}**  "
        f"→ routing accuracy = {best['routing_accuracy']*100:.1f}%, "
        f"weighted error cost = {best['error_cost']:.1f} "
        f"(local→web = {best['local_to_web_count']}, "
        f"fallback→local = {best['fallback_to_local_count']})\n"
    )
    lines.append(
        "Trade in one line: this T minimizes the cost-weighted error, "
        "protecting against ungrounded local answers (high-harm) while accepting "
        "a small number of `local → web` rerouting (low-harm).\n"
    )
    if best["local_to_web_ids"]:
        lines.append("**Local items sent to web at this T** (low harm — web answer usually still correct):")
        for i in best["local_to_web_ids"]:
            lines.append(f"- `{i}`")
    if best["fallback_to_local_ids"]:
        lines.append("\n**Fallback items kept local at this T** (high harm — answer comes from chunks that don't contain the fact):")
        for i in best["fallback_to_local_ids"]:
            lines.append(f"- `{i}`")

    lines.append(
        "\n## Cached RAGAS means at the recommended T (informational only — "
        "these describe item quality, not the threshold)\n"
    )
    lines.append("| Category | faithfulness | answer_relevancy | context_precision | context_recall |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| local    | {best['local_faithfulness_mean']:.3f} | "
        f"{best['local_answer_relevancy_mean']:.3f} | "
        f"{best['local_context_precision_mean']:.3f} | "
        f"{best['local_context_recall_mean']:.3f} |"
    )
    lines.append(
        f"| fallback | {best['fallback_faithfulness_mean']:.3f} | "
        f"— | — | — |"
    )

    lines.append("\n## Known ceiling — why ~90% is the cap\n")
    lines.append(
        "The reranker scores **single-chunk relevance** between the query and one "
        "document chunk. Several `local` multi-doc questions (e.g. `multi-01`, `multi-06`, "
        "`multi-12`) cannot be answered from any one chunk — the answer is spread across "
        "2–4 docs — so their top reranker score is low (0.50–0.57). Those scores sit in "
        "the same band as the `fallback` near-misses (`fall-04`, `fall-09`, `fall-05`, "
        "`fall-06`), where chunks read as topical but the specific fact is absent.\n"
    )
    lines.append(
        "No reranker-score threshold can separate those two groups cleanly. Surpassing "
        "the ~90% ceiling would require either:\n\n"
        "- a dedicated **answerability classifier** on the top reranked chunk "
        "(does this chunk actually contain the answer, beyond being on-topic), or\n"
        "- fine-tuning the reranker on a `(query, chunk) -> answer-present` signal.\n\n"
        "**Both are out of scope for P3c.** This sweep finds the best operating point "
        "given the current reranker and corpus."
    )

    return "\n".join(lines) + "\n"


def write_csv(path: Path, results: list[dict]) -> None:
    cols = ["threshold", "routing_accuracy", "local_to_web_count",
            "fallback_to_local_count", "error_cost"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r[c] for c in cols])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="After the sweep, set config.rerank_routing_threshold "
                             "to the recommended value.")
    args = parser.parse_args()

    items = load_inputs()
    print(f"Loaded {len(items)} cached items.")

    coarse_results = [evaluate_threshold(items, T) for T in COARSE]
    fine_results = [evaluate_threshold(items, T) for T in FINE]
    all_results = coarse_results + fine_results
    best = select_best(all_results)

    out_dir = Path("evals/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"p3c_sweep_report_{stamp}.md"
    csv_path = out_dir / "p3c_sweep.csv"

    write_csv(csv_path, all_results)
    md_path.write_text(render_report(items, coarse_results, fine_results, best),
                       encoding="utf-8")

    print(f"\nBest T = {best['threshold']:.2f}  "
          f"acc={best['routing_accuracy']*100:.1f}%  "
          f"local->web={best['local_to_web_count']}  "
          f"fallback->local={best['fallback_to_local_count']}  "
          f"cost={best['error_cost']:.1f}")
    print(f"Report: {md_path}")
    print(f"CSV:    {csv_path}")

    if args.apply:
        update_config(best["threshold"])
        print(f"\nUpdated config.rerank_routing_threshold -> {best['threshold']:.2f}")


def update_config(new_value: float) -> None:
    """Replace the rerank_routing_threshold value in src/civicai/config.py."""
    import re

    cfg = Path("src/civicai/config.py")
    text = cfg.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"(rerank_routing_threshold\s*=\s*)[0-9.]+",
        rf"\g<1>{new_value:.2f}",
        text,
        count=1,
    )
    if n != 1:
        sys.stderr.write("FATAL: could not find rerank_routing_threshold assignment "
                         "in src/civicai/config.py\n")
        sys.exit(4)
    cfg.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    main()
