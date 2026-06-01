# CLAUDE.md — CivicAI working conventions

Guidance for Claude Code when working in this repo. Read before making changes.

## What this project is

CivicAI is a conversational assistant (Claude + LangGraph) that helps French expats
navigate Thai administrative procedures. It uses **multilingual hybrid RAG**: dense
retrieval (bge-m3) → cross-encoder rerank (bge-reranker-v2-m3) → web-search fallback
(Tavily) when the top reranked score falls below a threshold.

This is a **portfolio project that runs locally** — there is no deployment. Docker is
for local reproducibility, not a hosting target. Do not add deployment claims, hosting
config, or "deployable"-style wording unless explicitly asked.

## Architecture (where things live)

- `src/civicai/config.py` — **single source of truth for every constant** (model names,
  threshold, top_k/top_n, chunk sizes, paths, collection name). Nothing is hardcoded
  elsewhere; new constants go here.
- `src/civicai/rag/` — embeddings, vectorstore, retrieval, reranker, ingest.
- `src/civicai/agent/` — LangGraph state, nodes, graph.
- `src/civicai/tools/` — search_docs (retrieve→rerank→route), web_search, dispatcher.
- `src/civicai/api/` — FastAPI app factory, routes, schemas.
- `evals/` — RAGAS evaluation harness (see below).
- `tests/` — mocked pytest suite, no network calls.

## Hard rules

- **Package manager is `uv`.** Use `uv sync`, `uv run`. Never introduce pip/poetry/conda.
- **torch is CPU-only** (pinned via `[tool.uv.sources]` → pytorch-cpu index). The image is
  ~2 GB because the CUDA runtime is stripped. Do not let a dependency pull a CUDA/nvidia
  torch build back in; after any lock change, grep `uv.lock` for `nvidia`/`cuda`/`triton`
  and confirm they're absent.
- **Work on a feature branch** (`feat/...`), then merge to `main`. Don't commit directly
  to main.
- **Small, logically-grouped commits** with clear messages (the existing history uses
  `feat(rag):`, `evals:`, `build(docker):`, `docs(README):` style — follow it).
- **Tests must stay green** (`uv run pytest`, currently 29/29). They use mocks for
  Anthropic, Tavily, ChromaDB, and the embedder — keep them network-free and fast.
- **In-code prompts stay in French** (users query in French). Everything else — README,
  comments, docs — is in English.
- **README signature block** must remain exactly:
  ```
  ## Author

  **Alexis Monnier** — [@AlexMo-1205](https://github.com/AlexMo-1205)

  ML/AI Engineer - Data Scientist | Bangkok, Thailand
  ```

## Behavior vs. structure

The retrieval logic, the 0.67 routing threshold, the prompts, and the model choices are
**measured, deliberate decisions** — not defaults to tweak casually. Treat any change to
them as a behavior change:

- Don't alter the threshold, prompts, top_k/top_n, or chunking without a reason, and when
  there is one, **validate it against the eval set**, don't just assert it's fine.
- Refactors must be behavior-preserving. If a change could shift outputs, say so and
  validate before committing.

## The eval harness (`evals/`)

Three decoupled phases — keep them decoupled (this separation exists because coupling them
caused an out-of-memory crash):

1. **Pipeline** — `evals/runner.py --no-score` runs the agent once per dataset item and
   freezes records to `evals/runs/p3b_records.jsonl`.
2. **Scoring** — `evals/scorer.py` is **standalone and must NOT import `civicai.rag.*`**
   (importing it eagerly loads bge-m3 + the reranker, ~3 GB, which OOM-kills the scorer
   alongside RAGAS). It reads the frozen records, scores with RAGAS (Claude judge + bge-m3
   embeddings — never the OpenAI default), appends to `p3b_results.jsonl`.
3. **Sweep** — `evals/sweep.py` is **pure data, zero LLM calls**: it recomputes the routing
   decision per threshold over cached scores. Never make the sweep re-score.

Dataset: `evals/dataset.jsonl`, 73 French items in three categories — `local` (KB answers),
`fallback` (KB lacks it, should route to web), `adversarial` (false premise to correct).
Routing accuracy is computed over `local`+`fallback` only; `adversarial` is excluded from
routing math and reported separately.

## Operating principles (learned the hard way)

- **Long/expensive runs must be incremental + resumable.** Append-flush per item; on
  restart, skip already-done items. A run that holds everything in memory and writes at the
  end will lose everything if it dies.
- **Never re-launch a failed run blind.** First find out *why* it died (rate-limit vs OOM
  vs exception on one item) and report it. Each full RAGAS run costs real Anthropic tokens.
- **Don't dismiss an outlier as "noise" without checking.** A reproducible 0.000 or NaN is
  a signal, not noise — re-score or inspect before concluding.
- **Optimize the product, not the metric.** If a metric penalizes correct behavior (e.g.
  RAGAS answer_relevancy misfires on honest "this figure isn't in my sources" answers), keep
  the right behavior and document the metric's limitation. Do not degrade the product to
  chase a score.
- **Stop at decision points.** For multi-step work, pause and report before irreversible or
  expensive steps rather than pushing straight through.
