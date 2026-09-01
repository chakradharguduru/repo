# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A file-per-concern RAG (retrieval-augmented generation) CLI/web app that lets you ask natural-language
questions about a target codebase (referred to as the "AXE Data Platform" in prompts, but configurable
to any repo via `REPO_ROOT`) plus, optionally, a Confluence space. It walks the target repo, splits
Python files into function/class-level chunks via AST, keeps YAML whole, and (if configured) fetches a
Confluence space and splits pages into heading-level chunks; all of it is embedded locally, indexed
together with FAISS, and answered using an LLM (Llama via Groq's API) grounded in the retrieved
code/docs.

Note: this project (the RAG tool) and the repo it indexes (`ROOT_DIR`/`REPO_ROOT`) are two different
codebases — don't confuse "the repo" in prompts/docstrings (the *indexed target*) with the files here.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # + pytest/pytest-cov/pytest-mock/httpx, for tests/scripts

# Run the CLI (builds index on first run, loads saved index after)
python main.py
python main.py --rebuild     # force re-ingest + re-embed

# Run the web UI (two terminals)
uvicorn backend:app --reload
streamlit run frontend/app.py

# Tests (unit + integration + non-live e2e; @pytest.mark.live excluded by
# default via pyproject.toml's addopts — needs no API keys/secrets)
pytest
pytest --cov=. --cov-report=term-missing --cov-report=html   # + coverage, htmlcov/

# Live E2E test against the real Groq API (needs GROQ_API_KEY in .env)
pytest -m live tests/e2e/test_api_e2e.py

# Evaluation scripts (scripts/) — standalone, run against the REAL saved
# index in index_store/ and (except eval_retrieval.py) the real Groq API.
# Not part of `pytest` at all.
python scripts/generate_test_queries.py   # (re)generate scripts/test_queries.json from real index content
python scripts/eval_retrieval.py          # Top-3 recall, no API key needed
python scripts/eval_hallucination.py      # grounding judge + out-of-scope guard check
python scripts/eval_citations.py          # citation correctness
python scripts/run_evaluation_harness.py  # all of the above in one pass -> scripts/evaluation_report.md
```

Linter: none configured. Build step: none — it's a plain script/CLI project, nothing to compile.

Configuration is via `.env` (see `config.py` for all variables): `GROQ_API_KEY` (required for
answers/routing), `REPO_ROOT` (target repo to index), `INDEX_DIR`, `ROUTER_MODEL`, `GROQ_MODEL`,
`GROQ_TEMPERATURE`, `DEBUG`. `config.py` prints the loaded `.env` contents and resolved env vars to
stdout on import — this is intentional startup diagnostics, not leftover debug code.

Confluence ingestion is opt-in and controlled by `CONFLUENCE_ENABLED` (true only when all three of
`CONFLUENCE_BASE_URL`, `CONFLUENCE_EMAIL`, `CONFLUENCE_API_TOKEN` are set — see `config.py`).
`CONFLUENCE_SPACE_KEY` picks the space; `CONFLUENCE_CACHE_DIR` (default `confluence_cache/`, gitignored)
is where fetched pages are cached as plain text. Auth is HTTP Basic (email + Atlassian API token), not a
bearer token, since this targets Confluence Cloud (`*.atlassian.net/wiki`). Never hardcode these
credentials — they belong in `.env` only.

## Architecture

Fixed pipeline, each stage in its own file — when changing behavior, edit the one file that owns it
rather than touching the pipeline shape:

```
config.py            all settings (paths, model names, thresholds) — other files import from here
ingest.py             walks ROOT_DIR -> list of chunk dicts (AST-splits .py per function/class, keeps .yaml/.yml whole)
confluence_ingest.py  fetches a Confluence space -> list of chunk dicts in the SAME shape (splits pages at h1/h2, caches page text to confluence_cache/ so context.py can re-read it like a repo file)
embed_index.py        chunks -> embeddings (SentenceTransformer) + FAISS IndexFlatL2; save/load to index_store/
search.py             RepoSearchEngine: query -> ranked chunks (semantic + file-name affinity re-ranking)
context.py            retrieved chunks -> full-file text block (re-reads whole source files, deduped)
router.py             query + history -> {mode: general|codebase, standalone_query} via a cheap Groq call
llm.py                (context, query) -> final answer via Groq; separate path for "general" (non-repo/docs) questions
rag_agent.py          RepoRAGAgent.ask() — orchestrates router -> search -> context -> llm for one turn
main.py               CLI entrypoint (REPL loop over rag_agent); merges repo + Confluence chunks before indexing
backend.py            FastAPI wrapper exposing /chat, /stats, /health, /rebuild over rag_agent; same merge via _ingest_all()
frontend/app.py       Streamlit chat UI that calls backend.py over HTTP
```

`confluence_ingest.py` is ingested and merged alongside `ingest.py`'s repo chunks (both entrypoints call
both and concatenate) rather than being a separate pipeline — chunks from either source share the same
dict shape (`chunk_id`, `file_path`, `file_name`, `file_type`, `layer`, `domain`, `chunk_type`,
`chunk_name`, `content`), plus an optional `source_url` that only Confluence chunks set (used by
`context.py` and `rag_agent.py`'s `sources` to point back at the original page instead of the local
cache file). `search.py` needs no changes: Confluence chunks use `layer="confluence"` and
`domain=<space key>`, which the existing generic layer/domain/file-name logic already handles.

Request flow (`rag_agent.RepoRAGAgent.ask`):
1. `router.route_query` classifies the message as `general` (no repo context needed — greetings, math,
   generic concepts) or `codebase`, and for follow-ups rewrites it into a self-contained
   `standalone_query` by resolving references against conversation history (e.g. "what is the table
   name" -> "what table name does survey_summary.py write its data into").
2. `general` -> answered directly by `llm.answer_general`, no retrieval, no sources returned.
3. `codebase` -> `search.RepoSearchEngine.search` retrieves on the **standalone** query; `context.build_context`
   re-reads the **full source files** of matched chunks (not just the isolated chunk) so the LLM sees
   imports/sibling functions/overall flow; `llm.generate_answer` answers using the **original** query
   wording, grounded in that context.

`search.py` file-name matching is a soft re-rank, never a hard filter — a wrong file-name guess can only
reorder results, never exclude the correct chunk. Common English words (see `STOPWORD_ALIASES`) are
excluded from single-word file matching since they're too generic to identify a specific file;
multi-word phrases and exact full-name matches are still promoted. `PROMOTE_MIN_AFFINITY` controls the
cutoff for promoting a file-name match into tier 1 of the ranking.

The index in `index_store/` (`chunks.json` + `embeddings.npy`) is a cache of the last ingest — it goes
stale when the target repo (`ROOT_DIR`) changes and must be rebuilt (`--rebuild` CLI flag or the
`/rebuild` API endpoint) to pick up edits.

`llm.generate_answer` writes the full prompt sent to Groq to `config.PROMPT_LOG_FILE` (default
`prompt.txt`, gitignored) on every call (for inspection/debugging) and prints extensive diagnostics to
stdout — this is by design for this project, not something to strip out. That stdout dump (and
`context.py`'s equivalent) goes through `utils.safe_print` rather than a bare `print()`, because raw
retrieved content can contain characters Windows' default `cp1252` console encoding can't represent
(em dashes, arrows — common from Confluence pages) and a bare `print()` crashes outright on them.

`config.py` also redacts anything that looks like a secret (`KEY`/`TOKEN`/`SECRET`/`PASSWORD` in the
env var name) before printing `.env` contents or `GROQ_API_KEY` at import time — this file gets imported
by every test and by CI, so the raw values must never hit stdout/CI logs.

## Testing & evaluation

```
tests/
  conftest.py         FakeEmbedder (deterministic, no model download) + FakeGroqClient fixtures
  fixtures/            small sample .py/.yaml files + Confluence HTML fixtures ingestion tests run against
  unit/                 one file per module — external LLM calls mocked, no network/real model
  integration/           ingest->embed->FAISS->retrieval per source type; router->retrieval->prompt w/ history rewrite
  e2e/                    FastAPI TestClient scenarios (general/repo-code/confluence-docs) + one @pytest.mark.live test
scripts/
  eval_common.py        shared helpers (load real index/agent, judge_grounding/judge_quality, retry-on-429)
  generate_test_queries.py  samples real chunks from index_store/ -> scripts/test_queries.json (placeholder queries — review before trusting)
  eval_retrieval.py     Top-3 recall vs scripts/test_queries.json (no LLM calls)
  eval_hallucination.py grounding-judge pass + fixed out-of-scope queries (hallucination guard check)
  eval_citations.py     citation correctness (structural + prose-parsed)
  run_evaluation_harness.py  runs everything once per query -> scripts/evaluation_report.md
```

Three refactors exist purely for testability (documented where made): `embed_index.build_vector_index()`
takes an optional `embedder=` so unit tests inject `FakeEmbedder` instead of loading the real ~130MB
SentenceTransformer; `backend.py`'s RAG agent is built lazily via `get_agent()` on first request instead
of at import time, so `TestClient(backend.app)` doesn't walk `ROOT_DIR`/hit Confluence just from
importing the module; `router.route_query`'s `Groq(...)` client construction is inside its own
try/except (previously outside it), matching the function's documented "never raises" contract.

`scripts/eval_*.py` run against the REAL saved index and REAL Groq API (except `eval_retrieval.py`,
which only needs the index) — they are evaluation tooling, not part of the mocked `pytest` suite, and
will burn real API quota. The free/on-demand Groq tier's tokens-per-minute AND tokens-per-day limits are
easy to exhaust running the full query set — `eval_common.with_rate_limit_retry` retries transient 429s,
but a single oversized request (HTTP 413, one retrieved file's full content alone exceeding the TPM
budget) or daily quota exhaustion cannot be retried away. `scripts/run_evaluation_harness.py` never lets
one query's failure crash the run (`eval_common.safe_ask`) — a failed query shows up as `mode: "error"`
in the report instead of losing the whole evaluation.
