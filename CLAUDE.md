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
pip install -r requirements.txt

# Run the CLI (builds index on first run, loads saved index after)
python main.py
python main.py --rebuild     # force re-ingest + re-embed

# Run the web UI (two terminals)
uvicorn backend:app --reload
streamlit run frontend/app.py
```

There is no test suite, linter, or build step configured in this repo.

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

`llm.generate_answer` writes the full prompt sent to Groq to `prompt.txt` on every call (for
inspection/debugging) and prints extensive diagnostics to stdout — this is by design for this project,
not something to strip out.
