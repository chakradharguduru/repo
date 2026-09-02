"""
eval_common.py — shared helpers for the scripts/eval_*.py and
run_evaluation_harness.py scripts. Not a pytest module: these scripts run
against the REAL saved index (index_store/) and, where noted, the real Groq
API, as standalone evaluation tooling — separate from the mocked, offline
pytest suite in tests/.
"""
from pathlib import Path
from typing import Dict, List
import json
import re
import sys
import time

SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_QUERIES_FILE = SCRIPTS_DIR / "test_queries.json"

sys.path.insert(0, str(REPO_ROOT))  # so `import config`, `import search`, etc. work run from anywhere

import config  # noqa: E402
from embed_index import load_index, index_exists  # noqa: E402
from search import RepoSearchEngine  # noqa: E402
from rag_agent import RepoRAGAgent  # noqa: E402
from groq import RateLimitError  # noqa: E402


def with_rate_limit_retry(func, *args, max_retries: int = 2, backoff_seconds: int = 25, **kwargs):
    """Retry ONLY on groq.RateLimitError (HTTP 429 — a transient, will-clear
    condition on the free tier's tokens-per-minute budget). A 413 "request
    too large" is a different, permanent failure (that single request will
    never fit no matter how long you wait) and must NOT be retried here —
    it's left to the caller's normal error handling.
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            if attempt == max_retries:
                raise
            print(f"    Rate limited — waiting {backoff_seconds}s before retry "
                  f"({attempt + 1}/{max_retries})...")
            time.sleep(backoff_seconds)

# ---------------------------------------------------------------------
# Deliberately fabricated references for the hallucination-guard check —
# nothing in a real AXE-style repo or its Confluence space should match
# these, so the system should decline rather than invent an answer.
# ---------------------------------------------------------------------
OUT_OF_SCOPE_QUERIES = [
    "is there any information on auto loader or spark structured streaming?",
    "Summarize the blockchain_integration Confluence page.",
    "is there any information on powerBi import to AI/BI Dashboards",
]

_NOT_FOUND_PHRASES = (
    "don't find", "do not find", "couldn't find", "could not find",
    "no information", "not contain", "doesn't contain", "does not contain",
    "not available in", "i don't know", "unable to find", "no relevant",
)


def guard_triggered(answer: str) -> bool:
    """True if the answer looks like a "not found in indexed sources"
    decline rather than a fabricated/invented claim."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _NOT_FOUND_PHRASES)


# Matches bracketed or bare filename-looking tokens in answer prose, e.g.
# "[givex.py]", "(order.txt)", "survey_summary.py".
_FILENAME_TOKEN_RE = re.compile(r"[\[\(`]?([a-zA-Z0-9_\-]+\.(?:py|yaml|yml|txt))[\]\)`]?")


def extract_cited_filenames(answer: str) -> set:
    return {m.lower() for m in _FILENAME_TOKEN_RE.findall(answer)}


_JUDGE_SYSTEM = (
    "You are a strict fact-checker for a RAG system. Given SOURCE MATERIAL "
    "and an ANSWER, decide whether every factual claim in the ANSWER is "
    "actually supported by the SOURCE MATERIAL. An answer that says it "
    "could not find the information is always grounded (it isn't claiming "
    "any unsupported fact). Respond with ONLY a JSON object: "
    '{"grounded": true|false, "reason": "<one short sentence>"}'
)

_JUDGE_TEMPLATE = """SOURCE MATERIAL:
{context}

ANSWER TO CHECK:
{answer}
"""


def judge_grounding(client, context_block: str, answer: str) -> dict:
    """Second Groq call: does `answer` claim anything not supported by
    `context_block`? Returns {"grounded": bool|None, "reason": str}."""
    prompt = _JUDGE_TEMPLATE.format(context=context_block[:8000], answer=answer)
    try:
        resp = with_rate_limit_retry(
            client.chat.completions.create,
            model=config.GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"grounded": None, "reason": f"judge call failed: {e}"}


_QUALITY_SYSTEM = (
    "You are evaluating the quality of an AI assistant's answer to a "
    "question about a codebase and its documentation. Rate the answer from "
    "1 (poor: irrelevant, wrong, or unhelpful) to 5 (excellent: accurate, "
    "complete, clearly grounded in and citing the source material). "
    'Respond with ONLY a JSON object: {"score": 1-5, "justification": "<one sentence>"}'
)

_QUALITY_TEMPLATE = """QUESTION:
{question}

SOURCE MATERIAL:
{context}

ANSWER TO RATE:
{answer}
"""


def judge_quality(client, question: str, context_block: str, answer: str) -> dict:
    """Third Groq call: LLM-assisted 1-5 answer-quality score."""
    prompt = _QUALITY_TEMPLATE.format(question=question, context=context_block[:8000], answer=answer)
    try:
        resp = with_rate_limit_retry(
            client.chat.completions.create,
            model=config.GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _QUALITY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"score": None, "justification": f"judge call failed: {e}"}


def load_engine() -> RepoSearchEngine:
    """Load the real saved index (index_store/) and build a search engine
    over it, using the real embedding model — these scripts evaluate the
    actual production pipeline, not a test double."""
    if not index_exists():
        print(
            f"No saved index found at {config.INDEX_DIR}/. "
            "Run `python main.py --rebuild` first so there's something to evaluate."
        )
        sys.exit(1)
    chunks, embeddings = load_index()
    return RepoSearchEngine(chunks, embeddings, model=None)


def load_agent() -> RepoRAGAgent:
    return RepoRAGAgent(load_engine())


def safe_ask(agent: RepoRAGAgent, query: str, **kwargs) -> Dict:
    """agent.ask() that never raises — a single query's API failure (rate
    limit, oversized request, transient network error) must not take down
    an entire eval run over 10+ queries each making several real Groq
    calls. Failures come back as a normal-shaped result with mode="error"
    so callers can report them per-query instead of crashing.
    """
    try:
        return with_rate_limit_retry(agent.ask, query, **kwargs)
    except Exception as e:
        return {
            "answer": f"ERROR: {e}",
            "mode": "error",
            "standalone_query": None,
            "elapsed_seconds": None,
            "sources": [],
        }


def load_test_queries(path: Path = DEFAULT_QUERIES_FILE) -> List[Dict]:
    if not path.exists():
        print(
            f"No test query file at {path}.\n"
            "Generate one with: python scripts/generate_test_queries.py"
        )
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_table(rows: List[List[str]], headers: List[str], col_widths=None) -> None:
    """Minimal dependency-free table printer (no external table library)."""
    if col_widths is None:
        col_widths = [
            max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) if rows else len(str(headers[i]))
            for i in range(len(headers))
        ]
        col_widths = [min(w, 60) for w in col_widths]

    def fmt_row(cells):
        return " | ".join(str(c)[:w].ljust(w) for c, w in zip(cells, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))
