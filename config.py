"""
config.py — every setting the project needs, in one place.

Change things here, not inside the other files. That's the whole point
of splitting the project up: ingest.py, search.py, llm.py etc. import
from this file instead of hardcoding values themselves.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # reads a local .env file if present (for GROQ_API_KEY etc.)

env_path = Path(__file__).parent / ".env"

# Startup diagnostics print .env contents so you can eyeball what got loaded —
# kept intentionally (see CLAUDE.md) but values on lines that look like
# secrets (KEY/TOKEN/SECRET/PASSWORD) are masked. This file gets imported by
# every test and by CI, so the raw GROQ_API_KEY / CONFLUENCE_API_TOKEN must
# never hit stdout/CI logs in full.
_SECRET_NAME_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _redact_env_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            name, _, value = line.partition("=")
            if any(hint in name.upper() for hint in _SECRET_NAME_HINTS) and value:
                value = f"***REDACTED(len={len(value)})***"
            line = f"{name}={value}"
        lines.append(line)
    return "\n".join(lines)


def _redact_secret(value):
    if not value:
        return value
    return f"***REDACTED(len={len(value)})***"


print("Config file :", __file__)
print("Expected .env:", env_path)
print("\n========== .env Contents (secrets redacted) ==========")
print(_redact_env_text(env_path.read_text()) if env_path.exists() else "(no .env file found)")
print("===================================\n")
print("Exists:", env_path.exists())

print("=" * 80)
print("Current Working Directory :", os.getcwd())
print("Environment GROQ_MODEL    :", os.getenv("GROQ_MODEL"))
print("Environment REPO_ROOT     :", os.getenv("REPO_ROOT"))
print("Environment GROQ_API_KEY  :", _redact_secret(os.getenv("GROQ_API_KEY")))
print("=" * 80)

# -----------------------------
# Repo to index
# -----------------------------
# Point this at the codebase you want to search. Can be overridden with
# an environment variable so you don't have to edit this file on every
# machine you run the project on.
ROOT_DIR = Path(os.getenv("REPO_ROOT", r"C:\path\to\your\repo"))

# Which file extensions to ingest
SUPPORTED_EXTENSIONS = [".py", ".yaml", ".yml"]

# -----------------------------
# Where the built index is stored
# -----------------------------
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index_store"))
CHUNKS_FILE = INDEX_DIR / "chunks.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"

# -----------------------------
# Embedding model (local, free, no API key needed)
# -----------------------------
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Reproducible hang: the installed transformers/accelerate's default
# low_cpu_mem_usage=True lazily materializes weights on a meta device one
# tensor at a time, and in this environment that path deterministically
# stalls partway through (confirmed: same exact param, every time, even
# with no other load on the machine). Forcing low_cpu_mem_usage=False makes
# it load all weights directly instead — slower to import torch, but it
# actually completes instead of hanging indefinitely.
EMBED_MODEL_KWARGS = {"low_cpu_mem_usage": False}

# Tried groq/compound-mini here to dodge openai/gpt-oss-120b's 200K TPD cap
# — its own 429 confirmed it still routes some requests through that same
# exhausted model, unpredictably. Reverted; a fresh API key resets the
# quota rather than fighting compound's routing.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "openai/gpt-oss-20b")

ALIAS_MIN_SCORE = float(os.getenv("ALIAS_MIN_SCORE", "0"))

# -----------------------------
# Groq (LLM that answers using the retrieved chunks)
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Groq retired the llama-3.x family for this account (confirmed via
# client.models.list() — 404 model_not_found on llama-3.3-70b-versatile).
# openai/gpt-oss-120b has an 8K TPM / 200K TPD free-tier cap that real usage
# (full-file context, a multi-query eval harness) can trip — confirmed both
# 413 "request too large" and 429 daily-quota errors. Tried groq/compound
# and groq/compound-mini as a fix; both turned out to still route some
# requests through this same underlying model and its same exhausted daily
# pool, unpredictably, so it wasn't a real fix. Reverted.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))

# -----------------------------
# Confluence (documentation source, alongside the repo)
# -----------------------------
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "").rstrip("/")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "")

# Fetched pages are cached here as plain text so context.py can re-read them
# the same way it re-reads repo source files (see confluence_ingest.py).
CONFLUENCE_CACHE_DIR = Path(os.getenv("CONFLUENCE_CACHE_DIR", "confluence_cache"))

# Confluence ingestion is opt-in: only runs if all three are set in .env.
CONFLUENCE_ENABLED = bool(CONFLUENCE_BASE_URL and CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN)

# -----------------------------
# Search behavior
# -----------------------------
DEFAULT_TOP_K = 5

# -----------------------------
# Debug logging (filter/alias detection prints)
# -----------------------------
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Where llm.generate_answer() dumps the full prompt sent to Groq (see
# CLAUDE.md — intentional, not leftover debug code). Configurable so tests
# can redirect it to a tmp path instead of littering the repo root.
PROMPT_LOG_FILE = Path(os.getenv("PROMPT_LOG_FILE", "prompt.txt"))