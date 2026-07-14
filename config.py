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

print("Config file :", __file__)
print("Expected .env:", env_path)
print("\n========== .env Contents ==========")
print(env_path.read_text())
print("===================================\n")
print("Exists:", env_path.exists())

print("=" * 80)
print("Current Working Directory :", os.getcwd())
print("Environment GROQ_MODEL    :", os.getenv("GROQ_MODEL"))
print("Environment REPO_ROOT     :", os.getenv("REPO_ROOT"))
print("Environment GROQ_API_KEY  :", os.getenv("GROQ_API_KEY"))
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

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant")

ALIAS_MIN_SCORE = float(os.getenv("ALIAS_MIN_SCORE", "0"))

# -----------------------------
# Groq (LLM that answers using the retrieved chunks)
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# llama-3.3-70b-versatile is the model your working notebook used — much
# stronger reasoning than the smaller 8b-instant default, which matters
# for multi-step "what transformations happen here" style questions.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))

# -----------------------------
# Search behavior
# -----------------------------
DEFAULT_TOP_K = 5

# -----------------------------
# Debug logging (filter/alias detection prints)
# -----------------------------
DEBUG = os.getenv("DEBUG", "false").lower() == "true"