"""
backend.py — FastAPI wrapper around RepoRAGAgent.

Run with:  uvicorn backend:app --reload
Then app.py (Streamlit) talks to this over HTTP.
"""
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_repo
from confluence_ingest import ingest_confluence
from embed_index import (
    build_vector_index,
    load_index,
    save_index,
    index_exists,
)
from search import RepoSearchEngine
from rag_agent import RepoRAGAgent
import config

app = FastAPI(title="AXE Repository Assistant API")

# Streamlit runs on a different port (localhost:8501 by default), so it
# needs CORS permission to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ingest_all():
    """Repo chunks + (if configured) Confluence chunks, merged for one index."""
    chunks = ingest_repo(config.ROOT_DIR)
    if config.CONFLUENCE_ENABLED:
        try:
            chunks.extend(ingest_confluence())
        except Exception as e:
            print(f"Confluence ingestion failed, continuing with repo-only index: {e}")
    return chunks


# -------------------------
# RAG agent — lazily built on first use, not at import time.
#
# Building it at import time (the original design) meant `import backend`
# alone would walk config.ROOT_DIR and hit the live Confluence API — which
# breaks `from backend import app` in tests (FastAPI's TestClient, CI) where
# ROOT_DIR/Confluence aren't configured. Routes go through get_agent() so
# tests can pre-seed _state["agent"] with a fake and skip real I/O entirely,
# while normal `uvicorn backend:app` usage is unchanged (built on first
# request instead of at process start).
# -------------------------
_state = {"agent": None}


def get_agent() -> RepoRAGAgent:
    if _state["agent"] is None:
        if index_exists():
            chunks, embeddings = load_index()
            model = None
        else:
            chunks = _ingest_all()
            model, _, embeddings = build_vector_index(chunks)
            save_index(chunks, embeddings)

        search_engine = RepoSearchEngine(chunks, embeddings, model)
        _state["agent"] = RepoRAGAgent(search_engine)
    return _state["agent"]


# -------------------------
# Schemas
# -------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []
    top_k: int = config.DEFAULT_TOP_K


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return get_agent().stats()


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return get_agent().ask(
        query=request.query,
        history=request.history,
        top_k=request.top_k
    )


@app.post("/rebuild")
def rebuild():
    """Re-ingest the repo (+ Confluence) and rebuild the index without restarting the server."""
    new_chunks = _ingest_all()
    _, _, new_embeddings = build_vector_index(new_chunks)
    save_index(new_chunks, new_embeddings)

    search_engine = RepoSearchEngine(new_chunks, new_embeddings, None)
    _state["agent"] = RepoRAGAgent(search_engine)
    return {"status": "rebuilt", "total_chunks": len(new_chunks)}
