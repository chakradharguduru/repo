"""
backend.py — FastAPI wrapper around RepoRAGAgent.

Run with:  uvicorn backend:app --reload
Then app.py (Streamlit) talks to this over HTTP.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_repo
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

# -------------------------
# Load RAG once, at import time
# -------------------------
if index_exists():
    chunks, embeddings = load_index()
    model = None
else:
    chunks = ingest_repo(config.ROOT_DIR)
    model, _, embeddings = build_vector_index(chunks)
    save_index(chunks, embeddings)

search_engine = RepoSearchEngine(chunks, embeddings, model)
agent = RepoRAGAgent(search_engine)


# -------------------------
# Schemas
# -------------------------
from typing import List


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
    return agent.stats()


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return agent.ask(
        query=request.query,
        history=request.history,
        top_k=request.top_k
    )


@app.post("/rebuild")
def rebuild():
    """Re-ingest the repo and rebuild the index without restarting the server."""
    global chunks, embeddings, search_engine, agent
    new_chunks = ingest_repo(config.ROOT_DIR)
    _, _, new_embeddings = build_vector_index(new_chunks)
    save_index(new_chunks, new_embeddings)

    chunks, embeddings = new_chunks, new_embeddings
    search_engine = RepoSearchEngine(chunks, embeddings, None)
    agent = RepoRAGAgent(search_engine)
    return {"status": "rebuilt", "total_chunks": len(chunks)}