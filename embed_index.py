"""
embed_index.py — turns chunks into vectors (embeddings) and builds a
FAISS index over them, so `search.py` can find the closest matches to
a query.

Also handles saving/loading, so you don't have to re-embed the whole
repo every time you run the project — only when the repo changes.
"""
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


def build_vector_index(
    chunks: List[Dict], embedder: Optional[SentenceTransformer] = None
) -> Tuple[SentenceTransformer, "faiss.Index", np.ndarray]:
    """Embed every chunk's content and build a flat FAISS index over it.

    `embedder` defaults to loading the real SentenceTransformer model, but
    can be injected (any object with an `.encode()` method) — tests use this
    to avoid loading a ~130MB model just to check embedding/index shape.
    """
    model = embedder
    if model is None:
        print(f"Loading embedding model ({config.EMBED_MODEL_NAME})...")
        model = SentenceTransformer(config.EMBED_MODEL_NAME, model_kwargs=config.EMBED_MODEL_KWARGS)

    texts = [chunk["content"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index ready — {index.ntotal} vectors, dim={dimension}")

    return model, index, embeddings


def save_index(chunks: List[Dict], embeddings: np.ndarray) -> None:
    """Persist chunks + embeddings to disk so next run can skip re-embedding."""
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    np.save(config.EMBEDDINGS_FILE, embeddings)
    print(f"Saved index to {config.INDEX_DIR}/")


def load_index() -> Tuple[List[Dict], np.ndarray]:
    """Load previously saved chunks + embeddings. Raises if not found."""
    if not config.CHUNKS_FILE.exists() or not config.EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "No saved index found. Run `python main.py --rebuild` first."
        )
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(config.EMBEDDINGS_FILE)
    return chunks, embeddings


def index_exists() -> bool:
    return config.CHUNKS_FILE.exists() and config.EMBEDDINGS_FILE.exists()