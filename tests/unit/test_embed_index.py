"""Unit tests for embed_index.py — embedding shape, FAISS index creation,
and the save/load index_store/ roundtrip. Uses the injectable `embedder`
param (added for testability) instead of the real SentenceTransformer."""
import json

import numpy as np
import pytest

import config
from embed_index import build_vector_index, index_exists, load_index, save_index


@pytest.fixture
def three_chunks():
    return [
        {"chunk_id": "1", "content": "alpha beta gamma", "file_path": "a.py",
         "file_name": "a", "file_type": "python", "layer": None, "domain": None,
         "chunk_type": "function", "chunk_name": "f1"},
        {"chunk_id": "2", "content": "delta epsilon zeta", "file_path": "b.yaml",
         "file_name": "b", "file_type": "yaml", "layer": None, "domain": None,
         "chunk_type": "pipeline_config", "chunk_name": "b"},
        {"chunk_id": "3", "content": "eta theta iota", "file_path": "c.py",
         "file_name": "c", "file_type": "python", "layer": None, "domain": None,
         "chunk_type": "function", "chunk_name": "f2"},
    ]


# ---------------------------------------------------------------------
# build_vector_index — embedding shape + FAISS index
# ---------------------------------------------------------------------
def test_build_vector_index_embedding_shape(three_chunks, fake_embedder):
    model, index, embeddings = build_vector_index(three_chunks, embedder=fake_embedder)

    assert embeddings.shape == (3, fake_embedder.dim)
    assert model is fake_embedder


def test_build_vector_index_faiss_index_vector_count(three_chunks, fake_embedder):
    _, index, embeddings = build_vector_index(three_chunks, embedder=fake_embedder)

    assert index.ntotal == 3
    assert index.d == fake_embedder.dim


def test_build_vector_index_faiss_search_returns_results(three_chunks, fake_embedder):
    _, index, embeddings = build_vector_index(three_chunks, embedder=fake_embedder)

    # Querying with a chunk's own embedding should return that chunk as the
    # closest (distance ~0) match — the basic IndexFlatL2 contract.
    query = embeddings[0:1]
    distances, indices = index.search(query, k=2)

    assert indices.shape == (1, 2)
    assert indices[0][0] == 0
    assert distances[0][0] == pytest.approx(0.0, abs=1e-4)


def test_build_vector_index_empty_chunks(fake_embedder):
    model, index, embeddings = build_vector_index([], embedder=fake_embedder)
    assert embeddings.shape[0] == 0
    assert index.ntotal == 0


# ---------------------------------------------------------------------
# save_index / load_index / index_exists — roundtrip against tmp_path
# ---------------------------------------------------------------------
@pytest.fixture
def isolated_index_dir(tmp_path, monkeypatch):
    index_dir = tmp_path / "index_store"
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "CHUNKS_FILE", index_dir / "chunks.json")
    monkeypatch.setattr(config, "EMBEDDINGS_FILE", index_dir / "embeddings.npy")
    return index_dir


def test_index_exists_false_before_save(isolated_index_dir):
    assert index_exists() is False


def test_save_and_load_index_roundtrip(isolated_index_dir, three_chunks, fake_embedder):
    _, _, embeddings = build_vector_index(three_chunks, embedder=fake_embedder)

    save_index(three_chunks, embeddings)
    assert index_exists() is True

    loaded_chunks, loaded_embeddings = load_index()
    assert loaded_chunks == three_chunks
    np.testing.assert_array_almost_equal(loaded_embeddings, embeddings)


def test_save_index_writes_valid_json(isolated_index_dir, three_chunks, fake_embedder):
    _, _, embeddings = build_vector_index(three_chunks, embedder=fake_embedder)
    save_index(three_chunks, embeddings)

    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 3
    assert data[0]["chunk_id"] == "1"


def test_load_index_missing_raises(isolated_index_dir):
    with pytest.raises(FileNotFoundError):
        load_index()
