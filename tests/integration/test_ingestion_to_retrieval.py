"""Integration tests: real ingest_repo() / ingest_confluence() / build_vector_index()
/ RepoSearchEngine wiring, end to end, for each of the 3 source types this
project actually supports (Python, YAML, Confluence).

The embedding step uses the injected FakeEmbedder (see conftest.py) instead
of the real SentenceTransformer — this test verifies the PIPELINE WIRING
(ingest -> embed -> FAISS -> retrieval all correctly pass data to each
other and a targeted query finds the right chunk), not the semantic quality
of the real BGE model, which is out of this project's scope to re-test.
Confluence's network layer (requests.get) is mocked; everything else in the
chain — chunking, FAISS index construction, ranking — is real code.
"""
from pathlib import Path
from unittest.mock import patch

import config
import confluence_ingest as ci
from embed_index import build_vector_index
from ingest import ingest_repo
from search import RepoSearchEngine

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _engine_from(chunks, fake_embedder):
    _, _, embeddings = build_vector_index(chunks, embedder=fake_embedder)
    return RepoSearchEngine(chunks, embeddings, fake_embedder)


def test_python_source_ingestion_to_retrieval(sample_repo_dir, fake_embedder):
    chunks = ingest_repo(sample_repo_dir)
    engine = _engine_from(chunks, fake_embedder)

    results = engine.search("how do I compute the average of the rows", top_k=3)

    assert any(r["chunk_name"] == "compute_average" and r["file_type"] == "python"
               for r in results)


def test_yaml_source_ingestion_to_retrieval(sample_repo_dir, fake_embedder):
    chunks = ingest_repo(sample_repo_dir)
    engine = _engine_from(chunks, fake_embedder)

    results = engine.search("pipeline schedule cron staging sales raw", top_k=3)

    assert any(r["file_type"] == "yaml" and r["chunk_name"] == "sales_pipeline"
               for r in results)


def test_confluence_source_ingestion_to_retrieval(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setattr(config, "CONFLUENCE_CACHE_DIR", tmp_path / "confluence_cache")
    html = (FIXTURES_DIR / "confluence" / "page_with_headings.html").read_text()
    page = {"id": "999", "title": "Loyalty", "body": {"storage": {"value": html}}}

    with patch.object(ci, "fetch_pages", return_value=[page]):
        chunks = ci.ingest_confluence("DPT")

    engine = _engine_from(chunks, fake_embedder)
    results = engine.search("what are the primary keys store_id member_id", top_k=3)

    assert any(r["file_type"] == "confluence" and r["chunk_name"] == "Primary Keys"
               for r in results)


def test_mixed_corpus_retains_source_type_diversity(sample_repo_dir, tmp_path, monkeypatch, fake_embedder):
    """All 3 source types coexist in one index (the real production setup:
    main.py/backend.py concatenate repo + Confluence chunks before embedding)."""
    monkeypatch.setattr(config, "CONFLUENCE_CACHE_DIR", tmp_path / "confluence_cache")
    html = (FIXTURES_DIR / "confluence" / "page_with_headings.html").read_text()
    page = {"id": "999", "title": "Loyalty", "body": {"storage": {"value": html}}}

    repo_chunks = ingest_repo(sample_repo_dir)
    with patch.object(ci, "fetch_pages", return_value=[page]):
        confluence_chunks = ci.ingest_confluence("DPT")

    all_chunks = repo_chunks + confluence_chunks
    engine = _engine_from(all_chunks, fake_embedder)

    assert {c["file_type"] for c in all_chunks} == {"python", "yaml", "confluence"}

    py_result = engine.search("how do I compute the average of the rows", top_k=1)
    conf_result = engine.search("what are the primary keys store_id member_id", top_k=1)
    assert py_result[0]["file_type"] == "python"
    assert conf_result[0]["file_type"] == "confluence"
