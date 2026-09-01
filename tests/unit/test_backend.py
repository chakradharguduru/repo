"""Unit tests for backend.py's non-route logic: _ingest_all() and
get_agent()'s cold-start path (building the index the first time, when
_state["agent"] is still None). All ingestion/embedding/disk I/O is
mocked — these aren't integration tests, just verifying backend.py wires
the pieces together correctly."""
from unittest.mock import MagicMock, patch

import pytest

import backend
import config


@pytest.fixture(autouse=True)
def reset_backend_state():
    """Every test in this file mutates backend._state — never let it leak
    into other test modules that import backend.app."""
    backend._state["agent"] = None
    yield
    backend._state["agent"] = None


def test_ingest_all_merges_repo_and_confluence_chunks(monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_ENABLED", True)
    repo_chunks = [{"chunk_id": "r1"}]
    confluence_chunks = [{"chunk_id": "c1"}]
    # _ingest_all() extends the list ingest_repo() returns in place — compute
    # the expected value up front, since repo_chunks itself gets mutated.
    expected = repo_chunks + confluence_chunks

    with patch.object(backend, "ingest_repo", return_value=repo_chunks) as mock_repo, \
         patch.object(backend, "ingest_confluence", return_value=confluence_chunks) as mock_conf:
        result = backend._ingest_all()

    mock_repo.assert_called_once_with(config.ROOT_DIR)
    mock_conf.assert_called_once()
    assert result == expected


def test_ingest_all_skips_confluence_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_ENABLED", False)
    repo_chunks = [{"chunk_id": "r1"}]

    with patch.object(backend, "ingest_repo", return_value=repo_chunks), \
         patch.object(backend, "ingest_confluence") as mock_conf:
        result = backend._ingest_all()

    mock_conf.assert_not_called()
    assert result == repo_chunks


def test_ingest_all_continues_with_repo_only_if_confluence_fails(monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_ENABLED", True)
    repo_chunks = [{"chunk_id": "r1"}]

    with patch.object(backend, "ingest_repo", return_value=repo_chunks), \
         patch.object(backend, "ingest_confluence", side_effect=RuntimeError("network down")):
        result = backend._ingest_all()

    assert result == repo_chunks  # doesn't raise, doesn't lose the repo chunks


def test_get_agent_builds_fresh_index_when_none_saved(monkeypatch):
    """Cold-start path: no saved index_store/, no cached agent yet."""
    chunks = [{"chunk_id": "1", "content": "x"}]
    fake_model = MagicMock()
    fake_embeddings = MagicMock()

    with patch.object(backend, "index_exists", return_value=False), \
         patch.object(backend, "_ingest_all", return_value=chunks) as mock_ingest_all, \
         patch.object(backend, "build_vector_index", return_value=(fake_model, MagicMock(), fake_embeddings)) as mock_build, \
         patch.object(backend, "save_index") as mock_save, \
         patch.object(backend, "RepoSearchEngine") as mock_engine_cls:
        agent = backend.get_agent()

    mock_ingest_all.assert_called_once()
    mock_build.assert_called_once_with(chunks)
    mock_save.assert_called_once_with(chunks, fake_embeddings)
    mock_engine_cls.assert_called_once_with(chunks, fake_embeddings, fake_model)
    assert agent is backend._state["agent"]


def test_get_agent_loads_saved_index_when_available(monkeypatch):
    chunks = [{"chunk_id": "1", "content": "x"}]
    fake_embeddings = MagicMock()

    with patch.object(backend, "index_exists", return_value=True), \
         patch.object(backend, "load_index", return_value=(chunks, fake_embeddings)) as mock_load, \
         patch.object(backend, "_ingest_all") as mock_ingest_all, \
         patch.object(backend, "RepoSearchEngine") as mock_engine_cls:
        backend.get_agent()

    mock_load.assert_called_once()
    mock_ingest_all.assert_not_called()  # no need to re-ingest, index was already on disk


def test_get_agent_caches_across_calls():
    fake_agent = MagicMock()
    backend._state["agent"] = fake_agent

    with patch.object(backend, "index_exists") as mock_index_exists:
        result = backend.get_agent()

    mock_index_exists.assert_not_called()  # already cached, must not re-check disk at all
    assert result is fake_agent


def test_rebuild_endpoint_replaces_cached_agent():
    from fastapi.testclient import TestClient

    backend._state["agent"] = MagicMock()  # a stale agent that must get replaced
    new_chunks = [{"chunk_id": "1", "content": "x"}]
    fake_embeddings = MagicMock()

    with patch.object(backend, "_ingest_all", return_value=new_chunks), \
         patch.object(backend, "build_vector_index", return_value=(MagicMock(), MagicMock(), fake_embeddings)), \
         patch.object(backend, "save_index") as mock_save, \
         patch.object(backend, "RepoSearchEngine"):
        client = TestClient(backend.app)
        resp = client.post("/rebuild")

    assert resp.status_code == 200
    assert resp.json() == {"status": "rebuilt", "total_chunks": 1}
    mock_save.assert_called_once_with(new_chunks, fake_embeddings)
