"""End-to-end tests against the FastAPI app via TestClient (no live server).

backend.get_agent() is seeded directly with a RepoRAGAgent built from a
small fixture corpus + FakeEmbedder, so importing/using `backend.app` here
never touches config.ROOT_DIR, the real embedding model, or the live
Confluence API (see backend.py's lazy get_agent() refactor). The Groq LLM
boundary is mocked for every test except the one marked @pytest.mark.live.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import backend
import config
import llm
import router
from embed_index import build_vector_index
from rag_agent import RepoRAGAgent
from search import RepoSearchEngine


class _FakeGroqLike:
    def __init__(self, response_text):
        self.chat = self
        self.completions = self
        self._response_text = response_text

    def create(self, model, messages, temperature=0, **kwargs):
        msg = SimpleNamespace(content=self._response_text)
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


@pytest.fixture
def api_client(fake_embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(config, "PROMPT_LOG_FILE", tmp_path / "prompt.txt")

    repo_file = tmp_path / "givex.py"
    repo_file.write_text("def derive_brand(row):\n    return row['brand_code']\n", encoding="utf-8")
    confluence_file = tmp_path / "order.txt"
    confluence_file.write_text("Title: Order\n\nPrimary Keys: store_id, check_number", encoding="utf-8")

    chunks = [
        {
            "chunk_id": "1", "file_path": str(repo_file), "file_name": "givex",
            "file_type": "python", "layer": "transformations", "domain": "givex",
            "chunk_type": "function", "chunk_name": "derive_brand",
            "content": "def derive_brand(row):\n    return row['brand_code']",
        },
        {
            "chunk_id": "2", "file_path": str(confluence_file), "file_name": "order",
            "file_type": "confluence", "layer": "confluence", "domain": "DPT",
            "chunk_type": "confluence_section", "chunk_name": "Primary Keys",
            "content": "Primary Keys: store_id, check_number",
            "source_url": "https://example.atlassian.net/wiki/spaces/DPT/pages/1",
        },
    ]
    _, _, embeddings = build_vector_index(chunks, embedder=fake_embedder)
    engine = RepoSearchEngine(chunks, embeddings, fake_embedder)
    backend._state["agent"] = RepoRAGAgent(engine)

    yield TestClient(backend.app), chunks

    backend._state["agent"] = None  # don't leak state into other tests


def test_health_endpoint(api_client):
    client, _ = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_stats_endpoint_reports_mixed_sources(api_client):
    client, _ = api_client
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 2
    assert "confluence" in data["layers"]


def test_chat_empty_query_returns_400(api_client):
    client, _ = api_client
    resp = client.post("/chat", json={"query": "   "})
    assert resp.status_code == 400


# ---------------------------------------------------------------------
# Scenario 1: general / chit-chat query — no retrieval, no sources
# ---------------------------------------------------------------------
def test_e2e_general_query_scenario(api_client):
    client, _ = api_client
    router_resp = _FakeGroqLike('{"mode": "general", "standalone_query": ""}')
    llm_resp = _FakeGroqLike("Hello! Ask me anything about the codebase or docs.")

    with patch.object(router, "Groq", return_value=router_resp), \
         patch.object(llm, "Groq", return_value=llm_resp):
        resp = client.post("/chat", json={"query": "hi there"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "general"
    assert data["sources"] == []


# ---------------------------------------------------------------------
# Scenario 2: repo-code query — grounded answer with a real citation
# ---------------------------------------------------------------------
def test_e2e_repo_code_query_scenario_has_valid_citation(api_client):
    client, chunks = api_client
    router_resp = _FakeGroqLike(
        '{"mode": "codebase", "standalone_query": "how is brand derived in givex"}'
    )
    llm_resp = _FakeGroqLike("brand_code is returned directly. [givex.py]")

    with patch.object(router, "Groq", return_value=router_resp), \
         patch.object(llm, "Groq", return_value=llm_resp):
        resp = client.post("/chat", json={"query": "how is brand derived in the givex transform?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "codebase"
    assert len(data["sources"]) >= 1

    real_paths = {c["file_path"] for c in chunks}
    assert any(s["file_path"] in real_paths for s in data["sources"])
    assert any(s["file_name"] == "givex" for s in data["sources"])


# ---------------------------------------------------------------------
# Scenario 3: Confluence-docs query — grounded answer with a doc citation
# ---------------------------------------------------------------------
def test_e2e_confluence_docs_query_scenario_has_valid_citation(api_client):
    client, chunks = api_client
    router_resp = _FakeGroqLike(
        '{"mode": "codebase", "standalone_query": "what are the primary keys on the order page"}'
    )
    llm_resp = _FakeGroqLike("The primary keys are store_id and check_number. [order]")

    with patch.object(router, "Groq", return_value=router_resp), \
         patch.object(llm, "Groq", return_value=llm_resp):
        resp = client.post("/chat", json={"query": "what are the primary keys for the order confluence page?"})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sources"]) >= 1

    real_paths = {c["file_path"] for c in chunks}
    assert any(s["file_path"] in real_paths for s in data["sources"])
    confluence_sources = [s for s in data["sources"] if s.get("source_url")]
    assert confluence_sources, "expected at least one cited source with a Confluence source_url"


def test_e2e_conversation_history_round_trip(api_client):
    """A second request carrying history from the first must still work
    through the real FastAPI request/response schema (ChatMessage list)."""
    client, _ = api_client
    router_resp = _FakeGroqLike('{"mode": "codebase", "standalone_query": "what does derive_brand return"}')
    llm_resp = _FakeGroqLike("It returns row['brand_code']. [givex.py]")

    with patch.object(router, "Groq", return_value=router_resp), \
         patch.object(llm, "Groq", return_value=llm_resp):
        resp = client.post("/chat", json={
            "query": "what does it return",
            "history": [
                {"role": "user", "content": "tell me about derive_brand in givex"},
                {"role": "assistant", "content": "it derives the brand from the row"},
            ],
        })

    assert resp.status_code == 200
    assert resp.json()["mode"] == "codebase"


# ---------------------------------------------------------------------
# Query-category scenarios (RFP requirement: how-to / troubleshooting /
# architecture / policy — see scripts/test_queries.json's "category" field
# for the real-content-backed versions of these, evaluated against the live
# index by scripts/run_evaluation_harness.py). These 4 tests use small
# synthetic fixture content instead of the live Confluence cache — same
# design reason as `api_client` above: fast, deterministic, works
# identically in CI or locally, with no dependency on index_store/ or
# confluence_cache/ existing on disk.
#
# A code chunk ("givex") is deliberately included in every one of these
# fixtures alongside the category-appropriate chunk, and every test asserts
# it does NOT appear in the cited sources — this is the concrete check for
# "a code file bleeding into a policy/how-to/etc. question" that these
# tests exist to catch.
# ---------------------------------------------------------------------
@pytest.fixture
def category_client(fake_embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(config, "PROMPT_LOG_FILE", tmp_path / "prompt.txt")

    code_file = tmp_path / "givex.py"
    code_file.write_text("def derive_brand(row):\n    return row['brand_code']\n", encoding="utf-8")

    howto_file = tmp_path / "axe_setup.txt"
    howto_file.write_text(
        "Title: AXE Local Setup\n\nStep 1: install Python 3.10+. Step 2: install Java 11. "
        "Step 3: clone the repo and run pip install -r requirements.txt.",
        encoding="utf-8",
    )
    trouble_file = tmp_path / "airflow_issues.txt"
    trouble_file.write_text(
        "Title: Airflow Issue Log\n\nIssue: DAG parsing error caused by a missing function "
        "argument. Resolution: added the missing run_state argument to the DAG callable.",
        encoding="utf-8",
    )
    arch_file = tmp_path / "transaction_domain.txt"
    arch_file.write_text(
        "Title: Transaction Domain Design\n\nTransaction data is sourced from EDW for most "
        "brands, except Jamba, which sources directly from the Toast POS system.",
        encoding="utf-8",
    )
    policy_file = tmp_path / "delta_standards.txt"
    policy_file.write_text(
        "Title: Delta Table Design Standards\n\nTable names must not be plural. Table names "
        "must follow the format domain_name_table_name. Every table must have a "
        "dl_created_datetime column.",
        encoding="utf-8",
    )

    chunks = [
        {
            "chunk_id": "code", "file_path": str(code_file), "file_name": "givex",
            "file_type": "python", "layer": "transformations", "domain": "givex",
            "chunk_type": "function", "chunk_name": "derive_brand",
            "content": "def derive_brand(row):\n    return row['brand_code']",
        },
        {
            "chunk_id": "howto", "file_path": str(howto_file), "file_name": "axe_setup",
            "file_type": "confluence", "layer": "confluence", "domain": "DPT",
            "chunk_type": "confluence_section", "chunk_name": "Setup Steps",
            "content": "Step 1: install Python 3.10+. Step 2: install Java 11. "
                       "Step 3: clone the repo and run pip install -r requirements.txt.",
            "source_url": "https://example.atlassian.net/wiki/spaces/DPT/pages/1",
        },
        {
            "chunk_id": "trouble", "file_path": str(trouble_file), "file_name": "airflow_issues",
            "file_type": "confluence", "layer": "confluence", "domain": "DPT",
            "chunk_type": "confluence_section", "chunk_name": "Issue Log",
            "content": "Issue: DAG parsing error caused by a missing function argument. "
                       "Resolution: added the missing run_state argument to the DAG callable.",
            "source_url": "https://example.atlassian.net/wiki/spaces/DPT/pages/2",
        },
        {
            "chunk_id": "arch", "file_path": str(arch_file), "file_name": "transaction_domain",
            "file_type": "confluence", "layer": "confluence", "domain": "DPT",
            "chunk_type": "confluence_section", "chunk_name": "Domain Design",
            "content": "Transaction data is sourced from EDW for most brands, except Jamba, "
                       "which sources directly from the Toast POS system.",
            "source_url": "https://example.atlassian.net/wiki/spaces/DPT/pages/3",
        },
        {
            "chunk_id": "policy", "file_path": str(policy_file), "file_name": "delta_standards",
            "file_type": "confluence", "layer": "confluence", "domain": "DPT",
            "chunk_type": "confluence_section", "chunk_name": "Design Standards",
            "content": "Table names must not be plural. Table names must follow the format "
                       "domain_name_table_name. Every table must have a dl_created_datetime column.",
            "source_url": "https://example.atlassian.net/wiki/spaces/DPT/pages/4",
        },
    ]
    _, _, embeddings = build_vector_index(chunks, embedder=fake_embedder)
    engine = RepoSearchEngine(chunks, embeddings, fake_embedder)
    backend._state["agent"] = RepoRAGAgent(engine)

    yield TestClient(backend.app), chunks

    backend._state["agent"] = None


def test_e2e_how_to_category_scenario(category_client):
    client, _ = category_client
    router_resp = _FakeGroqLike('{"mode": "codebase", "standalone_query": "how do I set up AXE locally"}')
    llm_resp = _FakeGroqLike(
        "Install Python 3.10+, then Java 11, then run pip install -r requirements.txt. [axe_setup]"
    )

    with patch.object(router, "Groq", return_value=router_resp), \
         patch.object(llm, "Groq", return_value=llm_resp):
        resp = client.post("/chat", json={"query": "How do I set up AXE locally for development?", "top_k": 1})

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "codebase"
    assert len(data["sources"]) >= 1
    assert any(s["file_name"] == "axe_setup" for s in data["sources"]), \
        "expected the how-to source to be cited"
    assert not any(s["file_name"] == "givex" for s in data["sources"]), \
        "a code file must not bleed into a how-to answer"


def test_e2e_troubleshooting_category_scenario(category_client):
    client, _ = category_client
    router_resp = _FakeGroqLike('{"mode": "codebase", "standalone_query": "what caused the airflow DAG parsing error"}')
    llm_resp = _FakeGroqLike(
        "A missing run_state argument caused the DAG parsing error; it was fixed by adding it. [airflow_issues]"
    )

    with patch.object(router, "Groq", return_value=router_resp), \
         patch.object(llm, "Groq", return_value=llm_resp):
        resp = client.post("/chat", json={
            "query": "What caused the Airflow DAG parsing error and how was it resolved?", "top_k": 1
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "codebase"
    assert any(s["file_name"] == "airflow_issues" for s in data["sources"]), \
        "expected the troubleshooting/issue-log source to be cited"
    assert not any(s["file_name"] == "givex" for s in data["sources"]), \
        "a code file must not bleed into a troubleshooting answer"


def test_e2e_architecture_category_scenario(category_client):
    client, _ = category_client
    router_resp = _FakeGroqLike('{"mode": "codebase", "standalone_query": "how is transaction data sourced across brands"}')
    llm_resp = _FakeGroqLike(
        "Transaction data comes from EDW for most brands, except Jamba, which sources "
        "directly from Toast POS. [transaction_domain]"
    )

    with patch.object(router, "Groq", return_value=router_resp), \
         patch.object(llm, "Groq", return_value=llm_resp):
        resp = client.post("/chat", json={
            "query": "How is transaction data sourced and designed across different brands?", "top_k": 1
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "codebase"
    assert any(s["file_name"] == "transaction_domain" for s in data["sources"]), \
        "expected the architecture/domain-design source to be cited"
    assert not any(s["file_name"] == "givex" for s in data["sources"]), \
        "a code file must not bleed into an architecture answer"


def test_e2e_policy_category_scenario(category_client):
    client, _ = category_client
    router_resp = _FakeGroqLike('{"mode": "codebase", "standalone_query": "what are the delta table naming standards"}')
    llm_resp = _FakeGroqLike(
        "Table names must be singular, follow domain_name_table_name, and include "
        "dl_created_datetime. [delta_standards]"
    )

    with patch.object(router, "Groq", return_value=router_resp), \
         patch.object(llm, "Groq", return_value=llm_resp):
        resp = client.post("/chat", json={
            "query": "What are the required design standards for naming Delta tables?", "top_k": 1
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "codebase"
    assert any(s["file_name"] == "delta_standards" for s in data["sources"]), \
        "expected the policy/design-standards source to be cited"
    assert not any(s["file_name"] == "givex" for s in data["sources"]), \
        "a code file must not bleed into a policy answer"


# ---------------------------------------------------------------------
# Live: real Groq API call. Excluded from CI (see .github/workflows/tests.yml
# and pyproject.toml's marker config); run manually with:
#   pytest -m live tests/e2e/test_api_e2e.py
# ---------------------------------------------------------------------
@pytest.mark.live
def test_e2e_live_groq_end_to_end_has_real_citation(api_client):
    import os
    real_key = os.getenv("GROQ_API_KEY")
    if not real_key:
        pytest.skip("GROQ_API_KEY not set — skipping live Groq test")

    client, chunks = api_client
    with patch.object(config, "GROQ_API_KEY", real_key):
        resp = client.post("/chat", json={"query": "How is brand derived in the givex transform?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert len(data["sources"]) >= 1, "expected the real LLM's grounded answer to come with citations"

    real_paths = {c["file_path"] for c in chunks}
    assert any(s["file_path"] in real_paths for s in data["sources"]), (
        "citation must reference a file_path that was actually ingested"
    )
