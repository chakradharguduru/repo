"""Integration test: router -> retrieval -> prompt construction, exercised
through RepoRAGAgent.ask() — the real orchestration in rag_agent.py. Only
the LLM boundary (router's classification call, the final answer call) is
mocked; search.py/context.py run for real over a small fixture corpus.

Specifically covers conversation-history injection: a follow-up query must
be rewritten into a self-contained standalone_query using prior turns
*before* retrieval runs, so retrieval sees the rewritten text, not the
raw ambiguous follow-up.
"""
from types import SimpleNamespace
from unittest.mock import patch

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
        self.calls = []

    def create(self, model, messages, temperature=0, **kwargs):
        self.calls.append({"model": model, "messages": messages})
        msg = SimpleNamespace(content=self._response_text)
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


def _agent_over(chunks, fake_embedder):
    _, _, embeddings = build_vector_index(chunks, embedder=fake_embedder)
    engine = RepoSearchEngine(chunks, embeddings, fake_embedder)
    return RepoRAGAgent(engine)


def test_general_mode_skips_retrieval_entirely(fake_embedder, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    agent = _agent_over([{
        "chunk_id": "1", "file_path": "a.py", "file_name": "a", "file_type": "python",
        "layer": None, "domain": None, "chunk_type": "function", "chunk_name": "f",
        "content": "def f(): pass",
    }], fake_embedder)

    router_response = '{"mode": "general", "standalone_query": ""}'
    with patch.object(router, "Groq", return_value=_FakeGroqLike(router_response)), \
         patch.object(llm, "Groq", return_value=_FakeGroqLike("Hi! How can I help?")):
        result = agent.ask("hello there")

    assert result["mode"] == "general"
    assert result["sources"] == []
    assert result["answer"] == "Hi! How can I help?"


def test_followup_query_rewritten_before_retrieval_runs(tmp_path, fake_embedder, monkeypatch):
    """Core conversation-history-injection check: retrieval must run on the
    router's REWRITTEN standalone_query, not the raw ambiguous follow-up."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")

    f = tmp_path / "survey_summary.py"
    f.write_text("def write_table():\n    return 'survey_results'\n", encoding="utf-8")
    chunks = [{
        "chunk_id": "1", "file_path": str(f), "file_name": "survey_summary",
        "file_type": "python", "layer": "transformations", "domain": "survey",
        "chunk_type": "function", "chunk_name": "write_table",
        "content": "def write_table():\n    return 'survey_results'",
    }]
    agent = _agent_over(chunks, fake_embedder)

    rewritten_query = "what table does survey_summary write its data into"
    router_response = f'{{"mode": "codebase", "standalone_query": "{rewritten_query}"}}'
    router_fake = _FakeGroqLike(router_response)
    llm_fake = _FakeGroqLike("It writes to survey_results. [survey_summary.py]")

    history = [
        {"role": "user", "content": "what transformations happen in survey summary"},
        {"role": "assistant", "content": "it aggregates rows and writes a table"},
    ]

    with patch.object(router, "Groq", return_value=router_fake), \
         patch.object(llm, "Groq", return_value=llm_fake), \
         patch.object(RepoSearchEngine, "search", wraps=agent.search_engine.search) as spy_search:
        result = agent.ask("what is the table name", history=history)

    # retrieval must have been called with the REWRITTEN query, not the raw one
    spy_search.assert_called_once()
    called_query = spy_search.call_args.args[0] if spy_search.call_args.args else spy_search.call_args.kwargs["query"]
    assert called_query == rewritten_query

    assert result["mode"] == "codebase"
    assert result["standalone_query"] == rewritten_query
    assert len(result["sources"]) == 1
    assert result["sources"][0]["file_name"] == "survey_summary"

    # the FINAL answer call must use the ORIGINAL wording, per rag_agent.py's
    # documented design (retrieve on standalone, answer on original phrasing)
    final_prompt = llm_fake.calls[0]["messages"][-1]["content"]
    assert "what is the table name" in final_prompt


def test_codebase_mode_prompt_is_grounded_in_retrieved_chunk(tmp_path, fake_embedder, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(config, "PROMPT_LOG_FILE", tmp_path / "prompt.txt")

    f = tmp_path / "givex.py"
    f.write_text("def derive_brand(row):\n    return row['brand_code']\n", encoding="utf-8")
    chunks = [{
        "chunk_id": "1", "file_path": str(f), "file_name": "givex", "file_type": "python",
        "layer": "transformations", "domain": "givex", "chunk_type": "function",
        "chunk_name": "derive_brand", "content": "def derive_brand(row):\n    return row['brand_code']",
    }]
    agent = _agent_over(chunks, fake_embedder)

    router_response = '{"mode": "codebase", "standalone_query": "how is brand derived in givex"}'
    with patch.object(router, "Groq", return_value=_FakeGroqLike(router_response)), \
         patch.object(llm, "Groq", return_value=_FakeGroqLike("brand_code is used. [givex.py]")):
        result = agent.ask("how is brand derived in the givex transform?")

    assert result["mode"] == "codebase"
    assert result["answer"] == "brand_code is used. [givex.py]"
    assert result["sources"][0]["file_name"] == "givex"
    assert config.PROMPT_LOG_FILE.exists()
    assert "derive_brand" in config.PROMPT_LOG_FILE.read_text(encoding="utf-8")
