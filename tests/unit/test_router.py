"""Unit tests for router.py — route_query(). The Groq client is mocked
throughout; no live API calls in this tier."""
from types import SimpleNamespace
from unittest.mock import patch

import config
import router
from router import _render_history, route_query


def _patched_groq(response_text):
    return patch.object(router, "Groq", return_value=_FakeGroqLike(response_text))


class _FakeGroqLike:
    """Mimics groq.Groq(...).chat.completions.create(...).choices[0].message.content"""
    def __init__(self, response_text):
        self.chat = self
        self.completions = self
        self._response_text = response_text
        self.calls = []

    def create(self, model, messages, temperature=0):
        self.calls.append({"model": model, "messages": messages})
        msg = SimpleNamespace(content=self._response_text)
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


def test_route_query_no_api_key_defaults_to_codebase(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    result = route_query("what does this file do")
    assert result == {"mode": "codebase", "standalone_query": "what does this file do"}


def test_route_query_general_mode(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    with _patched_groq('{"mode": "general", "standalone_query": ""}'):
        result = route_query("hi there")
    assert result == {"mode": "general", "standalone_query": ""}


def test_route_query_codebase_mode_with_rewrite(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    response = '{"mode": "codebase", "standalone_query": "what table does survey_summary write to"}'
    with _patched_groq(response):
        result = route_query("what is the table name", history=[
            {"role": "user", "content": "what transformations happen in survey summary"},
            {"role": "assistant", "content": "it aggregates rows"},
        ])
    assert result["mode"] == "codebase"
    assert result["standalone_query"] == "what table does survey_summary write to"


def test_route_query_strips_markdown_fences(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    response = '```json\n{"mode": "general", "standalone_query": ""}\n```'
    with _patched_groq(response):
        result = route_query("thanks")
    assert result["mode"] == "general"


def test_route_query_malformed_json_falls_back_to_codebase(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    with _patched_groq("not valid json at all"):
        result = route_query("what does givex.py do")
    assert result == {"mode": "codebase", "standalone_query": "what does givex.py do"}


def test_route_query_invalid_mode_value_defaults_to_codebase(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    with _patched_groq('{"mode": "something_else", "standalone_query": ""}'):
        result = route_query("some query")
    assert result["mode"] == "codebase"


def test_route_query_codebase_empty_standalone_falls_back_to_raw_query(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    with _patched_groq('{"mode": "codebase", "standalone_query": ""}'):
        result = route_query("what table name")
    assert result["standalone_query"] == "what table name"


def test_route_query_client_exception_falls_back_to_codebase(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    with patch.object(router, "Groq", side_effect=RuntimeError("network down")):
        result = route_query("what does this do")
    assert result == {"mode": "codebase", "standalone_query": "what does this do"}


# ---------------------------------------------------------------------
# _render_history — supports both dicts and object-like history entries
# ---------------------------------------------------------------------
def test_render_history_empty():
    assert _render_history(None) == "(none)"
    assert _render_history([]) == "(none)"


def test_render_history_with_dicts():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    rendered = _render_history(history)
    assert "USER: hi" in rendered
    assert "ASSISTANT: hello" in rendered


def test_render_history_with_objects():
    history = [SimpleNamespace(role="user", content="hi")]
    rendered = _render_history(history)
    assert "USER: hi" in rendered


def test_render_history_truncates_to_last_8():
    history = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
    rendered = _render_history(history)
    assert "msg19" in rendered
    assert "msg0" not in rendered
