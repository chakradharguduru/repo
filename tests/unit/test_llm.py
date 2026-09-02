"""Unit tests for llm.py — generate_answer() and answer_general(). The Groq
client is mocked throughout; no live API calls in this tier."""
from types import SimpleNamespace
from unittest.mock import patch

import config
import llm


class _FakeGroqLike:
    def __init__(self, response_text):
        self.chat = self
        self.completions = self
        self._response_text = response_text
        self.calls = []

    def create(self, model, messages, temperature=0, **kwargs):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        msg = SimpleNamespace(content=self._response_text)
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


def _patched_groq(response_text="a test answer"):
    return patch.object(llm, "Groq", return_value=_FakeGroqLike(response_text))


def test_get_client_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    try:
        llm._get_client()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "GROQ_API_KEY" in str(e)


def test_answer_general_returns_llm_content(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    with _patched_groq("2 + 2 is 4."):
        answer = llm.answer_general("what is 2+2")
    assert answer == "2 + 2 is 4."


def test_answer_general_includes_history_in_messages(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    fake = _FakeGroqLike("ok")
    with patch.object(llm, "Groq", return_value=fake):
        llm.answer_general("follow up question", history=[
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ])
    roles = [m["role"] for m in fake.calls[0]["messages"]]
    assert "user" in roles and "assistant" in roles


def test_generate_answer_no_chunks_short_circuits(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    answer = llm.generate_answer("what is this", chunks=[])
    assert "couldn't find" in answer.lower()


def test_generate_answer_grounded_in_context(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(config, "PROMPT_LOG_FILE", tmp_path / "prompt.txt")

    f = tmp_path / "a.py"
    f.write_text("def compute_total(rows):\n    return sum(rows)\n", encoding="utf-8")
    chunks = [{"file_path": str(f), "file_name": "a", "file_type": "python", "chunk_name": "compute_total"}]

    fake = _FakeGroqLike("compute_total sums the rows. [a.py]")
    with patch.object(llm, "Groq", return_value=fake):
        answer = llm.generate_answer("what does compute_total do", chunks)

    assert answer == "compute_total sums the rows. [a.py]"
    # the chunk's file content must actually have been sent to the model
    sent_content = fake.calls[0]["messages"][-1]["content"]
    assert "compute_total sums" not in sent_content  # sanity: not echoing the mocked answer
    assert "def compute_total" in sent_content


def test_generate_answer_writes_prompt_log_to_configured_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    log_path = tmp_path / "custom_prompt.txt"
    monkeypatch.setattr(config, "PROMPT_LOG_FILE", log_path)

    f = tmp_path / "a.py"
    f.write_text("print('hi')", encoding="utf-8")
    chunks = [{"file_path": str(f), "file_name": "a", "file_type": "python", "chunk_name": "x"}]

    with _patched_groq("answer"):
        llm.generate_answer("query", chunks)

    assert log_path.exists()
    assert "query" in log_path.read_text(encoding="utf-8")


def test_generate_answer_unreadable_chunks_returns_apology(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    chunks = [{"file_path": "/nonexistent/path.py", "file_name": "x", "file_type": "python", "chunk_name": "x"}]

    answer = llm.generate_answer("what is this", chunks)
    assert "couldn't read" in answer.lower()
