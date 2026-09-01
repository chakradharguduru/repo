"""Unit tests for context.py — build_context() re-reads full source files
for each retrieved chunk, deduped by file_path."""
from context import build_context


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_build_context_reads_full_file_content(tmp_path):
    f = _write(tmp_path / "a.py", "def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    chunk = {
        "file_path": str(f), "file_name": "a", "file_type": "python", "chunk_name": "foo",
    }

    context = build_context([chunk])

    assert "def foo" in context
    assert "def bar" in context  # full file, not just the isolated chunk
    assert "FILE : a" in context
    assert "TYPE : python" in context


def test_build_context_dedupes_by_file_path(tmp_path):
    f = _write(tmp_path / "a.py", "content-marker-xyz")
    chunk1 = {"file_path": str(f), "file_name": "a", "file_type": "python", "chunk_name": "foo"}
    chunk2 = {"file_path": str(f), "file_name": "a", "file_type": "python", "chunk_name": "bar"}

    context = build_context([chunk1, chunk2])

    assert context.count("content-marker-xyz") == 1


def test_build_context_multiple_distinct_files(tmp_path):
    f1 = _write(tmp_path / "a.py", "alpha content")
    f2 = _write(tmp_path / "b.yaml", "beta content")
    chunks = [
        {"file_path": str(f1), "file_name": "a", "file_type": "python", "chunk_name": "x"},
        {"file_path": str(f2), "file_name": "b", "file_type": "yaml", "chunk_name": "y"},
    ]

    context = build_context(chunks)
    assert "alpha content" in context
    assert "beta content" in context


def test_build_context_includes_source_url_for_confluence_chunks(tmp_path):
    f = _write(tmp_path / "page.txt", "Title: Order\n\nsome page text")
    chunk = {
        "file_path": str(f), "file_name": "order", "file_type": "confluence",
        "chunk_name": "Order", "source_url": "https://example.atlassian.net/wiki/spaces/DPT/pages/1",
    }

    context = build_context([chunk])
    assert "URL  : https://example.atlassian.net/wiki/spaces/DPT/pages/1" in context


def test_build_context_no_url_line_for_repo_chunks(tmp_path):
    f = _write(tmp_path / "a.py", "content")
    chunk = {"file_path": str(f), "file_name": "a", "file_type": "python", "chunk_name": "x"}

    context = build_context([chunk])
    assert "URL  :" not in context


def test_build_context_skips_unreadable_file(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.py")
    chunk = {"file_path": missing_path, "file_name": "missing", "file_type": "python", "chunk_name": "x"}

    context = build_context([chunk])
    assert context == ""


def test_build_context_empty_chunk_list_returns_empty_string():
    assert build_context([]) == ""
