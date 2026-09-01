"""Unit tests for utils.py — safe_read_file() and safe_print()."""
import io
import sys

from utils import safe_print, safe_read_file


def test_safe_read_file_reads_existing_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    assert safe_read_file(f) == "hello"


def test_safe_read_file_missing_file_returns_none(tmp_path):
    assert safe_read_file(tmp_path / "missing.txt") is None


def test_safe_print_normal_text(capsys):
    safe_print("hello world")
    assert capsys.readouterr().out.strip() == "hello world"


def test_safe_print_survives_unencodable_characters(monkeypatch):
    """Regression test: context.py's debug dump of retrieved content
    crashed with UnicodeEncodeError on Windows' cp1252 console when the
    content had characters (em dash, arrows) cp1252 can't represent —
    safe_print must degrade gracefully instead of raising."""

    class Cp1252Stream(io.TextIOWrapper):
        def __init__(self):
            super().__init__(io.BytesIO(), encoding="cp1252", errors="strict")

    fake_stdout = Cp1252Stream()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    # Should not raise, despite "—" and "→" not being representable in cp1252.
    safe_print("unicode test — em dash and arrow →")
