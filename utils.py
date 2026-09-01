"""
utils.py — small helpers shared across modules, defined once here so
ingest.py and context.py don't each keep their own copy.
"""
from pathlib import Path
from typing import Optional, Union
import sys


def safe_read_file(file_path: Union[str, Path]) -> Optional[str]:
    """Read a file as UTF-8 text, returning None if that fails for any reason."""
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except Exception:
        return None


def safe_print(*args, **kwargs) -> None:
    """print() that can't crash on Windows' cp1252 console encoding.

    context.py/llm.py print raw retrieved content and LLM output as debug
    diagnostics (by design — see CLAUDE.md). That content can contain
    characters (em dashes, curly quotes, arrows — common from Confluence
    pages) that aren't representable in cp1252, which is still the default
    stdout encoding in a plain Windows terminal. A crash in a debug print
    must never take down the actual answer; unrepresentable characters are
    replaced instead.
    """
    text = " ".join(str(a) for a in args)
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding), **kwargs)