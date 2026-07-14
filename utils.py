"""
utils.py — small helpers shared across modules, defined once here so
ingest.py and context.py don't each keep their own copy.
"""
from pathlib import Path
from typing import Optional, Union


def safe_read_file(file_path: Union[str, Path]) -> Optional[str]:
    """Read a file as UTF-8 text, returning None if that fails for any reason."""
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except Exception:
        return None