"""
ingest.py — walks ROOT_DIR and turns every supported file into a list
of "chunks" (dicts with content + metadata) ready for embedding.

This is your original logic, cleaned up:
- Python files are split per function/class using the AST (so each chunk
  is a complete, meaningful unit instead of an arbitrary slice of text).
- YAML files are kept whole (pipeline configs are usually short).
- Path segments like "transformations/<domain>" or
  "pipelines/<layer>/<domain>" are turned into metadata so you can filter
  search results by layer/domain later.
"""
from pathlib import Path
from typing import List, Dict
import ast
import uuid

import config
from utils import safe_read_file


def extract_metadata(file_path: Path) -> Dict:
    """
    Infer 'layer' and 'domain' from the file's path.

    Examples this is designed for:
      .../transformations/<domain>/...          -> layer=transformations
      .../pipelines/<layer>/<domain>/...         -> layer=<layer>, domain=<domain>
    """
    parts = file_path.parts
    metadata = {"layer": None, "domain": None}

    if "transformations" in parts:
        metadata["layer"] = "transformations"
        idx = parts.index("transformations")
        if len(parts) > idx + 1:
            metadata["domain"] = parts[idx + 1]

    elif "pipelines" in parts:
        idx = parts.index("pipelines")
        if len(parts) > idx + 1:
            metadata["layer"] = parts[idx + 1]  # e.g. staging / curated
        if len(parts) > idx + 2:
            metadata["domain"] = parts[idx + 2]

    return metadata


def split_python_functions(content: str) -> List[Dict]:
    """Split a Python file into one chunk per top-level function/class."""
    chunks = []
    try:
        tree = ast.parse(content)
    except Exception:
        # Not valid Python (or a syntax error) — skip rather than crash ingestion
        return []

    lines = content.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno - 1
            end_line = node.end_lineno
            block = "\n".join(lines[start_line:end_line])
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "chunk_type": "function",
                "chunk_name": node.name,
                "content": block,
            })
    return chunks


def ingest_repo(root_dir: Path) -> List[Dict]:
    """Walk root_dir and return a flat list of chunk dicts."""
    if not root_dir.exists():
        raise FileNotFoundError(
            f"REPO_ROOT does not exist: {root_dir}\n"
            f"Set it in config.py or via the REPO_ROOT environment variable."
        )

    all_chunks: List[Dict] = []

    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue

        content = safe_read_file(file_path)
        if content is None:
            continue

        metadata = extract_metadata(file_path)

        if file_path.suffix.lower() == ".py":
            for chunk in split_python_functions(content):
                all_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "file_path": str(file_path),
                    "file_name": file_path.stem,
                    "file_type": "python",
                    "layer": metadata["layer"],
                    "domain": metadata["domain"],
                    "chunk_type": chunk["chunk_type"],
                    "chunk_name": chunk["chunk_name"],
                    "content": chunk["content"],
                })
        else:  # .yaml / .yml
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "file_path": str(file_path),
                "file_name": file_path.stem,
                "file_type": "yaml",
                "layer": metadata["layer"],
                "domain": metadata["domain"],
                "chunk_type": "pipeline_config",
                "chunk_name": file_path.stem,
                "content": content,
            })

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_repo(config.ROOT_DIR)
    print(f"Ingested {len(chunks)} chunks from {config.ROOT_DIR}")