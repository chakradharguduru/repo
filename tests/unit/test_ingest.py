"""Unit tests for ingest.py — AST-based Python chunking, YAML whole-file
chunking, path-based layer/domain metadata extraction."""
from pathlib import Path

import pytest

from ingest import extract_metadata, ingest_repo, split_python_functions


# ---------------------------------------------------------------------
# split_python_functions — AST chunking
# ---------------------------------------------------------------------
def test_split_python_functions_counts_top_level_defs():
    content = (Path(__file__).parent.parent / "fixtures" / "sample_repo"
               / "transformations" / "sales" / "sales_summary.py").read_text()

    chunks = split_python_functions(content)

    # 2 top-level functions + 1 class = 3 chunks
    assert len(chunks) == 3
    names = {c["chunk_name"] for c in chunks}
    assert names == {"compute_total", "compute_average", "SalesSummary"}


def test_split_python_functions_chunk_shape():
    content = "def foo():\n    return 1\n"
    chunks = split_python_functions(content)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert set(chunk.keys()) == {"chunk_id", "chunk_type", "chunk_name", "content"}
    assert chunk["chunk_name"] == "foo"
    assert chunk["content"] == "def foo():\n    return 1"
    assert chunk["chunk_id"]  # non-empty uuid string


def test_split_python_functions_class_chunk_type_matches_function():
    """Documents actual behavior: chunk_type is hard-coded to "function" for
    both FunctionDef and ClassDef nodes — there is no separate "class" type."""
    content = "class Foo:\n    pass\n"
    chunks = split_python_functions(content)

    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "function"
    assert chunks[0]["chunk_name"] == "Foo"


def test_split_python_functions_ignores_module_level_code():
    content = "x = 1\nprint(x)\n\ndef only_this():\n    pass\n"
    chunks = split_python_functions(content)

    assert len(chunks) == 1
    assert chunks[0]["chunk_name"] == "only_this"


def test_split_python_functions_invalid_syntax_returns_empty_list():
    content = "def broken(:\n    return 1\n"
    assert split_python_functions(content) == []


def test_split_python_functions_empty_file_returns_empty_list():
    assert split_python_functions("") == []


# ---------------------------------------------------------------------
# extract_metadata — layer/domain from path segments
# ---------------------------------------------------------------------
def test_extract_metadata_transformations_path():
    path = Path("/repo/transformations/sales/sales_summary.py")
    meta = extract_metadata(path)
    assert meta == {"layer": "transformations", "domain": "sales"}


def test_extract_metadata_pipelines_path():
    path = Path("/repo/pipelines/staging/sales/sales_pipeline.yaml")
    meta = extract_metadata(path)
    assert meta == {"layer": "staging", "domain": "sales"}


def test_extract_metadata_unmatched_path_returns_none_fields():
    path = Path("/repo/misc/notes.py")
    meta = extract_metadata(path)
    assert meta == {"layer": None, "domain": None}


def test_extract_metadata_transformations_without_domain_segment():
    path = Path("/repo/transformations/only_file.py")
    meta = extract_metadata(path)
    assert meta["layer"] == "transformations"
    assert meta["domain"] == "only_file.py"  # next path segment, even if it's the filename


# ---------------------------------------------------------------------
# ingest_repo — full walk over a fixture directory
# ---------------------------------------------------------------------
def test_ingest_repo_over_fixture_tree(sample_repo_dir):
    chunks = ingest_repo(sample_repo_dir)

    # sales_summary.py -> 3 chunks, sales_pipeline.yaml -> 1 chunk.
    # broken_syntax.py -> 0 (invalid syntax skipped), notes.txt -> skipped (unsupported ext).
    assert len(chunks) == 4

    file_types = {c["file_type"] for c in chunks}
    assert file_types == {"python", "yaml"}

    python_chunks = [c for c in chunks if c["file_type"] == "python"]
    assert {c["chunk_name"] for c in python_chunks} == {
        "compute_total", "compute_average", "SalesSummary"
    }
    for c in python_chunks:
        assert c["layer"] == "transformations"
        assert c["domain"] == "sales"
        assert c["file_name"] == "sales_summary"
        assert Path(c["file_path"]).exists()

    yaml_chunks = [c for c in chunks if c["file_type"] == "yaml"]
    assert len(yaml_chunks) == 1
    yaml_chunk = yaml_chunks[0]
    assert yaml_chunk["layer"] == "staging"
    assert yaml_chunk["domain"] == "sales"
    assert yaml_chunk["chunk_type"] == "pipeline_config"
    assert yaml_chunk["chunk_name"] == "sales_pipeline"
    assert "pipeline_name: sales_daily" in yaml_chunk["content"]


def test_ingest_repo_chunk_ids_are_unique(sample_repo_dir):
    chunks = ingest_repo(sample_repo_dir)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_ingest_repo_missing_root_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        ingest_repo(missing)


def test_ingest_repo_skips_unsupported_extensions(sample_repo_dir):
    chunks = ingest_repo(sample_repo_dir)
    assert not any(c["file_name"] == "notes" for c in chunks)


def test_ingest_repo_skips_files_with_syntax_errors(sample_repo_dir):
    chunks = ingest_repo(sample_repo_dir)
    assert not any("broken" in c["file_path"] for c in chunks)
