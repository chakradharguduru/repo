"""
eval_retrieval.py — retrieval accuracy eval. Target: >=70% Top-3 recall.

Runs each query in scripts/test_queries.json through the REAL retrieval
pipeline (the saved index in index_store/ + search.RepoSearchEngine — no
LLM calls, no Groq API key needed) and checks whether the expected source
file appears anywhere in the Top-3 results.

Input file format (see scripts/generate_test_queries.py for how to make one):
    [{"query": "...", "expected_source_file_path": "..."}, ...]
Extra fields (expected_chunk_name, source_type) are ignored if present.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --queries scripts/test_queries.json --top-k 3
"""
from pathlib import Path
import argparse

from eval_common import DEFAULT_QUERIES_FILE, load_engine, load_test_queries, print_table

RECALL_TARGET = 0.70


def _same_file(path_a: str, path_b: str) -> bool:
    """Compare by resolved path where possible, falling back to raw string
    equality — the saved index may contain paths from a machine/drive
    layout that doesn't match exactly but still refers to the same file."""
    try:
        return Path(path_a).resolve() == Path(path_b).resolve()
    except (OSError, ValueError):
        return path_a == path_b


def evaluate(queries_path: Path = DEFAULT_QUERIES_FILE, top_k: int = 3):
    engine = load_engine()
    queries = load_test_queries(queries_path)

    rows = []
    passed = 0
    for item in queries:
        query = item["query"]
        expected = item["expected_source_file_path"]

        results = engine.search(query, top_k=top_k)
        retrieved_paths = [r["file_path"] for r in results]
        hit = any(_same_file(expected, p) for p in retrieved_paths)
        passed += hit

        rows.append([
            query,
            Path(expected).name,
            "PASS" if hit else "FAIL",
            ", ".join(Path(p).name for p in retrieved_paths) or "(none)",
        ])

    recall = passed / len(queries) if queries else 0.0

    print_table(rows, headers=["Query", "Expected File", "Result", f"Top-{top_k} Retrieved Files"])
    print()
    print(f"Top-{top_k} Recall: {passed}/{len(queries)} = {recall:.1%}  "
          f"(target: >={RECALL_TARGET:.0%})  "
          f"{'PASS' if recall >= RECALL_TARGET else 'FAIL'}")

    return {"recall": recall, "passed": passed, "total": len(queries), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_FILE)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    evaluate(args.queries, args.top_k)


if __name__ == "__main__":
    main()
