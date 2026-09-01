"""
generate_test_queries.py — one-off helper that samples real chunks from the
currently saved index (index_store/) and writes scripts/test_queries.json:
placeholder {query, expected_source_file_path} pairs for the eval scripts.

These are naive, template-generated queries drawn from real chunk/file
names — meant to be reviewed and rewritten with real user-style questions
before being trusted as an actual accuracy benchmark (see eval_retrieval.py's
docstring). Re-run any time to refresh the sample.

Usage: python scripts/generate_test_queries.py [--count N]
"""
from pathlib import Path
import argparse
import json
import random

from eval_common import DEFAULT_QUERIES_FILE, load_index


def _query_for_chunk(chunk: dict) -> str:
    file_type = chunk.get("file_type")
    chunk_name = chunk.get("chunk_name") or chunk.get("file_name")
    file_name = chunk.get("file_name")

    if file_type == "python":
        return f"What does {chunk_name} do in {file_name}?"
    if file_type == "yaml":
        return f"What does the {file_name} pipeline configure?"
    if file_type == "confluence":
        if chunk.get("chunk_type") == "confluence_section" and chunk_name != file_name:
            return f"What does the {file_name} Confluence page say about {chunk_name}?"
        return f"What is documented on the {file_name} Confluence page?"
    return f"Tell me about {chunk_name}."


def generate(count: int, seed: int = 42) -> list:
    chunks, _ = load_index()
    by_type: dict = {}
    for c in chunks:
        by_type.setdefault(c.get("file_type"), []).append(c)

    rng = random.Random(seed)
    types_present = [t for t in ("python", "yaml", "confluence") if by_type.get(t)]
    if not types_present:
        raise RuntimeError("No chunks found in the saved index — nothing to sample from.")

    per_type = max(1, count // len(types_present))
    sampled = []
    for t in types_present:
        pool = by_type[t]
        rng.shuffle(pool)
        sampled.extend(pool[:per_type])

    # top up to `count` from whatever's left, in case a source type was thin
    remaining_pool = [c for t in types_present for c in by_type[t] if c not in sampled]
    rng.shuffle(remaining_pool)
    while len(sampled) < count and remaining_pool:
        sampled.append(remaining_pool.pop())

    queries = []
    for chunk in sampled[:count]:
        queries.append({
            "query": _query_for_chunk(chunk),
            "expected_source_file_path": chunk["file_path"],
            "expected_chunk_name": chunk.get("chunk_name"),
            "source_type": chunk.get("file_type"),
        })
    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12, help="number of placeholder queries to generate")
    parser.add_argument("--out", type=Path, default=DEFAULT_QUERIES_FILE)
    args = parser.parse_args()

    queries = generate(args.count)
    args.out.write_text(json.dumps(queries, indent=2), encoding="utf-8")
    print(f"Wrote {len(queries)} placeholder queries to {args.out}")
    print("These are naive, auto-generated from real chunk content — review and")
    print("replace with real user-style questions before trusting the recall %.")


if __name__ == "__main__":
    main()
