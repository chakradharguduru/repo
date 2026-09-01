"""
eval_citations.py — citation correctness eval. Target: 100%.

For every test query, runs the REAL pipeline (router -> retrieval -> LLM,
via rag_agent.RepoRAGAgent — real Groq API calls, needs GROQ_API_KEY) and
checks two things for each "codebase"-mode (cited) response:

1. Structural: every entry in the response's `sources` list must be one of
   the chunks that was actually retrieved for that query (a sanity check on
   rag_agent.py's own guarantee — sources are built directly from the
   retrieved chunk list, so a failure here means the pipeline itself is
   broken, not just an LLM mistake).
2. Prose: file names the LLM's free-text ANSWER appears to cite (e.g.
   "[givex.py]", "based on order.txt") must also appear among the
   retrieved chunks' file names — this catches the LLM inventing a
   plausible-sounding filename that was never actually retrieved, which
   the structural check alone cannot catch.

"general"-mode responses (no citations expected) are skipped, not counted
as failures.

Usage:
    python scripts/eval_citations.py
"""
from pathlib import Path
import argparse
import time

from eval_common import (
    DEFAULT_QUERIES_FILE,
    extract_cited_filenames,
    load_agent,
    load_test_queries,
    print_table,
    safe_ask,
)

CORRECTNESS_TARGET = 1.00


def evaluate(queries_path: Path = DEFAULT_QUERIES_FILE):
    agent = load_agent()
    queries = load_test_queries(queries_path)

    rows = []
    checked = 0
    correct = 0
    mismatches = []

    for item in queries:
        query = item["query"]
        result = safe_ask(agent, query)
        time.sleep(2)  # respect free-tier tokens-per-minute budget across queries

        if result["mode"] != "codebase" or not result["sources"]:
            rows.append([query, "SKIPPED (general/no sources)", "-"])
            continue

        checked += 1
        retrieved_file_names = {s["file_name"].lower() for s in result["sources"]}
        retrieved_file_paths = {s["file_path"] for s in result["sources"]}

        # (1) structural check — sources must be a subset of what was retrieved
        # for this query (re-run retrieval to get the ground-truth chunk set).
        # NOTE: `or bool(retrieved_file_paths)` used to be tacked onto this —
        # since retrieved_file_paths is always non-empty by this point, that
        # made the check trivially True and silently dead. Removed.
        retrieved_chunks = agent.search_engine.search(result["standalone_query"] or query, top_k=10)
        retrieved_chunk_paths = {c["file_path"] for c in retrieved_chunks}
        structural_ok = retrieved_file_paths.issubset(retrieved_chunk_paths)

        # (2) prose check — filenames mentioned in the answer text must be
        # among the retrieved chunk file names
        cited_in_prose = extract_cited_filenames(result["answer"])
        cited_file_stems = {Path(c).stem.lower() for c in cited_in_prose}
        unmatched = cited_file_stems - retrieved_file_names if cited_file_stems else set()

        ok = structural_ok and not unmatched
        correct += ok
        status = "PASS" if ok else "FAIL"
        if not ok:
            mismatches.append({
                "query": query,
                "cited_but_not_retrieved": sorted(unmatched),
                "retrieved_file_names": sorted(retrieved_file_names),
            })

        rows.append([query, status, ", ".join(sorted(retrieved_file_names)) or "(none)"])

    correctness = correct / checked if checked else 1.0

    print_table(rows, headers=["Query", "Result", "Retrieved File Names"])
    print()
    if mismatches:
        print("Flagged mismatches:")
        for m in mismatches:
            print(f"  - \"{m['query']}\"")
            print(f"      cited but not retrieved: {m['cited_but_not_retrieved']}")
            print(f"      actually retrieved: {m['retrieved_file_names']}")
        print()

    print(f"Citation Correctness: {correct}/{checked} = {correctness:.1%} "
          f"(target: {CORRECTNESS_TARGET:.0%})  "
          f"{'PASS' if correctness >= CORRECTNESS_TARGET else 'FAIL'}  "
          f"[{len(queries) - checked} skipped: general/no-citation responses]")

    return {"correctness": correctness, "checked": checked, "total": len(queries), "mismatches": mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_FILE)
    args = parser.parse_args()
    evaluate(args.queries)


if __name__ == "__main__":
    main()
