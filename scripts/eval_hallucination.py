"""
eval_hallucination.py — hallucination-rate eval. Target: <15% across the
test query set.

Runs each query through the REAL pipeline (needs GROQ_API_KEY — makes real
Groq API calls, both for the answer itself and for a second "judge" LLM
call that checks whether the answer's claims are grounded in the retrieved
source material). Also runs a small fixed set of deliberately
out-of-scope queries (referencing files/tables that do not exist in the
indexed repo — see eval_common.OUT_OF_SCOPE_QUERIES) to confirm the
system's hallucination guard actually triggers ("I don't find this in the
indexed sources") instead of fabricating a plausible-sounding answer.

Usage:
    python scripts/eval_hallucination.py
"""
from pathlib import Path
import argparse
import time

from groq import Groq

# eval_common must be imported first — it puts the repo root on sys.path
# so `import config` / `from context import ...` below can resolve.
from eval_common import (
    DEFAULT_QUERIES_FILE,
    OUT_OF_SCOPE_QUERIES,
    guard_triggered,
    judge_grounding,
    load_agent,
    load_test_queries,
    print_table,
    safe_ask,
)
import config
from context import build_context

HALLUCINATION_TARGET = 0.15


def evaluate(queries_path: Path = DEFAULT_QUERIES_FILE, include_out_of_scope: bool = True):
    if not config.GROQ_API_KEY:
        print("GROQ_API_KEY is not set — this eval needs real Groq API access.")
        return None

    agent = load_agent()
    client = Groq(api_key=config.GROQ_API_KEY)

    real_queries = [{"query": item["query"], "out_of_scope": False} for item in load_test_queries(queries_path)]
    oos_queries = [{"query": q, "out_of_scope": True} for q in OUT_OF_SCOPE_QUERIES] if include_out_of_scope else []
    all_queries = real_queries + oos_queries

    rows = []
    judged = 0
    grounded_count = 0
    guard_failures = []

    for item in all_queries:
        query = item["query"]
        result = safe_ask(agent, query)
        time.sleep(2)  # respect free-tier tokens-per-minute budget across queries

        if result["mode"] != "codebase" or not result["sources"]:
            rows.append([query, "OOS" if item["out_of_scope"] else "", "general (skipped)", "-"])
            continue

        chunks = agent.search_engine.search(result["standalone_query"] or query, top_k=5)
        context_block = build_context(chunks)
        verdict = judge_grounding(client, context_block, result["answer"])

        judged += 1
        is_grounded = verdict.get("grounded") is True
        grounded_count += is_grounded

        if item["out_of_scope"] and not guard_triggered(result["answer"]) and not is_grounded:
            guard_failures.append(query)

        status = "GROUNDED" if is_grounded else "NOT GROUNDED"
        tag = "OOS" if item["out_of_scope"] else ""
        rows.append([query, tag, status, verdict.get("reason", "")[:60]])

    hallucination_rate = 1 - (grounded_count / judged) if judged else 0.0

    print_table(rows, headers=["Query", "Tag", "Verdict", "Judge Reason"])
    print()
    print(f"Hallucination Rate: {judged - grounded_count}/{judged} not grounded "
          f"= {hallucination_rate:.1%}  (target: <{HALLUCINATION_TARGET:.0%})  "
          f"{'PASS' if hallucination_rate < HALLUCINATION_TARGET else 'FAIL'}")

    if include_out_of_scope:
        n_oos = len(oos_queries)
        n_guard_ok = n_oos - len(guard_failures)
        print(f"Out-of-scope guard: {n_guard_ok}/{n_oos} correctly declined/flagged as ungrounded")
        if guard_failures:
            print("  Guard did NOT trigger for:")
            for q in guard_failures:
                print(f"    - {q}")

    return {
        "hallucination_rate": hallucination_rate,
        "judged": judged,
        "grounded": grounded_count,
        "guard_failures": guard_failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_FILE)
    parser.add_argument("--no-out-of-scope", action="store_true")
    args = parser.parse_args()
    evaluate(args.queries, include_out_of_scope=not args.no_out_of_scope)


if __name__ == "__main__":
    main()
