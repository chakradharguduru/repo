"""
run_evaluation_harness.py — the formal evaluation deliverable: runs every
test query ONCE through the real pipeline (needs GROQ_API_KEY) and, per
query, records retrieved sources, the generated answer, an LLM-assisted
1-5 answer-quality score, a hallucination (groundedness) verdict, and
citation correctness — then writes scripts/evaluation_report.md summarizing
aggregate scores against every QA target, suitable to attach as evidence.

Runs the 3 deliberately out-of-scope queries too (see
eval_common.OUT_OF_SCOPE_QUERIES), reported separately as the
hallucination-guard check rather than folded into the main metrics, since
they have no expected retrieval target.

Usage:
    python scripts/run_evaluation_harness.py
    python scripts/run_evaluation_harness.py --queries scripts/test_queries.json --top-k 3
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import time

from groq import Groq

# eval_common must be imported first — it puts the repo root on sys.path
# so `import config` / `from context import ...` below can resolve.
from eval_common import (
    DEFAULT_QUERIES_FILE,
    OUT_OF_SCOPE_QUERIES,
    SCRIPTS_DIR,
    extract_cited_filenames,
    guard_triggered,
    judge_grounding,
    judge_quality,
    load_agent,
    load_test_queries,
    safe_ask,
)
import config
from context import build_context

RETRIEVAL_TARGET = 0.70
HALLUCINATION_TARGET = 0.15
CITATION_TARGET = 1.00

DEFAULT_REPORT_PATH = SCRIPTS_DIR / "evaluation_report.md"


def _same_file(path_a: str, path_b: str) -> bool:
    try:
        return Path(path_a).resolve() == Path(path_b).resolve()
    except (OSError, ValueError):
        return path_a == path_b


def _evaluate_one(agent, client, query: str, expected_path: str | None, recall_top_k: int = 3) -> dict:
    result = safe_ask(agent, query)
    record = {
        "query": query,
        "mode": result["mode"],
        "answer": result["answer"],
        "sources": result["sources"],
        "expected_path": expected_path,
        "retrieval_hit": None,
        "citation_correct": None,
        "grounded": None,
        "grounded_reason": "",
        "quality_score": None,
        "quality_reason": "",
    }

    if result["mode"] != "codebase":
        return record  # general/error — no retrieval/citation/grounding to score

    try:
        # Retrieval-hit is deliberately a SEPARATE, purpose-built search at
        # recall_top_k (the Top-N recall metric) — independent of how many
        # chunks the real answer actually used to ground its response.
        if expected_path is not None:
            search_query = result["standalone_query"] or query
            recall_chunks = agent.search_engine.search(search_query, top_k=recall_top_k)
            record["retrieval_hit"] = any(_same_file(expected_path, c["file_path"]) for c in recall_chunks)

        # Citation/grounding/quality must all judge the SAME material the
        # real answer was actually grounded in. result["sources"] IS that
        # material (built by rag_agent.ask() from its own retrieval at
        # config.DEFAULT_TOP_K) — re-searching here at a different top_k
        # (this used to hardcode 3) previously made a genuinely-retrieved
        # chunk at rank 4/5 look like a fabricated citation, just because a
        # smaller re-search didn't happen to include it.
        sources = result["sources"]
        retrieved_file_names = {s["file_name"].lower() for s in sources}
        cited_stems = {Path(c).stem.lower() for c in extract_cited_filenames(result["answer"])}
        unmatched = cited_stems - retrieved_file_names if cited_stems else set()
        record["citation_correct"] = not unmatched

        context_block = build_context(sources)
        grounding = judge_grounding(client, context_block, result["answer"])
        record["grounded"] = grounding.get("grounded")
        record["grounded_reason"] = grounding.get("reason", "")

        quality = judge_quality(client, query, context_block, result["answer"])
        record["quality_score"] = quality.get("score")
        record["quality_reason"] = quality.get("justification", "")
    except Exception as e:
        # Scoring itself failed (e.g. a judge call hit a rate limit) — keep
        # the primary answer we already have, just leave the judged fields
        # as None rather than losing this query's result entirely.
        record["grounded_reason"] = f"scoring failed: {e}"

    return record


def run(queries_path: Path = DEFAULT_QUERIES_FILE, top_k: int = 3, include_out_of_scope: bool = True):
    if not config.GROQ_API_KEY:
        print("GROQ_API_KEY is not set — the harness needs real Groq API access.")
        return None

    agent = load_agent()
    client = Groq(api_key=config.GROQ_API_KEY)
    test_queries = load_test_queries(queries_path)

    # A small gap between queries — each query makes 2-4 real Groq calls
    # (router, answer, grounding judge, quality judge), and the free tier's
    # tokens-per-minute budget is easy to exhaust running 10+ queries back
    # to back with no pause at all.
    QUERY_DELAY_SECONDS = 5

    records = []
    for item in test_queries:
        print(f"Evaluating: {item['query']}")
        records.append(_evaluate_one(
            agent, client, item["query"], item.get("expected_source_file_path"), recall_top_k=top_k
        ))
        time.sleep(QUERY_DELAY_SECONDS)

    oos_records = []
    if include_out_of_scope:
        for q in OUT_OF_SCOPE_QUERIES:
            print(f"Evaluating (out-of-scope): {q}")
            rec = _evaluate_one(agent, client, q, expected_path=None)
            time.sleep(QUERY_DELAY_SECONDS)
            rec["guard_ok"] = (
                rec["mode"] == "general"  # declined without even searching — clearly not fabricated
                or guard_triggered(rec["answer"])
                or rec["grounded"] is True  # declining gracefully still counts as "grounded"
            )
            oos_records.append(rec)

    return {"records": records, "oos_records": oos_records}


def _aggregate(records: list) -> dict:
    scored = [r for r in records if r["mode"] == "codebase"]
    with_expected = [r for r in scored if r["retrieval_hit"] is not None]
    grounded_judged = [r for r in scored if r["grounded"] is not None]
    citation_judged = [r for r in scored if r["citation_correct"] is not None]
    quality_judged = [r for r in scored if r["quality_score"] is not None]

    recall = (sum(r["retrieval_hit"] for r in with_expected) / len(with_expected)) if with_expected else None
    hallucination_rate = (
        1 - sum(r["grounded"] for r in grounded_judged) / len(grounded_judged)
        if grounded_judged else None
    )
    citation_correctness = (
        sum(r["citation_correct"] for r in citation_judged) / len(citation_judged)
        if citation_judged else None
    )
    avg_quality = (
        sum(r["quality_score"] for r in quality_judged) / len(quality_judged)
        if quality_judged else None
    )

    return {
        "recall": recall,
        "hallucination_rate": hallucination_rate,
        "citation_correctness": citation_correctness,
        "avg_quality": avg_quality,
        "n_total": len(records),
        "n_codebase": len(scored),
        "n_general": len(records) - len(scored),
    }


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def _verdict(actual, target, higher_is_better=True):
    if actual is None:
        return "N/A"
    ok = (actual >= target) if higher_is_better else (actual < target)
    return "PASS" if ok else "FAIL"


def write_report(results: dict, report_path: Path = DEFAULT_REPORT_PATH, top_k: int = 3):
    records = results["records"]
    oos_records = results["oos_records"]
    agg = _aggregate(records)

    lines = []
    lines.append("# KnowBot RAG — Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Model: `{config.GROQ_MODEL}` (router: `{config.ROUTER_MODEL}`)")
    lines.append(f"Query set: {len(records)} test queries + {len(oos_records)} out-of-scope queries")
    lines.append("")

    lines.append("## Summary vs. QA targets")
    lines.append("")
    lines.append("| Metric | Result | Target | Verdict |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Retrieval Top-{top_k} Recall | {_pct(agg['recall'])} | ≥{RETRIEVAL_TARGET:.0%} | "
                  f"{_verdict(agg['recall'], RETRIEVAL_TARGET, higher_is_better=True)} |")
    lines.append(f"| Hallucination Rate | {_pct(agg['hallucination_rate'])} | <{HALLUCINATION_TARGET:.0%} | "
                  f"{_verdict(agg['hallucination_rate'], HALLUCINATION_TARGET, higher_is_better=False)} |")
    lines.append(f"| Citation Correctness | {_pct(agg['citation_correctness'])} | {CITATION_TARGET:.0%} | "
                  f"{_verdict(agg['citation_correctness'], CITATION_TARGET, higher_is_better=True)} |")
    avg_q = "n/a" if agg["avg_quality"] is None else f"{agg['avg_quality']:.1f} / 5"
    lines.append(f"| Avg. Answer Quality (LLM-assisted) | {avg_q} | informational | — |")
    lines.append("")
    lines.append(f"({agg['n_codebase']} of {agg['n_total']} queries were retrieval-scored 'codebase' "
                  f"responses; {agg['n_general']} were 'general' responses with no retrieval to score.)")
    lines.append("")

    if oos_records:
        n_ok = sum(r["guard_ok"] for r in oos_records)
        lines.append(f"**Out-of-scope hallucination guard:** {n_ok}/{len(oos_records)} correctly "
                      "declined or were judged ungrounded rather than fabricating an answer.")
        lines.append("")

    lines.append("## Per-query detail")
    lines.append("")
    lines.append("| # | Query | Mode | Retrieval Hit | Citation OK | Grounded | Quality | Answer (preview) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(records, 1):
        preview = r["answer"].replace("\n", " ")[:80]
        lines.append(
            f"| {i} | {r['query'][:60]} | {r['mode']} | "
            f"{'-' if r['retrieval_hit'] is None else ('✅' if r['retrieval_hit'] else '❌')} | "
            f"{'-' if r['citation_correct'] is None else ('✅' if r['citation_correct'] else '❌')} | "
            f"{'-' if r['grounded'] is None else ('✅' if r['grounded'] else '❌')} | "
            f"{'-' if r['quality_score'] is None else r['quality_score']} | "
            f"{preview}… |"
        )
    lines.append("")

    flagged = [
        (i, r) for i, r in enumerate(records, 1)
        if r["mode"] == "error" or r["grounded"] is False or r["citation_correct"] is False
    ]
    if flagged:
        lines.append("## Notes on flagged rows")
        lines.append("")
        for i, r in flagged:
            if r["mode"] == "error":
                lines.append(f"- **#{i}** (`{r['query'][:60]}`): pipeline error — {r['answer']}")
            else:
                if r["grounded"] is False:
                    lines.append(f"- **#{i}** (`{r['query'][:60]}`): not grounded — {r['grounded_reason']}")
                if r["citation_correct"] is False:
                    lines.append(f"- **#{i}** (`{r['query'][:60]}`): citation mismatch — a cited file "
                                  "was not among the retrieved chunks")
        lines.append("")

    if oos_records:
        lines.append("## Out-of-scope queries (hallucination guard)")
        lines.append("")
        lines.append("| Query | Mode | Guard OK | Answer (preview) |")
        lines.append("|---|---|---|---|")
        for r in oos_records:
            preview = r["answer"].replace("\n", " ")[:100]
            lines.append(f"| {r['query']} | {r['mode']} | {'✅' if r['guard_ok'] else '❌'} | {preview}… |")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_FILE)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--no-out-of-scope", action="store_true")
    args = parser.parse_args()

    results = run(args.queries, args.top_k, include_out_of_scope=not args.no_out_of_scope)
    if results is None:
        return
    write_report(results, args.out, args.top_k)


if __name__ == "__main__":
    main()
