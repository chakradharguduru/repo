"""
rag_agent.py — the "R" and "G" of RAG in one place, now with a routing
step in front.

Flow:
  1. route_query(query, history) decides general vs codebase, and rewrites
     follow-ups into a standalone search query.
  2. general -> answer directly with the LLM, no retrieval, no sources.
  3. codebase -> retrieve on the STANDALONE query (so follow-ups keep the
     file we were talking about), then generate a grounded answer using the
     user's ORIGINAL wording.
"""
import time
from typing import Dict

from search import RepoSearchEngine
from router import route_query
import llm
import config


class RepoRAGAgent:
    def __init__(self, search_engine: RepoSearchEngine):
        self.search_engine = search_engine

    def ask(self, query: str, history=None, top_k: int = config.DEFAULT_TOP_K) -> Dict:
        start = time.time()
        history = history or []

        # ---------------------------------------------------------
        # 1. Route: general vs codebase (+ history-aware rewrite)
        # ---------------------------------------------------------
        route = route_query(query, history=history)

        if route["mode"] == "general":
            answer = llm.answer_general(query, history=history)
            return {
                "answer": answer,
                "mode": "general",
                "standalone_query": None,
                "elapsed_seconds": round(time.time() - start, 2),
                "sources": [],
            }

        # ---------------------------------------------------------
        # 2. Codebase: retrieve on the standalone (rewritten) query
        # ---------------------------------------------------------
        search_query = route["standalone_query"]
        chunks = self.search_engine.search(search_query, top_k=top_k)

        print("\nRetrieved Chunks")
        print("=" * 80)
        for i, c in enumerate(chunks, 1):
            print(f"{i}. {c['file_name']}")
            print(f"   Path : {c['file_path']}")
            print(f"   Chunk: {c['chunk_name']}")
            print(f"   Size : {len(c['content'])}")

        if config.DEBUG:
            print("\nRetrieved Chunks")
            print("=" * 80)
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {c['file_path']}  ({c['chunk_name']}, {len(c['content'])} chars)")

        # 3. Generate using the ORIGINAL query for natural phrasing,
        #    grounded in the chunks the standalone query retrieved.
        answer = llm.generate_answer(query, chunks, history=history)
        elapsed = round(time.time() - start, 2)

        return {
            "answer": answer,
            "mode": "codebase",
            "standalone_query": search_query,
            "elapsed_seconds": elapsed,
            "sources": [
                {
                    "file_path": c["file_path"],
                    "file_name": c["file_name"],
                    "chunk_name": c["chunk_name"],
                    "layer": c["layer"],
                    "domain": c["domain"],
                    "preview": c["content"][:300],
                }
                for c in chunks
            ],
        }

    def stats(self) -> Dict:
        """Summary info for a health/stats UI panel."""
        return {
            "total_chunks": len(self.search_engine.chunks),
            "layers": self.search_engine.layers,
            "domains": self.search_engine.domains,
            "file_count": len(self.search_engine.file_names),
        }