"""
main.py — run this file.

    python main.py            # uses saved index if it exists, else builds one
    python main.py --rebuild  # forces a fresh ingest + re-embed

Then type questions at the prompt. Ctrl+C or "exit" to quit.
"""
import sys

import config
from ingest import ingest_repo
from confluence_ingest import ingest_confluence
from embed_index import build_vector_index, save_index, load_index, index_exists
from search import RepoSearchEngine
from rag_agent import RepoRAGAgent


def build_fresh_index():
    print(f"Ingesting repo at {config.ROOT_DIR} ...")
    chunks = ingest_repo(config.ROOT_DIR)
    print(f"Got {len(chunks)} repo chunks.")

    if config.CONFLUENCE_ENABLED:
        print(f"Ingesting Confluence space {config.CONFLUENCE_SPACE_KEY} ...")
        try:
            confluence_chunks = ingest_confluence()
            print(f"Got {len(confluence_chunks)} Confluence chunks.")
            chunks.extend(confluence_chunks)
        except Exception as e:
            print(f"Confluence ingestion failed, continuing with repo-only index: {e}")
    else:
        print("Confluence not configured (CONFLUENCE_BASE_URL/EMAIL/API_TOKEN) — skipping.")

    if not chunks:
        print("No chunks were created. Check ROOT_DIR and SUPPORTED_EXTENSIONS in config.py.")
        sys.exit(1)

    model, _index, embeddings = build_vector_index(chunks)
    save_index(chunks, embeddings)
    return chunks, embeddings, model


def main():
    rebuild = "--rebuild" in sys.argv

    if rebuild or not index_exists():
        chunks, embeddings, model = build_fresh_index()
    else:
        print("Loading saved index...")
        chunks, embeddings = load_index()
        model = None  # RepoSearchEngine will load the embedding model itself

    engine = RepoSearchEngine(chunks, embeddings, model)
    agent = RepoRAGAgent(engine)

    print(f"\nReady. Indexed {len(chunks)} chunks. Ask a question (or 'exit').\n")

    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        result = agent.ask(query)
        print("\n" + result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            print(f"  - {s['file_path']}  ({s['chunk_name']})")
        print()


if __name__ == "__main__":
    main()