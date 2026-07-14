"""
context.py — turns retrieved chunks into the text block sent to the LLM.

This mirrors your working notebook's get_context(): instead of sending
only the small AST-split chunk, it reads the FULL file each matched
chunk came from (deduplicated, so a file isn't repeated if 2+ of its
chunks were retrieved). That gives the LLM the complete picture —
imports, sibling functions, overall flow — instead of one isolated
function, which is a big part of why the notebook version gave more
complete answers than sending bare chunks did.
"""
from pathlib import Path
from typing import List, Dict

from utils import safe_read_file


def build_context(chunks: List[Dict]) -> str:
    """Read the full source file for each retrieved chunk, deduped by file path."""

    seen_files = set()
    parts = []

    print("\n" + "=" * 80)
    print("BUILDING CONTEXT")
    print("=" * 80)

    for i, chunk in enumerate(chunks, start=1):

        print(f"\nChunk {i}")
        print(f"File  : {chunk['file_name']}")
        print(f"Path  : {chunk['file_path']}")
        print(f"Chunk : {chunk['chunk_name']}")

        file_path = chunk["file_path"]

        if file_path in seen_files:
            print("-> Already added. Skipping duplicate.")
            continue

        seen_files.add(file_path)

        content = safe_read_file(Path(file_path))

        if content is None:
            print("-> Could not read file.")
            continue

        print(f"-> Read {len(content)} characters")

        parts.append(
            "==================================================\n"
            f"FILE : {chunk['file_name']}\n"
            f"PATH : {file_path}\n"
            f"TYPE : {chunk.get('file_type', 'unknown')}\n"
            "==================================================\n"
            f"{content}"
        )

    context = "\n".join(parts)

    print("\n" + "=" * 80)
    print("CONTEXT SUMMARY")
    print("=" * 80)
    print(f"Files Added      : {len(seen_files)}")
    print(f"Context Length   : {len(context)} characters")
    print("=" * 80)

    print("\nFirst 2000 characters of context\n")
    print(context[:2000])

    print("\n")
    print("=" * 80)
    print("END OF CONTEXT PREVIEW")
    print("=" * 80)

    return context