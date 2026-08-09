"""
confluence_ingest.py — fetches pages from a Confluence Cloud space and turns
them into the same chunk-dict shape ingest.py produces for the local repo
(chunk_id, file_path, file_name, file_type, layer, domain, chunk_type,
chunk_name, content), so they can be merged into one FAISS index and
searched/answered identically.

Auth: Confluence Cloud's REST API uses HTTP Basic Auth with your Atlassian
account email + an API token (NOT a bearer header). Credentials are read
from config.py (which reads them from .env) — never hardcoded here.

Pages are cached to disk under CONFLUENCE_CACHE_DIR as plain-text files, one
per page, split into sections at H1/H2 boundaries (the doc equivalent of
ingest.py splitting Python files at function/class boundaries). context.py
re-reads these cached files the same way it re-reads repo source files, so
no changes were needed there.
"""
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin
import re
import uuid

import requests
from bs4 import BeautifulSoup

import config


def _auth():
    return (config.CONFLUENCE_EMAIL, config.CONFLUENCE_API_TOKEN)


def _slugify(text: str) -> str:
    """Turn a page title into a file-name-safe, underscore-separated slug
    so search.py's alias matching (which splits file names on "_") works
    the same way for Confluence pages as it does for repo files."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:80] or "untitled"


def fetch_pages(space_key: str) -> List[Dict]:
    """Fetch every page in a Confluence space, with its body storage HTML."""
    if not config.CONFLUENCE_ENABLED:
        raise RuntimeError(
            "Confluence is not configured. Set CONFLUENCE_BASE_URL, "
            "CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN in .env."
        )

    pages: List[Dict] = []
    url = f"{config.CONFLUENCE_BASE_URL}/rest/api/content"
    params = {"spaceKey": space_key, "expand": "body.storage,version", "limit": 50}

    while url:
        try:
            resp = requests.get(url, params=params, auth=_auth(), timeout=30)
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach Confluence at {url}: {e}") from e

        if resp.status_code == 401:
            raise RuntimeError(
                "Confluence rejected the credentials (401). Check "
                "CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN in .env."
            )
        resp.raise_for_status()

        data = resp.json()
        pages.extend(data.get("results", []))

        # `_links.next` is relative to `_links.base` (which already includes
        # the "/wiki" context path) — NOT relative to the request URL, so a
        # plain urljoin(resp.url, next_link) silently drops "/wiki".
        links = data.get("_links", {})
        next_link = links.get("next")
        base = links.get("base", config.CONFLUENCE_BASE_URL)
        url = urljoin(base + "/", next_link.lstrip("/")) if next_link else None
        params = None  # next_link already carries the full query string

    return pages


# Many Confluence pages (tables, panels, macros) have no h1/h2 at all, and
# even a "section" under one heading can run long — either way, one giant
# chunk dilutes the embedding across too many unrelated sentences and loses
# to smaller, focused chunks in semantic search. Cap chunk size the way
# ingest.py's function-level split naturally caps Python chunks.
MAX_CHUNK_CHARS = 1500


def _split_large_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Greedily group paragraphs into <= max_chars pieces, splitting a single
    oversized paragraph on its own if needed. Never cuts mid-paragraph otherwise."""
    paragraphs = [p for p in text.split("\n") if p.strip()]
    pieces: List[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(para) <= max_chars:
            current = para
        else:
            for i in range(0, len(para), max_chars):
                pieces.append(para[i:i + max_chars])
            current = ""

    if current:
        pieces.append(current)
    return pieces or [text]


def _split_by_headings(html: str) -> List[Dict]:
    """Split a page's storage-format HTML into sections at h1/h2 boundaries,
    then cap each section (or the whole page, if it has no headings) to
    MAX_CHUNK_CHARS-sized pieces.

    Mirrors ingest.py splitting Python files at function/class boundaries:
    each heading becomes one retrievable chunk instead of one giant blob.
    Pages with no headings fall back to size-based splitting of the whole
    page (same idea as ingest.py keeping short YAML files whole — but a
    long, heading-less Confluence page is not "short").
    """
    soup = BeautifulSoup(html, "html.parser")
    headings = soup.find_all(["h1", "h2"])

    raw_sections: List[Dict] = []
    if not headings:
        text = soup.get_text(separator="\n", strip=True)
        if text:
            raw_sections.append({"heading": None, "text": text})
    else:
        for heading in headings:
            heading_text = heading.get_text(strip=True)
            parts = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ("h1", "h2"):
                    break
                parts.append(sibling.get_text(separator="\n", strip=True))
            body = "\n".join(p for p in parts if p)
            text = f"{heading_text}\n{body}".strip()
            if text:
                raw_sections.append({"heading": heading_text, "text": text})

    sections = []
    for section in raw_sections:
        pieces = _split_large_text(section["text"])
        for i, piece in enumerate(pieces, start=1):
            sections.append({
                "heading": section["heading"],
                "text": piece,
                "part": i,
                "total_parts": len(pieces),
            })

    return sections


def ingest_confluence(space_key: Optional[str] = None) -> List[Dict]:
    """Fetch a Confluence space and return chunk dicts matching ingest.py's shape."""
    space_key = space_key or config.CONFLUENCE_SPACE_KEY
    if not space_key:
        raise RuntimeError("CONFLUENCE_SPACE_KEY is not set in .env.")

    cache_dir = config.CONFLUENCE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict] = []

    for page in fetch_pages(space_key):
        title = page.get("title", "untitled")
        page_id = page.get("id")
        html = page.get("body", {}).get("storage", {}).get("value", "")
        page_url = f"{config.CONFLUENCE_BASE_URL}/spaces/{space_key}/pages/{page_id}"

        sections = _split_by_headings(html)
        if not sections:
            continue

        slug = _slugify(title)
        cache_file = cache_dir / f"{page_id}_{slug}.txt"
        full_text = "\n\n".join(s["text"] for s in sections)
        cache_file.write_text(
            f"Title: {title}\nURL: {page_url}\n\n{full_text}", encoding="utf-8"
        )

        for section in sections:
            chunk_name = section["heading"] or title
            if section["total_parts"] > 1:
                chunk_name += f" (part {section['part']}/{section['total_parts']})"
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "file_path": str(cache_file),
                "file_name": slug,
                "file_type": "confluence",
                "layer": "confluence",
                "domain": space_key,
                "chunk_type": "confluence_section" if section["heading"] else "confluence_page",
                "chunk_name": chunk_name,
                "content": section["text"],
                "source_url": page_url,
            })

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_confluence()
    print(f"Ingested {len(chunks)} chunks from Confluence space {config.CONFLUENCE_SPACE_KEY}")
