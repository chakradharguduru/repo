"""Unit tests for confluence_ingest.py. All network calls (requests.get) are
mocked — no live Confluence API access in this tier."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import config
import confluence_ingest as ci

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "confluence"


# ---------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------
@pytest.mark.parametrize("title,expected", [
    ("Order", "order"),
    ("List of Data Sources", "list_of_data_sources"),
    ("Transaction Domain", "transaction_domain"),
    ("", "untitled"),
    ("!!!", "untitled"),
])
def test_slugify(title, expected):
    assert ci._slugify(title) == expected


def test_slugify_uses_underscores_not_hyphens():
    """search.py's alias matching splits file names on '_' — slugify must
    match that convention, not hyphenate, or Confluence titles never get
    the same fragment-matching repo file names get."""
    slug = ci._slugify("Customer And Transaction Bridge")
    assert "-" not in slug
    assert slug == "customer_and_transaction_bridge"


# ---------------------------------------------------------------------
# _split_large_text
# ---------------------------------------------------------------------
def test_split_large_text_under_limit_returns_single_piece():
    text = "short paragraph"
    pieces = ci._split_large_text(text, max_chars=1500)
    assert pieces == [text]


def test_split_large_text_splits_on_paragraph_boundaries():
    paragraphs = [f"Paragraph {i} " + ("x" * 100) for i in range(20)]
    text = "\n".join(paragraphs)
    pieces = ci._split_large_text(text, max_chars=300)

    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 300
    # nothing lost
    assert sum(p.count("Paragraph") for p in pieces) == 20


def test_split_large_text_hard_splits_a_single_oversized_paragraph():
    huge_paragraph = "x" * 5000  # one "paragraph", no newlines to split on
    pieces = ci._split_large_text(huge_paragraph, max_chars=1500)

    assert len(pieces) == 4  # ceil(5000/1500)
    assert "".join(pieces) == huge_paragraph
    for piece in pieces[:-1]:
        assert len(piece) == 1500


# ---------------------------------------------------------------------
# _split_by_headings
# ---------------------------------------------------------------------
def test_split_by_headings_with_headings():
    html = (FIXTURES_DIR / "page_with_headings.html").read_text()
    sections = ci._split_by_headings(html)

    headings = [s["heading"] for s in sections]
    assert "Overview" in headings
    assert "Data Sources" in headings
    assert "Primary Keys" in headings

    data_sources = next(s for s in sections if s["heading"] == "Data Sources")
    assert "Punchh API" in data_sources["text"]
    assert data_sources["total_parts"] == 1


def test_split_by_headings_no_headings_falls_back_to_size_split():
    html = (FIXTURES_DIR / "page_no_headings.html").read_text()
    sections = ci._split_by_headings(html)

    assert len(sections) > 1  # fixture is intentionally longer than MAX_CHUNK_CHARS
    for s in sections:
        assert s["heading"] is None
        assert len(s["text"]) <= ci.MAX_CHUNK_CHARS
        assert s["total_parts"] == len(sections)
    # content is preserved across the pieces
    assert any("Primary Keys" in s["text"] for s in sections)
    assert any("Qu API" in s["text"] for s in sections)


def test_split_by_headings_empty_html_returns_no_sections():
    assert ci._split_by_headings("") == []
    assert ci._split_by_headings("<p></p>") == []


# ---------------------------------------------------------------------
# fetch_pages — mocked HTTP
# ---------------------------------------------------------------------
def _mock_response(json_data, status_code=200, url="https://example.atlassian.net/wiki/rest/api/content"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.url = url
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture(autouse=True)
def confluence_configured(monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_BASE_URL", "https://example.atlassian.net/wiki")
    monkeypatch.setattr(config, "CONFLUENCE_EMAIL", "bot@example.com")
    monkeypatch.setattr(config, "CONFLUENCE_API_TOKEN", "fake-token")
    monkeypatch.setattr(config, "CONFLUENCE_SPACE_KEY", "DPT")
    monkeypatch.setattr(config, "CONFLUENCE_ENABLED", True)
    monkeypatch.setattr(ci.config, "CONFLUENCE_ENABLED", True, raising=False)


def test_fetch_pages_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_ENABLED", False)
    with pytest.raises(RuntimeError, match="not configured"):
        ci.fetch_pages("DPT")


def test_fetch_pages_single_page_no_pagination():
    page = {"id": "1", "title": "Order", "body": {"storage": {"value": "<p>hi</p>"}}}
    response = _mock_response({"results": [page], "_links": {}})

    with patch.object(ci.requests, "get", return_value=response) as mock_get:
        pages = ci.fetch_pages("DPT")

    assert pages == [page]
    mock_get.assert_called_once()


def test_fetch_pages_follows_pagination_via_links_base_and_next():
    page1 = {"id": "1", "title": "Page One"}
    page2 = {"id": "2", "title": "Page Two"}
    first = _mock_response({
        "results": [page1],
        "_links": {"base": "https://example.atlassian.net/wiki", "next": "/rest/api/content?start=50"},
    })
    second = _mock_response({"results": [page2], "_links": {}})

    with patch.object(ci.requests, "get", side_effect=[first, second]) as mock_get:
        pages = ci.fetch_pages("DPT")

    assert pages == [page1, page2]
    assert mock_get.call_count == 2
    # the "/wiki" context path must survive into the second request's URL
    second_call_url = mock_get.call_args_list[1].args[0]
    assert "/wiki/rest/api/content" in second_call_url


def test_fetch_pages_401_raises_friendly_error():
    response = _mock_response({}, status_code=401)
    with patch.object(ci.requests, "get", return_value=response):
        with pytest.raises(RuntimeError, match="401"):
            ci.fetch_pages("DPT")


def test_fetch_pages_network_error_wrapped_in_runtime_error():
    with patch.object(ci.requests, "get", side_effect=requests.ConnectionError("boom")):
        with pytest.raises(RuntimeError, match="Could not reach Confluence"):
            ci.fetch_pages("DPT")


# ---------------------------------------------------------------------
# ingest_confluence — end to end with mocked fetch_pages
# ---------------------------------------------------------------------
def test_ingest_confluence_produces_chunks_matching_ingest_py_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_CACHE_DIR", tmp_path / "confluence_cache")
    html = (FIXTURES_DIR / "page_with_headings.html").read_text()
    page = {"id": "999", "title": "Loyalty", "body": {"storage": {"value": html}}}

    with patch.object(ci, "fetch_pages", return_value=[page]):
        chunks = ci.ingest_confluence("DPT")

    assert len(chunks) == 3  # Overview / Data Sources / Primary Keys
    for c in chunks:
        assert set(c.keys()) >= {
            "chunk_id", "file_path", "file_name", "file_type", "layer",
            "domain", "chunk_type", "chunk_name", "content", "source_url",
        }
        assert c["file_type"] == "confluence"
        assert c["layer"] == "confluence"
        assert c["domain"] == "DPT"
        assert c["source_url"] == "https://example.atlassian.net/wiki/spaces/DPT/pages/999"
        assert Path(c["file_path"]).exists()  # cache file was actually written


def test_ingest_confluence_caches_full_page_text_once_per_page(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_CACHE_DIR", tmp_path / "confluence_cache")
    html = (FIXTURES_DIR / "page_with_headings.html").read_text()
    page = {"id": "999", "title": "Loyalty", "body": {"storage": {"value": html}}}

    with patch.object(ci, "fetch_pages", return_value=[page]):
        chunks = ci.ingest_confluence("DPT")

    file_paths = {c["file_path"] for c in chunks}
    assert len(file_paths) == 1  # all 3 chunks share the same cached page file

    cached_text = Path(list(file_paths)[0]).read_text(encoding="utf-8")
    assert "Title: Loyalty" in cached_text
    assert "Punchh API" in cached_text


def test_ingest_confluence_labels_multi_part_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_CACHE_DIR", tmp_path / "confluence_cache")
    html = (FIXTURES_DIR / "page_no_headings.html").read_text()
    page = {"id": "123", "title": "Order", "body": {"storage": {"value": html}}}

    with patch.object(ci, "fetch_pages", return_value=[page]):
        chunks = ci.ingest_confluence("DPT")

    assert len(chunks) > 1
    assert all("part" in c["chunk_name"] for c in chunks)
    assert all(c["file_name"] == "order" for c in chunks)


def test_ingest_confluence_no_space_key_raises(monkeypatch):
    monkeypatch.setattr(config, "CONFLUENCE_SPACE_KEY", "")
    with pytest.raises(RuntimeError, match="CONFLUENCE_SPACE_KEY"):
        ci.ingest_confluence()
