"""
conftest.py — fixtures shared across unit/integration/e2e tests.

Two heavy externals are stubbed out everywhere except where a test is
specifically exercising them: the SentenceTransformer embedding model
(FakeEmbedder — deterministic, no ~130MB download) and the Groq LLM client
(FakeGroqClient / make_groq_response — no network, no API key needed). Live
tests (marked @pytest.mark.live) opt back into the real Groq API explicitly.
"""
from pathlib import Path
import hashlib
import re
import shutil

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------
# Fake embedder — deterministic, fast, no model download.
#
# Text is embedded by hashing overlapping word shingles into a fixed-size
# vector, so semantically/lexically different strings land in different
# regions of the space (good enough for FAISS nearest-neighbor tests)
# while being 100% reproducible and independent of any ML framework.
# ---------------------------------------------------------------------
class FakeEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        # Word-boundary extraction (not naive .split()) so punctuation glued
        # to a word — a docstring's opening \"\"\"Average, a trailing colon
        # on rows: — doesn't stop it matching the clean word in a query.
        words = re.findall(r"[a-z0-9]+", text.lower())
        if not words:
            return vec
        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        out = np.stack([self._embed_one(t) for t in texts]).astype("float32")
        return out[0] if single else out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


# ---------------------------------------------------------------------
# Fake Groq client — mimics the tiny slice of the SDK the project uses:
# client.chat.completions.create(...).choices[0].message.content
# ---------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeGroqClient:
    """Records every call and returns pre-scripted responses in order,
    or a single fixed response for every call if `responses` is a string."""

    def __init__(self, responses):
        self._responses = responses if isinstance(responses, list) else None
        self._fixed = responses if isinstance(responses, str) else None
        self._call_index = 0
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature=0):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        if self._fixed is not None:
            content = self._fixed
        else:
            content = self._responses[min(self._call_index, len(self._responses) - 1)]
            self._call_index += 1
        return _FakeCompletion(content)


@pytest.fixture
def fake_groq_client():
    return FakeGroqClient("This is a test answer.")


# ---------------------------------------------------------------------
# Sample chunk fixtures (plain dicts matching ingest.py's chunk schema)
# ---------------------------------------------------------------------
@pytest.fixture
def sample_repo_dir(tmp_path):
    """Copy the small fixture repo tree into a fresh tmp_path per test."""
    dest = tmp_path / "sample_repo"
    shutil.copytree(FIXTURES_DIR / "sample_repo", dest)
    return dest


@pytest.fixture
def sample_chunks():
    """A tiny, hand-built chunk list spanning python/yaml/confluence — no
    disk or network I/O required. Used by search/context/router/llm unit
    tests that need chunk dicts but aren't testing ingestion itself."""
    return [
        {
            "chunk_id": "c1",
            "file_path": "sample_repo/transformations/sales/sales_summary.py",
            "file_name": "sales_summary",
            "file_type": "python",
            "layer": "transformations",
            "domain": "sales",
            "chunk_type": "function",
            "chunk_name": "compute_total",
            "content": "def compute_total(rows):\n    return sum(r['amount'] for r in rows)",
        },
        {
            "chunk_id": "c2",
            "file_path": "sample_repo/pipelines/staging/sales/sales_pipeline.yaml",
            "file_name": "sales_pipeline",
            "file_type": "yaml",
            "layer": "staging",
            "domain": "sales",
            "chunk_type": "pipeline_config",
            "chunk_name": "sales_pipeline",
            "content": "pipeline_name: sales_daily\nsource: staging.sales_raw\ntarget: curated.sales_summary",
        },
        {
            "chunk_id": "c3",
            "file_path": "confluence_cache/999_loyalty.txt",
            "file_name": "loyalty",
            "file_type": "confluence",
            "layer": "confluence",
            "domain": "DPT",
            "chunk_type": "confluence_section",
            "chunk_name": "Data Sources",
            "content": "Data Sources\nPunchh API feeds staging tables loyalty_staging and loyalty_txn_staging.",
            "source_url": "https://example.atlassian.net/wiki/spaces/DPT/pages/999",
        },
    ]
