"""Unit tests for search.py — RepoSearchEngine ranking.

Several of these are regression tests for two real bugs found and fixed
while integrating Confluence (see project history): (1) STOPWORD_ALIASES
was discarding exact file-name matches outright, hiding a Confluence page
literally titled "Order"; (2) multi-word alias matching didn't require the
query words to be adjacent, so long natural-language questions spuriously
promoted unrelated files that happened to share two words anywhere in the
query.
"""
import numpy as np
import pytest

from search import RepoSearchEngine, PROMOTE_MIN_AFFINITY, STOPWORD_ALIASES


def make_chunk(file_name, content="placeholder content", chunk_name=None, layer=None, domain=None):
    return {
        "chunk_id": file_name + "-" + (chunk_name or file_name),
        "file_path": f"{file_name}.py",
        "file_name": file_name,
        "file_type": "python",
        "layer": layer,
        "domain": domain,
        "chunk_type": "function",
        "chunk_name": chunk_name or file_name,
        "content": content,
    }


def build_engine(chunks, embedder):
    texts = [c["content"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True) if texts else np.zeros((0, embedder.dim), dtype="float32")
    return RepoSearchEngine(chunks, embeddings, embedder)


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------
def test_engine_derives_layers_domains_file_names(fake_embedder):
    chunks = [
        make_chunk("a", layer="transformations", domain="sales"),
        make_chunk("b", layer="staging", domain="sales"),
        make_chunk("a2", chunk_name="other_chunk", layer="transformations", domain="sales"),
    ]
    engine = build_engine(chunks, fake_embedder)

    assert engine.layers == ["staging", "transformations"]
    assert engine.domains == ["sales"]
    assert set(engine.file_names) == {"a", "b", "a2"}


# ---------------------------------------------------------------------
# _matched_layer
# ---------------------------------------------------------------------
def test_matched_layer_found(fake_embedder):
    chunks = [make_chunk("a", layer="staging"), make_chunk("b", layer="curated")]
    engine = build_engine(chunks, fake_embedder)
    assert engine._matched_layer("what happens in the staging layer") == "staging"


def test_matched_layer_none(fake_embedder):
    chunks = [make_chunk("a", layer="staging")]
    engine = build_engine(chunks, fake_embedder)
    assert engine._matched_layer("what happens here") is None


# ---------------------------------------------------------------------
# _file_affinities — stopword exact-match bug regression
# ---------------------------------------------------------------------
def test_stopword_word_is_promoted_when_it_is_the_exact_file_name(fake_embedder):
    assert "order" in STOPWORD_ALIASES  # sanity: this word IS a stopword

    chunks = [
        make_chunk("order"),                 # exact match target
        make_chunk("delivery_order"),         # fragment-only match, must NOT promote
        make_chunk("catering_order"),
    ]
    engine = build_engine(chunks, fake_embedder)

    affinity, hits = engine._file_affinities("what is in the order confluence page")
    promoted = {f: s for f, s in affinity.items() if s >= PROMOTE_MIN_AFFINITY}

    assert "order" in promoted
    assert "delivery_order" not in promoted
    assert "catering_order" not in promoted
    # exact match must clearly outrank nothing else being promoted
    assert promoted["order"] > 1.0  # specificity + 1.0 exact bonus


def test_stopword_fragment_alone_does_not_promote_unrelated_files(fake_embedder):
    """The word "order" merely appearing in the query must not promote files
    that only contain it as a generic fragment — this is the ORIGINAL intent
    of STOPWORD_ALIASES and must still hold after the exact-match fix."""
    chunks = [make_chunk("delivery_order"), make_chunk("catering_order")]
    engine = build_engine(chunks, fake_embedder)

    affinity, hits = engine._file_affinities("tell me about order handling in general")
    promoted = {f: s for f, s in affinity.items() if s >= PROMOTE_MIN_AFFINITY}

    assert promoted == {}


def test_non_stopword_exact_match_still_promotes(fake_embedder):
    chunks = [make_chunk("survey_summary"), make_chunk("survey_summary_hist")]
    engine = build_engine(chunks, fake_embedder)

    affinity, hits = engine._file_affinities("what does survey_summary do")
    promoted = {f: s for f, s in affinity.items() if s >= PROMOTE_MIN_AFFINITY}

    assert "survey_summary" in promoted
    assert promoted["survey_summary"] > promoted.get("survey_summary_hist", 0)


# ---------------------------------------------------------------------
# _file_affinities — multi-word adjacency bug regression
# ---------------------------------------------------------------------
def test_multiword_alias_requires_adjacency_in_query(fake_embedder):
    """"source_target" is a real 2-word alias fragment of this file name.
    Both halves are individually STOPWORD_ALIASES (fragment-suppressed, not
    exact matches), so only the multi-word phrase match can promote this
    file — and that must require "source" and "target" to be adjacent in
    the query, not just both present somewhere in it."""
    chunks = [make_chunk("source_target_mapping")]
    engine = build_engine(chunks, fake_embedder)

    non_adjacent = "what is the source of our target system"
    affinity, hits = engine._file_affinities(non_adjacent)
    promoted = {f: s for f, s in affinity.items() if s >= PROMOTE_MIN_AFFINITY}
    assert "source_target_mapping" not in promoted

    adjacent = "please describe the source target relationship"
    affinity, hits = engine._file_affinities(adjacent)
    promoted = {f: s for f, s in affinity.items() if s >= PROMOTE_MIN_AFFINITY}
    assert "source_target_mapping" in promoted


def test_multiword_alias_matches_when_adjacent_in_query(fake_embedder):
    chunks = [make_chunk("payment_type"), make_chunk("unrelated_file")]
    engine = build_engine(chunks, fake_embedder)

    affinity, hits = engine._file_affinities("what is the payment type for this order")
    promoted = {f: s for f, s in affinity.items() if s >= PROMOTE_MIN_AFFINITY}

    assert "payment_type" in promoted


def test_multiword_alias_matches_reordered_underscored_form_in_query(fake_embedder):
    """The literal underscored alias ("payment_type") appearing verbatim in
    the query must also match, independent of the adjacency check."""
    chunks = [make_chunk("payment_type")]
    engine = build_engine(chunks, fake_embedder)

    affinity, hits = engine._file_affinities("show me the payment_type column")
    promoted = {f: s for f, s in affinity.items() if s >= PROMOTE_MIN_AFFINITY}
    assert "payment_type" in promoted


# ---------------------------------------------------------------------
# search() — end-to-end ranking
# ---------------------------------------------------------------------
def test_search_promotes_named_file_to_top(fake_embedder):
    chunks = [
        make_chunk("order", content="order primary keys store_id check_number business_date"),
        make_chunk("unrelated_topic", content="something about marketing campaigns and emails"),
    ]
    engine = build_engine(chunks, fake_embedder)

    results = engine.search("what is in the order confluence page", top_k=2)

    assert results[0]["file_name"] == "order"


def test_search_falls_back_to_semantic_similarity_without_file_name_hint(fake_embedder):
    chunks = [
        make_chunk("apples", content="apple pie recipe with cinnamon and sugar"),
        make_chunk("cars", content="engine transmission tires wheels and brakes"),
    ]
    engine = build_engine(chunks, fake_embedder)

    results = engine.search("tell me about pie and cinnamon desserts", top_k=1)

    assert results[0]["file_name"] == "apples"


def test_search_empty_chunks_returns_empty_list(fake_embedder):
    engine = build_engine([], fake_embedder)
    assert engine.search("anything") == []


def test_search_respects_top_k(fake_embedder):
    chunks = [make_chunk(f"file{i}", content=f"content number {i}") for i in range(10)]
    engine = build_engine(chunks, fake_embedder)

    results = engine.search("content", top_k=3)
    assert len(results) == 3


def test_search_never_excludes_via_wrong_file_name_guess(fake_embedder):
    """File-name matching is documented as a soft re-rank only — a chunk
    from a non-promoted file must still be returned when nothing else
    competes for the slots (never a hard filter)."""
    chunks = [make_chunk("only_file", content="the only content that exists")]
    engine = build_engine(chunks, fake_embedder)

    results = engine.search("completely different unrelated wording xyz", top_k=5)
    assert len(results) == 1
    assert results[0]["file_name"] == "only_file"
