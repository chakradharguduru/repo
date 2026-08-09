"""
search.py — RepoSearchEngine.

File-name matching is a SOFT re-rank (never a hard filter), and matches are
now WEIGHTED by how specific they are:

  * Exact full file-name match ("survey_summary" == the whole stem) is the
    strongest signal -> that file is ranked first.
  * A specific multi-word phrase ("survey_summary" pointing to a handful of
    files) is a medium signal -> those files are promoted next.
  * A broad single word ("survey" pointing to 16 files) is too weak to be a
    reliable "the user named this file" signal -> those files are NOT
    force-promoted; they fall back to plain semantic search.

This is what stops the base `survey_summary` from being buried under its
own `_hist` / `_api` / `db_` variants: exact-name affinity ranks it first,
the variants follow (so answers can combine them), and unrelated survey
files drop out of the promoted group.

Common English words that also appear in file names ("type", "data",
"summary", "table", ...) are stopwords: on their own they never identify a
file. Multi-word phrases are unaffected.

Nothing is ever excluded, so a wrong guess only reorders — it can't hide
the right answer.
"""
from typing import List, Dict, Optional, Set, Tuple
import re

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


# Single words that must NOT identify a file on their own (too generic).
# Multi-word phrases (e.g. "payment_type") still match when both words appear.
STOPWORD_ALIASES: Set[str] = {
    "type", "types", "name", "names", "data", "value", "values",
    "key", "keys", "config", "configs", "util", "utils", "main", "base",
    "common", "core", "test", "tests", "update", "updates", "table",
    "tables", "general", "custom", "order", "orders", "payment",
    "payments", "summary", "source", "target", "resolver", "resolvers",
    "get", "set", "list", "new", "old", "the", "and", "for", "with",
}

# A promoted file must clear this affinity. Broad single words score ~1/N
# (N = files they touch); a word hitting 8+ files (<=0.125) is dropped from
# promotion and left to semantic search. Phrases and exact names score high.
PROMOTE_MIN_AFFINITY = 0.15


class RepoSearchEngine:
    def __init__(self, chunks: List[Dict], embeddings: np.ndarray, model: Optional[SentenceTransformer] = None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model or SentenceTransformer(config.EMBED_MODEL_NAME)

        self.layers = sorted({c["layer"] for c in chunks if c["layer"]})
        self.domains = sorted({c["domain"] for c in chunks if c["domain"]})
        self.file_names = sorted({c["file_name"] for c in chunks if c["file_name"]})

        # One FAISS index over the WHOLE corpus, built once.
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        # Alias index: full name, words (>=3 chars), and two-word phrases.
        self.file_aliases: Dict[str, List[str]] = {}
        raw: Dict[str, set] = {}
        for file_name in self.file_names:
            parts = file_name.lower().split("_")
            aliases = {file_name.lower()}
            aliases.update(p for p in parts if len(p) >= 3)
            aliases.update("_".join(parts[i:i + 2]) for i in range(len(parts) - 1))
            for alias in aliases:
                raw.setdefault(alias, set()).add(file_name)

        self.alias_scores: Dict[str, float] = {}
        for alias, files in raw.items():
            files_sorted = sorted(files)
            self.file_aliases[alias] = files_sorted
            self.alias_scores[alias] = 1 / len(files_sorted)  # specificity: rarer = higher

        if config.DEBUG:
            print(f"Loaded {len(self.layers)} layers, "
                  f"{len(self.domains)} domains, {len(self.file_names)} file names")

    # ------------------------------------------------------------------
    # Signal extraction (used only to promote/re-rank, never to exclude)
    # ------------------------------------------------------------------
    def _matched_layer(self, query_lower: str) -> Optional[str]:
        for layer in self.layers:
            if layer.lower() in query_lower:
                return layer
        return None

    def _file_affinities(self, query: str) -> Tuple[Dict[str, float], List[str]]:
        """Return {file_name: affinity} plus the alias hits (for debugging).

        Affinity for a file = the strongest matched alias pointing to it:
          specificity (1/#files)  +0.1 if the alias is multi-word
                                   +1.0 if the alias IS the file's full name.

        STOPWORD_ALIASES only suppresses a single word acting as a FRAGMENT of
        a longer name (e.g. "order" inside "revel_staging_brand_order" — too
        generic to mean the user named that file). It must NOT suppress a word
        that is a file's entire name (e.g. a Confluence page literally titled
        "Order") — that's a deliberate, specific reference, not a stray
        business term, and dropping it entirely was hiding exact page-name
        matches whenever the title happened to also be a generic English word.

        Multi-word aliases require the words to be ADJACENT and in-order in
        the query (not just present anywhere), matching how they were built
        from adjacent parts of the file name. A bag-of-words check ("both
        words appear somewhere") is fine for short, deliberate queries like
        "what is the payment type", but on a long natural-language question
        it spuriously matches unrelated word pairs (e.g. "...data sources
        AND ... transaction domain..." bag-matching an alias "and_transaction"
        from a completely unrelated file) purely by coincidence.
        """
        query_words_seq = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        query_words = set(query_words_seq)

        matched_aliases: List[str] = []
        for alias in self.file_aliases:
            parts = alias.split("_")
            if len(parts) > 1:
                n = len(parts)
                adjacent = any(
                    query_words_seq[i:i + n] == parts
                    for i in range(len(query_words_seq) - n + 1)
                )
                if alias in query_words or adjacent:
                    matched_aliases.append(alias)
            else:
                if alias in query_words:
                    matched_aliases.append(alias)

        affinity: Dict[str, float] = {}
        for alias in matched_aliases:
            base = self.alias_scores[alias]
            multiword_bonus = 0.1 if "_" in alias else 0.0
            is_stopword = "_" not in alias and alias in STOPWORD_ALIASES
            for f in self.file_aliases[alias]:
                exact = (alias == f.lower())
                if is_stopword and not exact:
                    continue  # generic fragment match only — not a reliable signal
                score = base + multiword_bonus
                if exact:
                    score += 1.0
                if score > affinity.get(f, 0.0):
                    affinity[f] = score

        return affinity, matched_aliases

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict]:
        if not self.chunks:
            return []

        q = self.model.encode([query], convert_to_numpy=True)  # (1, dim)
        q_vec = q[0]

        affinity, hits = self._file_affinities(query)
        promoted = {f: s for f, s in affinity.items() if s >= PROMOTE_MIN_AFFINITY}
        layer = self._matched_layer(query.lower())

        if config.DEBUG:
            top = sorted(promoted.items(), key=lambda t: -t[1])
            print("\nFile-name hits :", hits or "(none)")
            print("Promoted files :", [f"{f}({s:.2f})" for f, s in top] or "(none)")
            print("Layer hint     :", layer or "(none)")

        # Tier 1: chunks from promoted files, ordered by (affinity, semantic).
        # Pulled in directly so a named file is always considered.
        tier1_idxs: List[int] = []
        if promoted:
            p_idxs = [i for i, c in enumerate(self.chunks)
                      if c["file_name"] in promoted]
            if p_idxs:
                sub = self.embeddings[p_idxs]
                dists = np.linalg.norm(sub - q_vec, axis=1)
                order = sorted(
                    range(len(p_idxs)),
                    key=lambda k: (-promoted[self.chunks[p_idxs[k]]["file_name"]], dists[k]),
                )
                tier1_idxs = [p_idxs[k] for k in order]
        tier1_set = set(tier1_idxs)

        # Semantic pool over the WHOLE corpus for everything else.
        pool = min(len(self.chunks), max(top_k * 8, 50))
        distances, indices = self.index.search(q, pool)

        tier2_idxs, tier3_idxs = [], []  # tier2 = layer match, tier3 = rest
        for idx in indices[0]:
            if idx == -1 or idx in tier1_set:
                continue
            if layer and self.chunks[idx]["layer"] == layer:
                tier2_idxs.append(idx)
            else:
                tier3_idxs.append(idx)

        ordered = tier1_idxs + tier2_idxs + tier3_idxs
        return [self.chunks[i] for i in ordered[:top_k]]