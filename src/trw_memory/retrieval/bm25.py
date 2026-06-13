"""BM25 sparse retrieval for trw-memory.

Tokenizes MemoryEntry objects (content + detail + tags) and scores them
against a query using the BM25Okapi algorithm.  Falls back to token-overlap
scoring when all BM25 scores are zero (common in small corpora where IDF
becomes zero for frequently appearing terms).

Requires the optional ``bm25`` extra::

    pip install "trw-memory[bm25]"

When ``rank_bm25`` is not installed the function returns an empty list so
callers degrade gracefully without raising.
"""

from __future__ import annotations

import re

import structlog

from trw_memory.models.memory import MemoryEntry

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

logger = structlog.get_logger(__name__)


# Punctuation stripper: keep alphanumerics, whitespace, and hyphens (for tag
# expansion).  Everything else is replaced with a space so "test." matches
# "test" and "trw-memory" is kept for hyphen-expansion below.
_PUNCT_RE = re.compile(r"[^\w\s-]")

# CamelCase / PascalCase splitter: insert a space before each uppercase letter
# that follows a lowercase letter or digit so "hybridSearch" → "hybrid Search".
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and split CamelCase."""
    text = _CAMEL_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return text


def _tokenize_entry(entry: MemoryEntry) -> list[str]:
    """Build a lowercased, punctuation-stripped token list for *entry*.

    Concatenates content, detail, and tags.  Hyphenated tags are expanded so
    that ``"pydantic-v2"`` also matches query tokens ``"pydantic"`` and
    ``"v2"``.  CamelCase identifiers are split so ``"hybridSearch"`` matches
    both ``"hybrid"`` and ``"search"``.

    Args:
        entry: The memory entry to tokenize.

    Returns:
        List of lowercase string tokens.
    """
    content = _normalize_text(entry.content)
    detail = _normalize_text(entry.detail)

    tag_parts: list[str] = []
    for tag in entry.tags:
        tag_str = _normalize_text(tag)
        tag_parts.append(tag_str)
        if "-" in tag_str:
            tag_parts.extend(tag_str.split("-"))

    tags_str = " ".join(tag_parts)
    text = f"{content} {detail} {tags_str}"
    return [t for t in text.split() if t]


def bm25_search(
    query: str,
    entries: list[MemoryEntry],
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """Run BM25 sparse retrieval over a list of :class:`~trw_memory.models.memory.MemoryEntry` objects.

    Args:
        query: The search query string.
        entries: Candidate memory entries to rank.
        top_k: Maximum number of results to return.

    Returns:
        List of ``(entry_id, score)`` pairs sorted by score descending.
        Returns an empty list when ``rank_bm25`` is unavailable or *entries*
        is empty.
    """
    if not _BM25_AVAILABLE or not entries:
        logger.debug(
            "bm25_search_skipped",
            reason="unavailable" if not _BM25_AVAILABLE else "empty_entries",
            entry_count=len(entries),
        )
        return []

    corpus: list[list[str]] = [_tokenize_entry(e) for e in entries]
    # Mirror the document tokenizer's hyphen-expansion so "pydantic-v2" in a
    # query matches both the composite token and the split tokens indexed from tags.
    _raw_q = [t for t in _normalize_text(query).split() if t]
    tokenized_query: list[str] = []
    for _t in _raw_q:
        tokenized_query.append(_t)
        if "-" in _t:
            tokenized_query.extend(_t.split("-"))
        # Query-side plural normalization: "colleagues" → also "colleague".
        # Documents index possessives as their stem ("colleague's" → "colleague"
        # via _PUNCT_RE), so an unmodified plural query misses possessive forms.
        # Additive expansion preserves exact-match scores; only applies to
        # words ending in 's' that are long enough to be meaningful stems.
        if _t.endswith("s") and not _t.endswith("ss") and len(_t) > 4:
            stem = _t[:-1]
            if len(stem) >= 3:
                tokenized_query.append(stem)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)

    # Build (entry_id, score) pairs — skip entries with blank ids
    paired: list[tuple[str, float]] = []
    for i, entry in enumerate(entries):
        entry_id = entry.id
        if entry_id:
            paired.append((entry_id, float(scores[i])))

    # BM25 IDF is 0 or negative when a term appears in >= N/2 documents (small
    # corpora).  Fall back to token-overlap scoring when no entries score > 0:
    # rank_bm25 BM25Okapi IDF can go negative (log((N+0.5)/(df+0.5)) < 0 when
    # df > N/2), so checking all(s == 0.0) misses the negative-score case.
    if all(s <= 0.0 for _, s in paired):
        query_set = set(tokenized_query)
        fallback: list[tuple[str, float]] = []
        for i, entry in enumerate(entries):
            entry_id = entry.id
            if not entry_id:
                continue
            entry_tokens = set(corpus[i])
            overlap = len(query_set & entry_tokens)
            if overlap > 0:
                jaccard = overlap / len(query_set | entry_tokens)
                fallback.append((entry_id, jaccard))
        fallback.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            "bm25_search_fallback",
            query=query,
            fallback_results=len(fallback),
        )
        return fallback[:top_k]

    paired.sort(key=lambda x: x[1], reverse=True)
    results = [(eid, s) for eid, s in paired if s > 0.0][:top_k]
    logger.debug(
        "bm25_search_complete",
        query=query,
        candidates=len(entries),
        returned=len(results),
    )
    return results
