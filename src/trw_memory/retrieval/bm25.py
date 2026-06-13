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
import threading
from typing import TYPE_CHECKING

import structlog

from trw_memory.models.memory import MemoryEntry

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BM25_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from rank_bm25 import BM25Okapi as _BM25OkapiType

logger = structlog.get_logger(__name__)


# Invalidation-based corpus cache (PRD: BM25Okapi rebuild O(N) per recall call).
# At 1M+ entries, rebuilding the tokenized corpus + BM25Okapi model on every
# recall costs 7-15GB RAM and seconds of CPU.  We cache the most-recently-built
# model keyed on the *set* of entry ids that produced it; when the next call
# presents the identical id set we reuse the model and the precomputed corpus
# tokenization instead of rebuilding.  Single-process, in-memory only — no
# persistence.  A lock guards the cache for thread safety (recall is called
# concurrently by parallel agents).
#
# Cache entry: (id_set, model, ordered_ids, corpus_tokens)
#   id_set        — frozenset of entry ids that built the model (invalidation key)
#   model         — the cached BM25Okapi instance
#   ordered_ids   — entry ids in the order the corpus rows were built
#   corpus_tokens — the tokenized corpus rows (reused for the Jaccard fallback)
_bm25_cache: tuple[frozenset[str], "_BM25OkapiType", list[str], list[list[str]]] | None = None
_bm25_cache_lock = threading.Lock()


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


def _build_or_reuse_model(
    entries: list[MemoryEntry],
) -> tuple["_BM25OkapiType", list[str], list[list[str]]]:
    """Return a BM25Okapi model + the entry-id order and corpus it was built on.

    Reuses the module-level cache when the *set* of entry ids is unchanged from
    the previous call.  The cache is invalidated (rebuilt) whenever the id set
    differs — added, removed, or swapped entries.  Both the model and the
    tokenized corpus rows are reused, so a cache hit skips re-tokenizing every
    entry as well as reconstructing the BM25Okapi index.

    The returned ``ordered_ids`` and ``corpus`` are in the model's *build order*
    (``model.get_scores()[i]`` corresponds to ``ordered_ids[i]`` /
    ``corpus[i]``).  Because the model's score vector is positionally bound to
    the order it was constructed in, the caller MUST align scores to entries by
    id — not by ``entries`` position — so a reordered (but set-identical) call
    still scores correctly off the cached model.

    Args:
        entries: Candidate memory entries (must be non-empty).

    Returns:
        ``(model, ordered_ids, corpus)`` — all three in the model's build order.
    """
    global _bm25_cache

    id_set = frozenset(e.id for e in entries)

    with _bm25_cache_lock:
        cached = _bm25_cache
        if (
            cached is not None
            and cached[0] == id_set
            and len(id_set) == len(entries)  # no duplicate ids — safe to map by id
        ):
            _, model, ordered_ids, cached_corpus = cached
            logger.debug("bm25_cache_hit", entry_count=len(entries))
            return model, ordered_ids, cached_corpus

    # Cache miss (or unsafe to reuse): rebuild outside the lock to avoid holding
    # it during the O(N) tokenization + index construction.
    corpus = [_tokenize_entry(e) for e in entries]
    model = BM25Okapi(corpus)
    ordered_ids = [e.id for e in entries]

    # Only cache when the id set is unambiguous (no duplicate ids); duplicate
    # ids would break the by-id score lookup on a subsequent hit.
    if len(id_set) == len(entries):
        with _bm25_cache_lock:
            _bm25_cache = (id_set, model, ordered_ids, corpus)
    logger.debug("bm25_cache_miss", entry_count=len(entries))
    return model, ordered_ids, corpus


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

    # Reuse a cached BM25Okapi model + tokenized corpus when the entry-id set is
    # unchanged from the prior call; otherwise rebuild and refresh the cache.
    # ``ordered_ids`` / ``corpus`` are in the model's BUILD order, which is the
    # order ``get_scores()`` returns — so we map scores to ids by build position,
    # never by ``entries`` position (the two can differ on a reordered cache hit).
    bm25, ordered_ids, corpus = _build_or_reuse_model(entries)

    # Mirror the document tokenizer's hyphen-expansion so "pydantic-v2" in a
    # query matches both the composite token and the split tokens indexed from tags.
    _raw_q = [t for t in _normalize_text(query).split() if t]
    tokenized_query: list[str] = []
    for _t in _raw_q:
        tokenized_query.append(_t)
        if "-" in _t:
            tokenized_query.extend(_t.split("-"))

    scores = bm25.get_scores(tokenized_query)

    # Build (entry_id, score) pairs by the model's build order — skip blank ids.
    # ``tokens_by_id`` lets the Jaccard fallback below address entries by id
    # regardless of the current ``entries`` ordering.
    tokens_by_id: dict[str, list[str]] = {}
    paired: list[tuple[str, float]] = []
    for i, entry_id in enumerate(ordered_ids):
        if entry_id:
            paired.append((entry_id, float(scores[i])))
            tokens_by_id[entry_id] = corpus[i]

    # BM25 IDF is 0 or negative when a term appears in >= N/2 documents (small
    # corpora).  Fall back to token-overlap scoring when no entries score > 0:
    # rank_bm25 BM25Okapi IDF can go negative (log((N+0.5)/(df+0.5)) < 0 when
    # df > N/2), so checking all(s == 0.0) misses the negative-score case.
    if all(s <= 0.0 for _, s in paired):
        query_set = set(tokenized_query)
        fallback: list[tuple[str, float]] = []
        # Address tokens by id (build-order safe) so a reordered cache hit and a
        # fresh build produce identical fallback rankings.  Blank ids were never
        # added to ``tokens_by_id`` so they are skipped here too.
        for entry_id, entry_tokens_list in tokens_by_id.items():
            entry_tokens = set(entry_tokens_list)
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
