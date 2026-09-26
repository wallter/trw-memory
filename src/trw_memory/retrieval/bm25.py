"""BM25 sparse retrieval for trw-memory.

Tokenizes MemoryEntry objects (content + detail + tags) and scores them
against a query using the BM25Okapi algorithm.  Falls back to token-overlap
scoring when all BM25 scores are zero (common in small corpora where IDF
becomes zero for frequently appearing terms).

``rank-bm25`` is a base dependency (PRD-CORE-302 FR08): without the lane EngMem
complete@10 fell from 0.975 to 0.713, and the entity-bridge hop reads this model.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict

import structlog
from rank_bm25 import BM25Okapi

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.lexical import bounded_query

logger = structlog.get_logger(__name__)


# Invalidation-based corpus cache (PRD: BM25Okapi rebuild O(N) per recall call).
# At 1M+ entries, rebuilding the tokenized corpus + BM25Okapi model on every
# recall costs 7-15GB RAM and seconds of CPU.  We cache recently built models
# keyed on entry ids AND every lexical input; when a later call presents
# identical lexical entries we reuse the model and the precomputed corpus
# tokenization instead of rebuilding.  Single-process, in-memory only — no
# persistence.  A lock guards the cache for thread safety (recall is called
# concurrently by parallel agents).
#
# The cache is a small LRU rather than one slot: concurrent recalls over
# different namespaces present different corpora, and a single slot made them
# evict each other so every call paid the O(N) rebuild.  It is bounded both by
# model count and by the total rows those models index, so a few huge corpora
# cannot pin many gigabytes; the most recently built model is always kept.
#
# Key: the corpus signature — exact immutable id/content/detail/tags tuples,
#      order-independent.  A dict hit compares the full signature for equality,
#      so a hash collision can never serve another corpus's model.
# Value: (model, ordered_ids, corpus_tokens)
#   model         — the cached BM25Okapi instance
#   ordered_ids   — entry ids in the order the corpus rows were built
#   corpus_tokens — the tokenized corpus rows (reused for the Jaccard fallback)
_CorpusSignature = frozenset[tuple[str, str, str, tuple[str, ...]]]
_CachedModel = tuple[BM25Okapi, list[str], list[list[str]]]
_BM25_CACHE_MAX_MODELS = 8
_BM25_CACHE_MAX_ROWS = 200_000
_bm25_cache: OrderedDict[_CorpusSignature, _CachedModel] = OrderedDict()
_bm25_cache_lock = threading.Lock()


def _cache_store(signature: _CorpusSignature, value: _CachedModel) -> None:
    """Insert as most-recently-used, then evict LRU models over either bound."""
    with _bm25_cache_lock:
        _bm25_cache[signature] = value
        _bm25_cache.move_to_end(signature)
        rows = sum(len(ids) for _, ids, _ in _bm25_cache.values())
        while len(_bm25_cache) > 1 and (len(_bm25_cache) > _BM25_CACHE_MAX_MODELS or rows > _BM25_CACHE_MAX_ROWS):
            _, (_, evicted_ids, _) = _bm25_cache.popitem(last=False)
            rows -= len(evicted_ids)


# Punctuation stripper: keep alphanumerics, whitespace, and hyphens (for tag
# expansion).  Everything else is replaced with a space so "test." matches
# "test" and "trw-memory" is kept for hyphen-expansion below.
_PUNCT_RE = re.compile(r"[^\w\s-]")

# PRD-CORE-278 FR04: ``\w`` includes the underscore, so ``favourite_language``
# was indexed as ONE token and the query "favourite language" matched nothing —
# a measured 1-of-5 lexical hit@3 on key-value content. Identifier separators are
# split the same way CamelCase already is; the composite token is preserved
# alongside the parts (below) so a query spelling the identifier in full still
# matches exactly.
_IDENTIFIER_RE = re.compile(r"_+")

# CamelCase / PascalCase splitter: insert a space before each uppercase letter
# that follows a lowercase letter or digit so "hybridSearch" → "hybrid Search".
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Query-side stopwords. Natural-language recall queries ("what did Caroline
# research?", "when did we last change the retry policy?") are dominated by
# function words that match nearly every document; with rank_bm25's IDF
# floor they still carry weight and push documents that merely share "what"
# and "did" above the ones that share the content terms. Dropping them from
# the QUERY only (documents keep every token, so phrase-like tags such as
# "how-to" still index) lifted LOCOMO evidence hit@10 from 57% to 64% for
# BM25 alone and from 64.5% to 69.7% after fusion (benchmarks/locomo,
# 2026-09-17). A query made entirely of stopwords keeps its tokens so it
# still matches something rather than nothing.
_QUERY_STOPWORD_TEXT = """
    a an the and or but if then than so as of to in on at by for from with
    about into over after before between under during without within
    is are was were be been being am do does did done doing have has had
    having can could may might must shall should will would
    i me my mine we us our ours you your yours he him his she her hers it its
    they them their theirs this that these those there here who whom whose
    which what when where why how
    not no nor yes any some all each every either neither both
"""
_QUERY_STOPWORDS = frozenset(_QUERY_STOPWORD_TEXT.split())


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and split CamelCase."""
    text = _CAMEL_RE.sub(" ", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return text


# Suffix-strip stemming applied to document AND query tokens so "researched"
# meets "research" and "agencies" meets "agency". Deliberately crude (no
# Porter tables, no dependency): only alphabetic tokens long enough that the
# stem keeps at least four characters are touched, so identifiers such as
# "v2", "sqlite3" or "fts" pass through untouched. On LOCOMO evidence
# retrieval this lifted BM25 hit@10 72.7% -> 75.8% and fused hit@50
# 90.6% -> 92.5% (385 questions, benchmarks/locomo, 2026-09-17).


def _stem_token(token: str) -> str:
    if not token.isalpha() or len(token) < 5:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("ss"):
        return token  # class, process, address
    # "-es" is only an added syllable after a sibilant in the ORIGINAL word
    # (boxes, classes, churches, wishes); "houses"/"phrases" end in -se + s, so
    # look at the letters before "es" on the token itself, never on the
    # 2-char-stripped remainder (which for "houses" is "hous" and ends in s).
    if token.endswith("es") and (token[-3] in "xz" or token[-4:-2] in ("ss", "ch", "sh")):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    for suffix in ("edly", "ing", "ed", "ly"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            base = token[: -len(suffix)]
            # stopped -> stop, running -> run (undo the doubled consonant)
            if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
                base = base[:-1]
            return base
    return token


def _with_hyphen_parts(tokens: list[str]) -> list[str]:
    """Each token, then its hyphen-separated parts: ``pydantic-v2`` also meets ``pydantic`` and ``v2``."""
    return [part for token in tokens for part in (token, *(token.split("-") if "-" in token else ()))]


def _split_identifiers(tokens: list[str]) -> list[str]:
    """Expand ``snake_case`` tokens into their parts, keeping the composite.

    Keeping the whole token means a query that spells the identifier exactly
    still scores it; adding the parts means a query that spells it as words
    scores it too. Both sides go through this helper so the index and the query
    agree (PRD-CORE-278 FR04).
    """
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if "_" in token:
            expanded.extend(part for part in _IDENTIFIER_RE.split(token) if part)
    return expanded


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

    tags_str = " ".join(_with_hyphen_parts([_normalize_text(tag) for tag in entry.tags]))
    text = f"{content} {detail} {tags_str}"
    return [_stem_token(t) for t in _split_identifiers([t for t in text.split() if t])]


def _build_or_reuse_model(
    entries: list[MemoryEntry],
) -> tuple[BM25Okapi, list[str], list[list[str]]]:
    """Return a BM25Okapi model + the entry-id order and corpus it was built on.

    Reuses a model from the module-level LRU when ids and all lexical inputs
    match a recently built corpus. Updates to content, detail or tags
    invalidate it; nonlexical changes do not. Both the model and the
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
    id_set = frozenset(e.id for e in entries)
    signature = frozenset((e.id, e.content, e.detail, tuple(e.tags)) for e in entries)

    # Duplicate ids are never cached (see below), so only an unambiguous id set
    # may look a model up — safe to map scores back by id.
    if len(id_set) == len(entries):
        with _bm25_cache_lock:
            cached = _bm25_cache.get(signature)
            if cached is not None:
                _bm25_cache.move_to_end(signature)
                logger.debug("bm25_cache_hit", entry_count=len(entries))
                return cached

    # Cache miss (or unsafe to reuse): rebuild outside the lock to avoid holding
    # it during the O(N) tokenization + index construction.
    corpus = [_tokenize_entry(e) for e in entries]
    model = BM25Okapi(corpus)
    ordered_ids = [e.id for e in entries]

    # Only cache when the id set is unambiguous (no duplicate ids); duplicate
    # ids would break the by-id score lookup on a subsequent hit.
    if len(id_set) == len(entries):
        _cache_store(signature, (model, ordered_ids, corpus))
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
        Returns an empty list when *entries* is empty.
    """
    if not entries:
        return []

    # Reuse a cached BM25Okapi model + tokenized corpus only when ids and lexical
    # inputs are unchanged; otherwise rebuild and refresh the cache.
    # ``ordered_ids`` / ``corpus`` are in the model's BUILD order, which is the
    # order ``get_scores()`` returns — so we map scores to ids by build position,
    # never by ``entries`` position (the two can differ on a reordered cache hit).
    bm25, ordered_ids, corpus = _build_or_reuse_model(entries)

    # Mirror the document tokenizer's hyphen and identifier expansion (PRD-CORE-278 FR04), so
    # "pydantic-v2" in a query matches both the composite token and the parts indexed from tags.
    tokenized_query = _split_identifiers(_with_hyphen_parts(_normalize_text(bounded_query(query)).split()))

    # Drop function words from the query; keep them only when nothing else
    # survives so an all-stopword query still degrades to the old behaviour.
    content_tokens = [t for t in tokenized_query if t not in _QUERY_STOPWORDS]
    if content_tokens:
        tokenized_query = content_tokens
    tokenized_query = [_stem_token(t) for t in tokenized_query]

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
