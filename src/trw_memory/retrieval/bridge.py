"""LLM-free entity-bridge second hop for re-ranked hybrid recall.

A question like "What activities does Melanie do?" has its evidence scattered
over turns that share few words with the question -- but share rare words
("pottery", "camping") with the turns the first hop already found. The second
hop takes the best few re-ranked rows, pulls their salient terms (high-IDF
tokens, capitalised words weighted up) that the query does not already hold,
and runs one more BM25 retrieval with ``query + terms``. Rows it surfaces from
the un-reranked tail are scored by the SAME cross-encoder against the SAME
query, so the second hop only widens the pool; it never outranks the reranker.

On LOCOMO (1,540 q, 2026-09-18) this lifted multi-hop recall@10 49.9 -> 51.3
(Wilcoxon p=0.020) and multi-hop recall@50 64.4 -> 70.5 (p<0.001), overall
hit@10 85.1 -> 85.6 (McNemar 11 vs 2, p=0.022); LongMemEval-S (470 q) was
unchanged (no hit@10 discordance). Paired median added latency per recall on
CPU: 29 ms on LOCOMO, 77 ms on LongMemEval (longer turns make the extra
cross-encoder pass, ~18 rows, dearer).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.bm25 import (
    _QUERY_STOPWORDS,
    _build_or_reuse_model,
    _normalize_text,
    _stem_token,
    bm25_search,
)

# Chosen on LOCOMO before the bridge-term filter below existed: 3 seeds / 20
# candidates / 4 terms gained less (multi-hop recall@10 p=0.083), and
# 8 / 40 / 8 matched 5 / 30 / 6 at @10 while scoring more rows.
BRIDGE_SEEDS = 5
BRIDGE_CANDIDATES = 30
BRIDGE_TERMS = 6
# A term held by more than this share of the corpus is not a bridge.
_MAX_DF_RATIO = 0.05
_CAPITAL_BOOST = 1.5
# A capitalised word that does not open the text or a sentence (names, places).
_CAPITALISED_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b")

Scorer = Callable[[list[MemoryEntry]], list[tuple[MemoryEntry, float]] | None]


def _content_tokens(text: str) -> list[str]:
    return [_stem_token(t) for t in _normalize_text(text).split() if t.isalpha() and t not in _QUERY_STOPWORDS]


def bridge_terms(
    query: str,
    seeds: list[MemoryEntry],
    entries: list[MemoryEntry],
    *,
    max_terms: int = BRIDGE_TERMS,
) -> list[str]:
    """Salient *seeds* terms absent from *query*, ranked by ``tf * idf``.

    IDF comes from the same (cached) BM25 model the first hop used over
    *entries*, so this costs no corpus pass. Tokens are in BM25's normalised,
    stemmed form, ready to append to a BM25 query.
    """
    if not seeds or not entries or max_terms < 1:
        return []
    bm25, _ids, corpus = _build_or_reuse_model(entries)
    idf: dict[str, float] = getattr(bm25, "idf", None) or {}
    # idf is monotone in document frequency, so the df ceiling is an idf floor.
    n_docs = len(corpus)
    max_df = max(2, int(_MAX_DF_RATIO * n_docs))
    min_idf = math.log((n_docs - max_df + 0.5) / (max_df + 0.5)) if n_docs > 2 * max_df else 0.0
    query_tokens = set(_content_tokens(query))
    tf: Counter[str] = Counter()
    capitalised: set[str] = set()
    for seed in seeds:
        tf.update(_content_tokens(seed.content))
        capitalised.update(_stem_token(word.lower()) for word in _CAPITALISED_RE.findall(seed.content))
    weighted: list[tuple[float, str]] = []
    for token, count in tf.items():
        token_idf = float(idf.get(token, 0.0))
        if token in query_tokens or len(token) < 3 or token_idf <= 0.0 or token_idf < min_idf:
            continue
        weighted.append((count * token_idf * (_CAPITAL_BOOST if token in capitalised else 1.0), token))
    weighted.sort(key=lambda pair: (-pair[0], pair[1]))
    return [token for _, token in weighted[:max_terms]]


def extend_with_bridge(
    query: str,
    scored: list[tuple[MemoryEntry, float]],
    tail: list[MemoryEntry],
    entries: list[MemoryEntry],
    *,
    score: Scorer,
    seeds: int = BRIDGE_SEEDS,
    candidates: int = BRIDGE_CANDIDATES,
    max_terms: int = BRIDGE_TERMS,
) -> tuple[list[tuple[MemoryEntry, float]], list[MemoryEntry], bool]:
    """Add second-hop rows from *tail* to the re-ranked *scored* list.

    Args:
        query: The retrieval query the first hop ran.
        scored: ``(entry, cross-encoder score)`` pairs, best first.
        tail: Eligible rows the first hop did NOT re-rank, in fusion order.
            Only these can be promoted, so every exclusion already applied
            upstream (namespace scope, validity prior, ``as_of``) still holds.
        entries: The full candidate list the first-hop BM25 indexed (a
            cache hit, not a rebuild).
        score: Scores a list of rows against the query; ``None`` = unavailable.

    Returns:
        ``(scored, tail, changed)``: the merged list re-sorted by score, the
        tail without the promoted rows, and whether anything moved.
    """
    unchanged = (scored, tail, False)
    if not scored or not tail or candidates < 1:
        return unchanged
    terms = bridge_terms(query, [entry for entry, _ in scored[:seeds]], entries, max_terms=max_terms)
    if not terms:
        return unchanged
    eligible = {entry.id: entry for entry in tail}
    # Look only a bounded distance into the second-hop ranking: its deep tail
    # matches a single bridge term and is noise the cross-encoder would pay for.
    # Terms first: the query bound keeps the leading chunks, and the bridge terms are the point.
    hits = bm25_search(f"{' '.join(terms)} {query}", entries, top_k=len(scored) + 3 * candidates)
    # Only rows that hold a bridge term are new evidence; a row matching the
    # query words alone was already ranked by the first hop, so scoring it
    # again only costs cross-encoder time.
    _bm25, ids, corpus = _build_or_reuse_model(entries)
    term_set = set(terms)
    bridged = {entry_id for entry_id, tokens in zip(ids, corpus, strict=True) if term_set.intersection(tokens)}
    fresh = [eligible[entry_id] for entry_id, _ in hits if entry_id in eligible and entry_id in bridged][:candidates]
    if not fresh:
        return unchanged
    extra = score(fresh)
    if not extra:
        return unchanged
    moved = {entry.id for entry, _ in extra}
    merged = sorted([*scored, *extra], key=lambda pair: pair[1], reverse=True)
    return merged, [entry for entry in tail if entry.id not in moved], True
