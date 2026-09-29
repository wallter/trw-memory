"""Search and warmup helpers for TierManager."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from heapq import nsmallest

import structlog
from pydantic import ValidationError

from trw_memory.lifecycle.tiers._scoring import compute_importance_score
from trw_memory.lifecycle.tiers._warm_sidecar_cache import ScoreMaxima
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recall_selection import LocalCandidate, RecallInvocation
from trw_memory.security.namespace_scope import NamespaceScopeError

logger = structlog.get_logger(__name__)


def entry_matches_tokens(entry: dict[str, object], query_tokens: list[str]) -> bool:
    """Return whether any token matches the entry text surface."""
    if not query_tokens:
        return True
    content = str(entry.get("content", "")).lower()
    detail = str(entry.get("detail", "")).lower()
    raw_tags = entry.get("tags", [])
    tag_text = " ".join(str(tag).lower() for tag in raw_tags) if isinstance(raw_tags, list) else ""
    entry_id = str(entry.get("id", "")).lower()
    haystack = f"{entry_id} {content} {detail} {tag_text}"
    return any(token in haystack for token in query_tokens)


def _parse_relevance_hint(entry: dict[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = entry.get(key)
        if value is None:
            continue
        try:
            return float(str(value))
        except ValueError:
            continue
    return None


def rank_search_hits(
    entries: Iterable[dict[str, object]],
    *,
    query_tokens: list[str],
    query_embedding: list[float] | None,
    config: MemoryConfig,
    relevance_hint_keys: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """Attach composite scores and return hits sorted by descending score."""
    scored = [
        dict(
            entry,
            score=compute_importance_score(
                entry,
                query_tokens,
                query_embedding=query_embedding,
                config=config,
                relevance_hint=_parse_relevance_hint(entry, relevance_hint_keys),
            ),
        )
        for entry in entries
    ]
    scored.sort(key=lambda entry: float(str(entry.get("score", 0.0))), reverse=True)
    return scored


def search_hot_entries(
    hot_entries: Iterable[dict[str, object]],
    *,
    query_tokens: list[str],
    tags: list[str] | None,
    top_k: int,
    config: MemoryConfig,
) -> list[dict[str, object]]:
    """Filter and rank hot-tier entries without touching disk."""
    tag_set = set(tags or [])
    filtered: list[dict[str, object]] = []
    for item in hot_entries:
        item_tags = item.get("tags", [])
        if tag_set and (not isinstance(item_tags, list) or not tag_set.issubset({str(tag) for tag in item_tags})):
            continue
        if not entry_matches_tokens(item, query_tokens):
            continue
        filtered.append(dict(item))

    return rank_search_hits(filtered, query_tokens=query_tokens, query_embedding=None, config=config)[:top_k]


def warmup_hot_from_warm_entries(
    warm_entries: list[dict[str, object]],
    *,
    target: int,
    hot_put_fn: Callable[[str, MemoryEntry], None],
    config: MemoryConfig,
) -> int:
    """Populate the hot cache from warm-sidecar rows."""
    if not warm_entries:
        return 0

    ranked = sorted(
        warm_entries,
        key=lambda entry: compute_importance_score(entry, [], config=config),
        reverse=True,
    )

    loaded = 0
    for item in ranked[:target]:
        try:
            entry = MemoryEntry.model_validate(item)
        except Exception:
            logger.warning("tier_warmup_invalid_sidecar_entry", exc_info=True)
            continue
        hot_put_fn(entry.id, entry)
        loaded += 1
    return loaded


def warmup_hot_from_entries(
    entries: list[MemoryEntry],
    *,
    target: int,
    hot_put_fn: Callable[[str, MemoryEntry], None],
    config: MemoryConfig,
    mirror_to_warm_fn: Callable[[str, dict[str, object], list[float] | None], None] | None = None,
) -> int:
    """Populate the hot cache from canonical entries."""
    if not entries:
        return 0

    ranked = sorted(
        entries,
        key=lambda entry: compute_importance_score(entry.model_dump(mode="json"), [], config=config),
        reverse=True,
    )

    loaded = 0
    for entry in ranked[:target]:
        hot_put_fn(entry.id, entry)
        if mirror_to_warm_fn is not None:
            mirror_to_warm_fn(entry.id, entry.model_dump(mode="json"), None)
        loaded += 1
    return loaded


def merge_search_results(
    hot_hits: list[dict[str, object]],
    warm_hits: list[dict[str, object]],
    cold_hits: list[dict[str, object]],
    *,
    query_tokens: list[str],
    query_embedding: list[float] | None,
    tags: list[str] | None,
    config: MemoryConfig,
) -> list[dict[str, object]]:
    """Merge tier hits and rank them with a single composite score."""
    tag_set = set(tags or [])
    merged: dict[str, dict[str, object]] = {}
    for source_hits in (hot_hits, warm_hits, cold_hits):
        for item in source_hits:
            entry_id = str(item.get("id", ""))
            if not entry_id:
                continue
            item_tags = item.get("tags", [])
            if tag_set and (not isinstance(item_tags, list) or not tag_set.issubset({str(tag) for tag in item_tags})):
                continue
            merged.setdefault(entry_id, item)

    return rank_search_hits(
        merged.values(),
        query_tokens=query_tokens,
        query_embedding=query_embedding,
        config=config,
        relevance_hint_keys=("_tier_relevance",),
    )


#: Tier discovery resolves at most this many times ``top_k`` rows per recall (PRD-CORE-318 FR02).
RESOLVE_MARGIN = 4
_RESOLVE_FIRST = (-1, -1, float("-inf"))


class WindowRank:
    """How recall would rank a warm snapshot, and the most any unseen row could score (PRD-CORE-318 FR02b).

    Both use the real scorer (``compute_importance_score``) and the real rank key, so the KNN
    stop rule tracks the weights. Only rows in recall's best rank class (temporally eligible,
    first source bucket) are scored; a snapshot that does not validate says nothing and scores
    ``inf``, as discovery orders it first (``_RESOLVE_FIRST``).
    """

    def __init__(
        self,
        invocation: RecallInvocation,
        *,
        query_tokens: list[str],
        query_embedding: list[float] | None,
        config: MemoryConfig,
    ) -> None:
        self._invocation = invocation
        self._tokens = query_tokens
        self._embedding = query_embedding
        self._config = config

    def score(self, data: dict[str, object], relevance: float) -> float | None:
        """*data*'s weighted rank score at *relevance*, or ``None`` outside the best rank class."""
        invocation = self._invocation
        try:
            score = compute_importance_score(
                data,
                self._tokens,
                query_embedding=self._embedding,
                config=self._config,
                relevance_hint=relevance,
                reference_time=invocation.temporal.reference_time,
            )
            snapshot = MemoryEntry.model_validate({**data, "namespace": invocation.namespace})
        except ValueError:  # trw-fail-silent-allow: unreadable, so ordered first, as discovery orders it
            return math.inf
        ineligible, bucket, negative = invocation.rank_key(snapshot, score)
        return -negative if (ineligible, bucket) == (0, 0) else None

    def ceiling(self, maxima: ScoreMaxima) -> Callable[[float], float]:
        """The best weighted rank score any warm row could reach at a relevance at most the argument.

        Every other term takes its bound from *maxima*, taken over the WHOLE warm tier (a looser
        bound than the uncovered rows alone, never an unsound one): the highest importance, the
        most recent access and the heaviest weight of any source family present. O(1) per call.
        """
        reference = self._invocation.temporal.reference_time
        weights = self._invocation.source.weights
        weight = max((weights.get(family, 1.0) for family in maxima.families), default=0.0)
        best = {"importance": maxima.importance, "last_accessed_at": maxima.newest_access or reference.date()}

        def bound(relevance: float) -> float:
            score = compute_importance_score(
                best, [], config=self._config, relevance_hint=relevance, reference_time=reference
            )
            return score * weight

        return bound


def discover_candidates(
    rows: Iterable[tuple[dict[str, object], bool]],
    *,
    invocation: RecallInvocation,
    resolve_entries: Callable[[list[str]], Mapping[str, MemoryEntry | None]],
    query_tokens: list[str],
    query_embedding: list[float] | None,
    config: MemoryConfig,
    top_k: int,
    covered_ids: frozenset[str] = frozenset(),
) -> list[LocalCandidate]:
    """Admit, score and cap tier rows on their authoritative entries, resolving a bounded set.

    *covered_ids* names primary-backend rows the caller has already ranked for
    this query (the hybrid candidate pool). Such a row can only duplicate a
    candidate the caller holds, so it is dropped before any lookup; the
    namespace containment assertion still runs on it first.

    PRD-CORE-318 FR02: batched, bounded resolution instead of one ``get`` per row.
    The scan reads no store: it orders every row by whether its snapshot could
    match (a vector hint or a query-token hit), then by the recall rank key
    computed on the snapshot. Snapshot text orders but never excludes. The best ``RESOLVE_MARGIN * top_k`` are resolved with ONE
    ``resolve_entries`` call and admitted, scored and ranked on the canonical entry
    exactly as before; a row admission rejects makes the result shorter, never the
    read longer. With fresh snapshots the result is the unbounded one whenever
    admission rejects fewer than ``(RESOLVE_MARGIN - 1) * top_k`` of the kept rows.
    Admission (status, source kind, tags, validity) is never judged on the
    snapshot, so a canonical row that a stale snapshot would reject is still
    admitted. What the bound gives up: a row ranked below the kept set, on its
    snapshot, whose canonical row would outrank an admitted one.
    """
    bound = RESOLVE_MARGIN * top_k
    ranked: list[tuple[tuple[int, int, int, float], int, dict[str, object], bool]] = []
    for position, (data, cold) in enumerate(rows):
        if "namespace" in data and str(data["namespace"]) != invocation.namespace:
            raise NamespaceScopeError("tier snapshot outside authorized namespace")
        if str(data.get("id", "")) in covered_ids:
            continue
        hint = _parse_relevance_hint(data, ("_tier_relevance",))
        # Snapshot text only ORDERS (a matching snapshot ranks first); it never
        # excludes, because the canonical entry may match where a stale snapshot
        # does not (review r2 P1-2). The canonical pass applies the text filter.
        unmatched = int(hint is None and not entry_matches_tokens(data, query_tokens))
        try:
            score = compute_importance_score(
                data,
                query_tokens,
                query_embedding=query_embedding,
                config=config,
                relevance_hint=hint,
                reference_time=invocation.temporal.reference_time,
            )
            snapshot = MemoryEntry.model_validate({**data, "namespace": invocation.namespace})
        except ValueError:  # trw-fail-silent-allow: a malformed snapshot (ValidationError is a ValueError) is skipped as an ORDERING input, logged; the canonical row decides (review r3)
            logger.debug("tier_discovery_malformed_snapshot", entry_id=str(data.get("id", "")))
            key = _RESOLVE_FIRST  # the snapshot says nothing; let the canonical row decide
        else:
            key = invocation.rank_key(snapshot, score)
        ranked.append(((unmatched, *key), position, data, cold))
    # One read of at most ``bound`` ids, in total: rows admission rejects shrink the
    # result, they never trigger another fetch (review r2 P1-1).
    kept = sorted(nsmallest(bound, ranked, key=lambda row: row[:2]), key=lambda row: row[1])  # first-seen wins
    resolved = resolve_entries(list(dict.fromkeys(str(row[2].get("id", "")) for row in kept))) if kept else {}
    found: list[LocalCandidate] = []
    seen: set[str] = set()
    for _key, _position, data, cold in kept:
        candidate = _admit(
            data,
            cold,
            resolved,
            invocation=invocation,
            query_tokens=query_tokens,
            query_embedding=query_embedding,
            config=config,
        )
        if candidate is None or candidate.entry.id in seen:
            continue
        seen.add(candidate.entry.id)
        found.append(candidate)
    return nsmallest(top_k, found, key=lambda c: invocation.rank_key(c.entry, c.raw_score))


def _admit(
    data: dict[str, object],
    cold: bool,
    resolved: Mapping[str, MemoryEntry | None],
    *,
    invocation: RecallInvocation,
    query_tokens: list[str],
    query_embedding: list[float] | None,
    config: MemoryConfig,
) -> LocalCandidate | None:
    """One tier row on its canonical entry, or its own snapshot when the store has none.

    An id resolved to ``None`` is held by the store but withheld by its read layer
    (quarantined): it is dropped, never replaced by its snapshot (review r2 P1-3).
    """
    entry_id = str(data.get("id", ""))
    if entry_id in resolved and resolved[entry_id] is None:
        return None
    canonical = resolved.get(entry_id)
    if canonical is not None:
        return _evaluate(
            canonical,
            data,
            False,
            invocation=invocation,
            query_tokens=query_tokens,
            query_embedding=query_embedding,
            config=config,
        )
    if "created_at" not in data:
        logger.warning("tier_discovery_missing_temporal_authority", entry_id=entry_id)
        return None
    try:
        entry = MemoryEntry.model_validate(data)
    except (
        ValidationError
    ):  # trw-fail-silent-allow: a corrupt snapshot with no canonical row is not a candidate; logged, as before FR02
        logger.warning("tier_discovery_invalid_entry", entry_id=entry_id)
        return None
    return _evaluate(
        entry,
        data,
        cold,
        invocation=invocation,
        query_tokens=query_tokens,
        query_embedding=query_embedding,
        config=config,
    )


def _evaluate(
    entry: MemoryEntry,
    data: dict[str, object],
    cold: bool,
    *,
    invocation: RecallInvocation,
    query_tokens: list[str],
    query_embedding: list[float] | None,
    config: MemoryConfig,
) -> LocalCandidate | None:
    """One row's admission and score; ``None`` when recall would not admit it."""
    if not invocation.allows_entry(entry):
        return None
    if not invocation.temporal.eligible(entry) and not invocation.temporal.include_superseded:
        return None
    hint = _parse_relevance_hint(data, ("_tier_relevance",))
    payload = entry.model_dump(mode="json")
    if hint is None and not entry_matches_tokens(payload, query_tokens):
        return None
    score = compute_importance_score(
        payload,
        query_tokens,
        query_embedding=query_embedding,
        config=config,
        relevance_hint=hint,
        reference_time=invocation.temporal.reference_time,
    )
    return LocalCandidate(entry, score, relevance_hint=hint, cold=cold)
