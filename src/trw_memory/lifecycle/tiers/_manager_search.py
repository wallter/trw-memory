"""Search and warmup helpers for TierManager."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import structlog

from trw_memory.lifecycle.tiers._scoring import compute_importance_score
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry

logger = structlog.get_logger(__name__)


def _entry_matches_tokens(entry: dict[str, object], query_tokens: list[str]) -> bool:
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
        if not _entry_matches_tokens(item, query_tokens):
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
