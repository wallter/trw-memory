"""Semantic deduplication for memory entries.

Prevents near-duplicate memories using embedding cosine similarity.
Three-tier decision: skip (>=skip_threshold), merge (>=merge_threshold), store (<merge_threshold).
Gracefully degrades to no-op when embeddings are unavailable.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal, NamedTuple

import structlog

from trw_memory.embeddings._similarity_calibration import calibrated_threshold
from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.exceptions import DimensionMismatchError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import Assertion, MemoryEntry, MemoryStatus, MemoryType, ProtectionTier
from trw_memory.retrieval.dense import cosine_similarity

logger = structlog.get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Collapse whitespace + casefold for exact-duplicate comparison.

    Used by the embedding-free lexical dedup fallback: two entries whose
    content+detail normalise to the same string are unambiguous exact
    duplicates (zero false-positive risk), so they can be deduped even when no
    embedder is available.
    """
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _lexical_duplicate(
    content: str,
    detail: str,
    entries: list[MemoryEntry],
) -> DedupResult | None:
    """Embedding-free exact-text duplicate check (degraded-path fallback).

    Returns a ``merge`` DedupResult for the first ACTIVE entry whose normalised
    content+detail exactly matches the incoming text, else None. Exact match is
    treated as similarity 1.0. This is the guard that stops identical entries
    accumulating when embeddings are unavailable (the silent-no-op gap that let
    one project's store reach ~79% near-duplicates).

    Action is ``merge`` (not ``skip``) so the re-learn's tags/evidence/impact
    still fold into the survivor and ``recurrence`` increments — preserving the
    rediscovery-count signal the lifecycle relies on. This matches the trw-mcp
    sibling package's documented exact-match policy (state/dedup.py).
    """
    target = _normalize_text(f"{content} {detail}")
    if not target:
        return None
    for entry in entries:
        if entry.status != MemoryStatus.ACTIVE:
            continue
        if _normalize_text(f"{entry.content} {entry.detail}") == target:
            return DedupResult("merge", entry.id, 1.0)
    return None


# Strength orderings for protection-preserving merges (higher index = stronger),
# one merge semantics for every caller (PRD-CORE-291; trw-mcp adapts onto this module).
_PROTECTION_TIER_ORDER: tuple[str, ...] = ("low", "normal", "high", "critical", "protected", "permanent")
_CONFIDENCE_ORDER: tuple[str, ...] = ("unverified", "low", "medium", "high", "verified")


def _tier_value(tier: ProtectionTier | str) -> str:
    """Normalise a protection tier (enum or str) to its string value."""
    return tier.value if isinstance(tier, ProtectionTier) else str(tier)


def _stronger(existing_val: str, new_val: str, order: tuple[str, ...], default: str) -> str:
    """Return whichever of *existing_val* / *new_val* ranks higher in *order*.

    Unknown values fall back to *default*'s rank so an unrecognised value
    never outranks a known stronger one.
    """

    def rank(v: str) -> int:
        return order.index(v) if v in order else order.index(default)

    return new_val if rank(new_val) > rank(existing_val) else existing_val


def _stronger_protection_tier(existing: ProtectionTier | str, incoming: ProtectionTier | str) -> str:
    """Return the string value of the stronger of two protection tiers."""
    return _stronger(_tier_value(existing), _tier_value(incoming), _PROTECTION_TIER_ORDER, ProtectionTier.NORMAL.value)


def _union_assertions(existing: list[Assertion], incoming: list[Assertion]) -> list[Assertion]:
    """Union two assertion lists, de-duplicating by (type, pattern, target)."""
    merged: list[Assertion] = list(existing)
    seen: set[tuple[str, str, str]] = {(_tier_value(a.type), a.pattern, a.target) for a in existing}
    for a in incoming:
        key = (_tier_value(a.type), a.pattern, a.target)
        if key not in seen:
            merged.append(a)
            seen.add(key)
    return merged


class DedupResult(NamedTuple):
    """Result of a deduplication check.

    Attributes:
        action: One of "skip", "merge", or "store".
        existing_id: ID of the matched entry (for skip/merge), None for store.
        similarity: Highest cosine similarity found (0.0 when no match).
    """

    action: Literal["skip", "merge", "store"]
    existing_id: str | None
    similarity: float


def _validated_thresholds(cfg: MemoryConfig, embedder: EmbeddingProvider | None) -> tuple[float, float]:
    """(skip, merge) for one dedup pass, shared by every dedup entry point (PRD-CORE-042).

    Merge must be strictly below skip; otherwise a WARNING is logged and both reset
    to the defaults (0.95, 0.85). Thresholds are stated on the reference-encoder
    scale and returned in this encoder's.
    """
    skip, merge = cfg.dedup_skip_threshold, cfg.dedup_merge_threshold
    if merge >= skip:
        logger.warning("dedup_threshold_invalid", merge=merge, skip=skip)
        skip, merge = 0.95, 0.85
    return calibrated_threshold(skip, embedder), calibrated_threshold(merge, embedder)


def check_duplicate(
    content: str,
    entries: list[MemoryEntry],
    embedder: EmbeddingProvider | None,
    *,
    detail: str = "",
    config: MemoryConfig | None = None,
) -> DedupResult:
    """Check if new content is a duplicate of an existing entry.

    Steps:
    1. Generate embedding for ``content + " " + detail``.
    2. If embedding unavailable AND ``dedup_lexical_fallback`` (default True) →
       check for an exact normalized-text match and return ``merge`` on a hit;
       otherwise return DedupResult("store", None, 0.0).
    3. Filter entries to active only.
    4. For each active entry, compute cosine similarity with the new embedding.
    5. Return DedupResult based on thresholds from config.

    Args:
        content: Content of the new memory entry.
        entries: Existing entries to check against.
        embedder: EmbeddingProvider to generate vectors. Pass None to skip.
        detail: Optional detail string appended to content for embedding.
        config: MemoryConfig with dedup thresholds. Uses defaults if None.

    Returns:
        DedupResult with action ("skip", "merge", or "store"), existing_id,
        and similarity score.
    """
    cfg = config or MemoryConfig()
    skip_threshold, merge_threshold = _validated_thresholds(cfg, embedder)
    lexical_fallback = cfg.dedup_lexical_fallback

    # Check embedder availability. When embeddings are unavailable, fall back to
    # an exact normalized-text match (zero false-positive risk) instead of a
    # silent no-op — otherwise identical entries accumulate unchecked.
    if embedder is None or not embedder.available():
        if lexical_fallback:
            lexical = _lexical_duplicate(content, detail, entries)
            if lexical is not None:
                logger.debug("dedup_lexical_match", existing_id=lexical.existing_id, reason="embeddings_unavailable")
                return lexical
        logger.debug("dedup_embed_unavailable", reason="no_embedder_or_unavailable")
        return DedupResult("store", None, 0.0)

    if not entries:
        return DedupResult("store", None, 0.0)

    # Generate embedding for the new content
    new_text = content + " " + detail
    new_vector = embedder.embed(new_text)

    if new_vector is None:
        if lexical_fallback:
            lexical = _lexical_duplicate(content, detail, entries)
            if lexical is not None:
                logger.debug("dedup_lexical_match", existing_id=lexical.existing_id, reason="embed_returned_none")
                return lexical
        logger.debug("dedup_embed_unavailable", text_len=len(new_text))
        return DedupResult("store", None, 0.0)

    # Filter to active entries and batch-embed for O(1) provider calls
    active_entries = [e for e in entries if e.status == MemoryStatus.ACTIVE]
    if not active_entries:
        return DedupResult("store", None, 0.0)

    entry_texts = [e.content + " " + e.detail for e in active_entries]
    entry_vectors = embedder.embed_batch(entry_texts)

    best_similarity = 0.0
    best_id: str | None = None

    for entry, vec in zip(active_entries, entry_vectors, strict=True):
        if vec is None:
            continue

        try:
            sim = cosine_similarity(new_vector, vec)
        except DimensionMismatchError:
            # Mixed-dimension store (e.g. after an embedding-model change): a
            # candidate with a different vector width cannot be a duplicate, so
            # skip it instead of aborting the whole dedup pass with an exception.
            continue
        if sim > best_similarity:
            best_similarity = sim
            best_id = entry.id

    # Determine action based on thresholds
    if best_id is not None and best_similarity >= skip_threshold:
        return DedupResult("skip", best_id, best_similarity)
    if best_id is not None and best_similarity >= merge_threshold:
        return DedupResult("merge", best_id, best_similarity)
    return DedupResult("store", None, best_similarity)


def merge_entries(
    existing: MemoryEntry,
    new_entry: MemoryEntry,
) -> MemoryEntry:
    """Merge a new memory entry into an existing entry (PRD-CORE-291: the one
    dedup/merge implementation, adopting trw_mcp.state.dedup's lossless semantics).

    Merge strategy:
    - Tags: union of both sets (existing order preserved, new-only appended)
    - Evidence: union of both sets
    - Importance: max(existing, new)
    - Recurrence: existing + 1
    - Content: survivor keeps its own; a differing incoming content rides an audit
      header appended to detail
    - Detail: the incoming detail is ALWAYS appended under that header (never
      dropped for being shorter), unless already present verbatim in the survivor
    - merged_from: append new entry's ID and its own merged_from (no duplicates)
    - protection_tier / confidence: keep the stronger of the two
    - type: upgrade pattern -> incident when the incoming entry is an incident
    - updated_at: now

    Args:
        existing: The existing MemoryEntry to merge into.
        new_entry: The new entry being merged (will be discarded by caller).

    Returns:
        Updated MemoryEntry with merged fields (same id as existing).
    """
    # Tags: union (preserve order, existing first)
    existing_tags = list(existing.tags)
    merged_tags = existing_tags + [t for t in new_entry.tags if t not in existing_tags]

    # Evidence: union
    existing_evidence = list(existing.evidence)
    merged_evidence = existing_evidence + [e for e in new_entry.evidence if e not in existing_evidence]

    # Importance: max
    merged_importance = max(existing.importance, new_entry.importance)

    # Recurrence: increment
    merged_recurrence = existing.recurrence + 1

    # Detail + content audit trail (bug fix, learning L-bvnz): a shorter incoming
    # detail used to be silently dropped and new_entry.content was never looked
    # at. Now the survivor keeps its content; a differing incoming content rides
    # the audit header, and the incoming detail is ALWAYS appended (whatever its
    # length) unless it is already present verbatim.
    existing_detail = existing.detail
    new_detail = new_entry.detail
    today = datetime.now(timezone.utc).date().isoformat()
    new_content = " ".join(new_entry.content.split())
    kept_content = new_content if new_content and new_content != " ".join(existing.content.split()) else ""
    if new_detail and new_detail in existing_detail:
        new_detail = ""  # already present verbatim: skipping it is lossless
    if new_detail or kept_content:
        header = f"Merged from {new_entry.id} on {today}:" + (f" {kept_content}" if kept_content else "")
        body = f"{header}\n{new_detail}" if new_detail else header
        merged_detail = f"{existing_detail}\n---\n{body}" if existing_detail else (body if kept_content else new_detail)
    else:
        merged_detail = existing_detail

    # merged_from: the incoming id, then the incoming entry's own ancestry, so a
    # chained merge keeps provenance (no duplicates, never the survivor itself).
    existing_merged = list(existing.merged_from)
    for ancestor in (new_entry.id, *new_entry.merged_from):
        if ancestor and ancestor != existing.id and ancestor not in existing_merged:
            existing_merged.append(ancestor)

    # Accumulation fields — mirror consolidation._create_consolidated_entry so
    # merges don't silently discard usage counters:
    #   access_count / recall_count: sum (cumulative counters)
    #   protection_tier / confidence: keep the stronger; type: pattern -> incident upgrade
    #   assertions: union by (type, pattern, target)
    merged_access_count = existing.access_count + new_entry.access_count
    merged_recall_count = existing.recall_count + new_entry.recall_count
    merged_protection_tier = _stronger_protection_tier(existing.protection_tier, new_entry.protection_tier)
    merged_confidence = _stronger(str(existing.confidence), str(new_entry.confidence), _CONFIDENCE_ORDER, "unverified")
    is_incident_upgrade = str(new_entry.type) == "incident" and str(existing.type) != "incident"
    merged_type = MemoryType.INCIDENT.value if is_incident_upgrade else existing.type
    merged_assertions = _union_assertions(existing.assertions, new_entry.assertions)

    logger.debug(
        "dedup_merge_complete",
        existing_id=existing.id,
        new_id=new_entry.id,
        recurrence=merged_recurrence,
    )

    return existing.model_copy(
        update={
            "tags": merged_tags,
            "evidence": merged_evidence,
            "importance": merged_importance,
            "recurrence": merged_recurrence,
            "detail": merged_detail,
            "merged_from": existing_merged,
            "access_count": merged_access_count,
            "recall_count": merged_recall_count,
            "protection_tier": merged_protection_tier,
            "confidence": merged_confidence,
            "type": merged_type,
            "assertions": merged_assertions,
            "updated_at": datetime.now(timezone.utc),
        }
    )


def batch_dedup(
    entries: list[MemoryEntry],
    embedder: EmbeddingProvider | None,
    *,
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    """One-time batch deduplication of existing memory entries.

    Scans all active entries, computes pairwise similarity, merges
    near-duplicates using the same merge strategy as check_duplicate.

    Args:
        entries: List of MemoryEntry objects to scan.
        embedder: EmbeddingProvider for vector similarity. Pass None to skip.
        config: MemoryConfig with dedup thresholds. Uses defaults if None.

    Returns:
        Dict with status, entries_scanned, entries_merged, entries_skipped,
        and updated_entries (list of modified MemoryEntry objects).
    """
    if not entries:
        return {
            "status": "skipped",
            "reason": "no entries",
            "entries_scanned": 0,
            "entries_merged": 0,
            "entries_skipped": 0,
            "updated_entries": [],
        }

    if embedder is None or not embedder.available():
        return {
            "status": "skipped",
            "reason": "embeddings unavailable",
            "entries_scanned": 0,
            "entries_merged": 0,
            "entries_skipped": 0,
            "updated_entries": [],
        }

    cfg = config or MemoryConfig()
    skip_threshold, merge_threshold = _validated_thresholds(cfg, embedder)

    # Collect active entries then batch-embed in a single model call so the
    # embedding provider (sentence-transformers, etc.) can process all texts
    # together instead of N individual round-trips. This is the documented
    # "dedup batch embed" optimisation (MEMORY.md). The previous loop called
    # embedder.embed() per entry, defeating batching entirely.
    only_active = [e for e in entries if e.status == MemoryStatus.ACTIVE]
    texts = [e.content + " " + e.detail for e in only_active]
    vectors = embedder.embed_batch(texts) if texts else []
    active_entries: list[tuple[MemoryEntry, list[float] | None]] = [
        (entry, vec) for entry, vec in zip(only_active, vectors, strict=True)
    ]

    merged_count = 0
    skipped_ids: set[str] = set()
    # Snapshot originals BEFORE the mutation loop so the survivor-changed check
    # at the end compares against pre-merge state, not post-merge state.
    # (Bug: building original_map after the loop means merged_i == orig → always
    # False → merged survivor silently dropped from updated_entries.)
    original_map: dict[str, MemoryEntry] = {e.id: e for e, _ in active_entries}
    # Track updated entries by id
    updated_map: dict[str, MemoryEntry] = {e.id: e for e, _ in active_entries}

    for i in range(len(active_entries)):
        entry_i, vec_i = active_entries[i]
        id_i = entry_i.id
        if id_i in skipped_ids or vec_i is None:
            continue

        for j in range(i + 1, len(active_entries)):
            entry_j, vec_j = active_entries[j]
            id_j = entry_j.id
            if id_j in skipped_ids or vec_j is None:
                continue

            sim = cosine_similarity(vec_i, vec_j)

            if sim >= skip_threshold:
                # Exact duplicate — mark newer (j) as obsolete
                obsoleted = entry_j.model_copy(
                    update={
                        "status": MemoryStatus.OBSOLETE,
                        "detail": entry_j.detail + f"\n[Auto-obsoleted: duplicate of {id_i}, similarity={sim:.3f}]",
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                updated_map[id_j] = obsoleted
                skipped_ids.add(id_j)
                merged_count += 1

            elif sim >= merge_threshold:
                # Near-duplicate — merge j into i
                current_i = updated_map[id_i]
                merged_i = merge_entries(current_i, entry_j)
                updated_map[id_i] = merged_i
                # Update active_entries[i] so subsequent comparisons use merged data
                active_entries[i] = (merged_i, vec_i)

                obsoleted_j = entry_j.model_copy(
                    update={
                        "status": MemoryStatus.OBSOLETE,
                        "detail": entry_j.detail + f"\n[Auto-merged into {id_i}, similarity={sim:.3f}]",
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                updated_map[id_j] = obsoleted_j
                skipped_ids.add(id_j)
                merged_count += 1

    # Collect all modified entries (only those that changed vs the pre-loop snapshot)
    updated_entries: list[MemoryEntry] = []
    for entry_id, current in updated_map.items():
        orig = original_map.get(entry_id)
        if orig is None or current != orig:
            updated_entries.append(current)

    logger.debug(
        "batch_dedup_complete",
        scanned=len(active_entries),
        merged=merged_count,
        skipped=len(skipped_ids),
    )

    return {
        "status": "completed",
        "entries_scanned": len(active_entries),
        "entries_merged": merged_count,
        "entries_skipped": len(skipped_ids),
        "updated_entries": updated_entries,
    }
