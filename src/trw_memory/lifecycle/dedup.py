"""Semantic deduplication for memory entries.

Prevents near-duplicate memories using embedding cosine similarity.
Three-tier decision: skip (>=skip_threshold), merge (>=merge_threshold), store (<merge_threshold).
Gracefully degrades to no-op when embeddings are unavailable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.dense import cosine_similarity

logger = structlog.get_logger(__name__)


class DedupResult(NamedTuple):
    """Result of a deduplication check.

    Attributes:
        action: One of "skip", "merge", or "store".
        existing_id: ID of the matched entry (for skip/merge), None for store.
        similarity: Highest cosine similarity found (0.0 when no match).
    """

    action: str  # "skip" | "merge" | "store"
    existing_id: str | None
    similarity: float


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
    2. If embedding unavailable → return DedupResult("store", None, 0.0).
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
    skip_threshold = cfg.dedup_skip_threshold
    merge_threshold = cfg.dedup_merge_threshold

    # Validate thresholds — merge must be strictly less than skip
    if merge_threshold >= skip_threshold:
        logger.warning(
            "dedup_threshold_invalid",
            merge=merge_threshold,
            skip=skip_threshold,
        )
        skip_threshold = 0.95
        merge_threshold = 0.85

    # Check embedder availability
    if embedder is None or not embedder.available():
        logger.debug("dedup_embed_unavailable", reason="no_embedder_or_unavailable")
        return DedupResult("store", None, 0.0)

    if not entries:
        return DedupResult("store", None, 0.0)

    # Generate embedding for the new content
    new_text = content + " " + detail
    new_vector = embedder.embed(new_text)

    if new_vector is None:
        logger.debug("dedup_embed_unavailable", text_len=len(new_text))
        return DedupResult("store", None, 0.0)

    # Compare against all active entries
    best_similarity = 0.0
    best_id: str | None = None

    for entry in entries:
        # Only compare against active entries
        if entry.status != MemoryStatus.ACTIVE:
            continue

        entry_text = entry.content + " " + entry.detail
        entry_vector = embedder.embed(entry_text)
        if entry_vector is None:
            continue

        sim = cosine_similarity(new_vector, entry_vector)
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
    """Merge a new memory entry into an existing entry.

    Merge strategy:
    - Tags: union of both sets (existing order preserved, new-only appended)
    - Evidence: union of both sets
    - Importance: max(existing, new)
    - Recurrence: existing + 1
    - Detail: if new detail is longer, append new detail to existing with audit trail
    - merged_from: append new entry's ID (no duplicates)
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

    # Detail: append if new detail is longer, with audit trail
    existing_detail = existing.detail
    new_detail = new_entry.detail
    if len(new_detail) > len(existing_detail):
        today = datetime.now(timezone.utc).date().isoformat()
        audit_marker = f"\n---\nMerged from {new_entry.id} on {today}:\n"
        if existing_detail:
            merged_detail = existing_detail + audit_marker + new_detail
        else:
            merged_detail = new_detail
    else:
        merged_detail = existing_detail

    # merged_from: append new entry ID (no duplicates)
    existing_merged = list(existing.merged_from)
    if new_entry.id and new_entry.id not in existing_merged:
        existing_merged.append(new_entry.id)

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
    skip_threshold = cfg.dedup_skip_threshold
    merge_threshold = cfg.dedup_merge_threshold

    if merge_threshold >= skip_threshold:
        skip_threshold = 0.95
        merge_threshold = 0.85

    # Load all active entries with their embeddings
    active_entries: list[tuple[MemoryEntry, list[float] | None]] = []
    for entry in entries:
        if entry.status != MemoryStatus.ACTIVE:
            continue
        text = entry.content + " " + entry.detail
        vec = embedder.embed(text)
        active_entries.append((entry, vec))

    merged_count = 0
    skipped_ids: set[str] = set()
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

    # Collect all modified entries (only those that changed)
    original_map = {e.id: e for e, _ in active_entries}
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
