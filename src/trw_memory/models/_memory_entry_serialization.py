"""Serialization helper for :class:`MemoryEntry`.

Keeping the full field projection outside ``models.memory`` makes the data
model module smaller while preserving the public ``MemoryEntry.to_dict`` seam.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry


def _dumped(items: list[Any]) -> list[object]:
    """Serialize model items, tolerating already-plain payloads.

    ``update()`` reconstructs an entry by ``setattr``-ing caller-supplied values
    before hashing it, so this can legitimately see raw dicts (or a JSON string)
    where models are declared. Raising here loses the caller's whole write
    inside a sync-hash computation, so degrade to the value as given instead.
    """
    out: list[object] = []
    for item in items:
        dump = getattr(item, "model_dump", None)
        out.append(dump(mode="json") if callable(dump) else item)
    return out


def memory_entry_to_dict(entry: MemoryEntry, *, fields: set[str] | None = None) -> dict[str, object]:
    """Serialize a memory entry to the legacy plain-dict shape."""
    full: dict[str, object] = {
        "id": entry.id,
        "content": entry.content,
        "detail": entry.detail,
        "tags": list(entry.tags),
        "evidence": list(entry.evidence),
        "importance": entry.importance,
        "status": entry.status.value if hasattr(entry.status, "value") else entry.status,
        "recurrence": entry.recurrence,
        "namespace": entry.namespace,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "last_accessed_at": entry.last_accessed_at.isoformat() if entry.last_accessed_at else None,
        # Bi-temporal validity (PRD-CORE-194). invalid_from/invalidated_by are
        # None for an open record (the common case); valid_from always present.
        "valid_from": entry.valid_from.isoformat(),
        "invalid_from": entry.invalid_from.isoformat() if entry.invalid_from else None,
        "invalidated_by": entry.invalidated_by,
        "access_count": entry.access_count,
        "session_count": entry.session_count,
        "q_value": entry.q_value,
        "q_observations": entry.q_observations,
        "source": entry.source,
        "source_identity": entry.source_identity,
        "client_profile": entry.client_profile,
        "model_id": entry.model_id,
        "merged_from": list(entry.merged_from),
        "consolidated_from": list(entry.consolidated_from),
        "consolidated_into": entry.consolidated_into,
        "type": entry.type.value if hasattr(entry.type, "value") else entry.type,
        "nudge_line": entry.nudge_line,
        "expires": entry.expires,
        "confidence": entry.confidence.value if hasattr(entry.confidence, "value") else entry.confidence,
        "task_type": entry.task_type,
        "domain": list(entry.domain),
        "phase_origin": entry.phase_origin,
        "phase_affinity": list(entry.phase_affinity),
        "team_origin": entry.team_origin,
        "protection_tier": entry.protection_tier.value
        if hasattr(entry.protection_tier, "value")
        else entry.protection_tier,
        "metadata": dict(entry.metadata),
        "vector_clock": dict(entry.vector_clock),
        "remote_id": entry.remote_id,
        "published_to_platform": entry.published_to_platform,
        "pending_delete": entry.pending_delete,
        "sync_hash": entry.sync_hash,
        "sync_seq": entry.sync_seq,
        "last_synced_at": entry.last_synced_at.isoformat() if entry.last_synced_at else None,
        "cross_validated": entry.cross_validated,
        "outcome_history": list(entry.outcome_history),
        "assertions": _dumped(list(entry.assertions)) if entry.assertions else [],
        "anchors": _dumped(list(entry.anchors)),
        "anchor_validity": entry.anchor_validity,
        "verification_status": entry.verification_status,
        "verification_checked_at": entry.verification_checked_at,
        "recall_count": entry.recall_count,
        "helpful_count": entry.helpful_count,
        "unhelpful_count": entry.unhelpful_count,
    }
    if fields is not None:
        return {key: value for key, value in full.items() if key in fields}
    return full
