"""Row-mapping helpers for the SQLite backend.

Converts between SQLite row tuples and :class:`MemoryEntry` model instances.
The column order is defined by :data:`trw_memory.storage._shared.ENTRY_COLUMNS`.
"""

from __future__ import annotations

import json
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, ValidationError

from trw_memory.models.memory import (
    Anchor,
    Assertion,
    Confidence,
    MemoryEntry,
    MemoryStatus,
    MemoryType,
    ProtectionTier,
)
from trw_memory.storage._parsing import (
    parse_dt,
    parse_float,
    parse_json_dict_int,
    parse_json_dict_str,
    parse_json_list,
)

# Source provenance values accepted by MemoryEntry.
_SourceType = Literal["human", "agent", "tool", "consolidated"]

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def parse_model_list(raw: object, model: type[_ModelT], *, strict: bool) -> list[_ModelT]:
    """Deserialise a JSON-encoded list of pydantic models, degrading to ``[]``.

    A single corrupted JSON column must never crash row mapping for an entire
    query, so malformed payloads (invalid JSON, a non-list root, or items that
    fail model validation) fall back to an empty list rather than propagating.

    ``strict`` is forwarded to :meth:`model_validate`; persisted enum values are
    stored as strings, so caller-controlled strictness keeps the round-trip
    consistent with each model's ``model_config``.
    """
    if not raw or raw == "[]":
        return []
    try:
        items = json.loads(str(raw))
        return [model.model_validate(item, strict=strict) for item in items]
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
        return []


def row_to_entry(row: tuple[object, ...]) -> MemoryEntry:
    """Convert a SQLite row tuple to a :class:`MemoryEntry`.

    The column order must match
    :data:`trw_memory.storage._shared.ENTRY_COLUMNS`.
    """
    (
        id_,
        content,
        detail,
        tags_json,
        evidence_json,
        importance,
        status,
        recurrence,
        namespace,
        created_at_s,
        updated_at_s,
        last_accessed_s,
        valid_from_s,
        invalid_from_s,
        invalidated_by_raw,
        access_count,
        session_count,
        q_value,
        q_obs,
        source,
        source_identity,
        client_profile,
        model_id,
        merged_json,
        cons_from_json,
        consolidated_into,
        metadata_json,
        vector_clock_json,
        remote_id,
        published_raw,
        pending_del_raw,
        cross_val_raw,
        outcome_json,
        assertions_json,
        anchors_json,
        anchor_validity,
        type_,
        nudge_line,
        expires,
        confidence,
        task_type,
        domain_json,
        phase_origin,
        phase_affinity_json,
        team_origin,
        protection_tier,
        sessions_surfaced,
        avg_rework_delta_raw,
        outcome_correlation_raw,
        sync_hash_raw,
        sync_seq_raw,
        last_synced_at_raw,
        recall_count_raw,
        helpful_count_raw,
        unhelpful_count_raw,
    ) = row

    # Deserialise assertions from JSON (PRD-CORE-086).
    # strict=False is required because the JSON round-trip stores enum values
    # as strings, and Assertion has strict=True on the model.
    assertions = parse_model_list(assertions_json, Assertion, strict=False)

    return MemoryEntry(
        id=str(id_),
        content=str(content),
        detail=str(detail) if detail else "",
        tags=parse_json_list(tags_json),
        evidence=parse_json_list(evidence_json),
        importance=float(str(importance)),
        status=MemoryStatus(status),
        recurrence=int(str(recurrence)),
        namespace=str(namespace),
        created_at=parse_dt(created_at_s),
        updated_at=parse_dt(updated_at_s),
        last_accessed_at=parse_dt(last_accessed_s) if last_accessed_s else None,
        # PRD-CORE-194: absent valid_from (pre-migration row) => open validity,
        # back-filled to created_at by the model ``mode="before"`` validator.
        valid_from=parse_dt(valid_from_s) if valid_from_s else parse_dt(created_at_s),
        invalid_from=parse_dt(invalid_from_s) if invalid_from_s else None,
        invalidated_by=str(invalidated_by_raw) if invalidated_by_raw else None,
        access_count=int(str(access_count)),
        session_count=int(str(session_count)) if session_count else 0,
        q_value=float(str(q_value)),
        q_observations=int(str(q_obs)),
        source=cast("_SourceType", str(source)),
        source_identity=str(source_identity) if source_identity else "",
        client_profile=str(client_profile) if client_profile else "",
        model_id=str(model_id) if model_id else "",
        merged_from=parse_json_list(merged_json),
        consolidated_from=parse_json_list(cons_from_json),
        consolidated_into=str(consolidated_into) if consolidated_into else None,
        metadata=parse_json_dict_str(metadata_json),
        vector_clock=parse_json_dict_int(vector_clock_json),
        remote_id=str(remote_id) if remote_id else None,
        published_to_platform=bool(published_raw),
        pending_delete=bool(pending_del_raw),
        cross_validated=bool(cross_val_raw),
        outcome_history=parse_json_list(outcome_json),
        assertions=assertions,
        anchors=parse_model_list(anchors_json, Anchor, strict=True),
        anchor_validity=parse_float(anchor_validity, default=1.0),
        type=MemoryType(type_),
        nudge_line=str(nudge_line) if nudge_line else "",
        expires=str(expires) if expires else "",
        confidence=Confidence(confidence),
        task_type=str(task_type) if task_type else "",
        domain=parse_json_list(domain_json),
        phase_origin=str(phase_origin) if phase_origin else "",
        phase_affinity=parse_json_list(phase_affinity_json),
        team_origin=str(team_origin) if team_origin else "",
        protection_tier=ProtectionTier(protection_tier),
        sessions_surfaced=int(str(sessions_surfaced)) if sessions_surfaced else 0,
        avg_rework_delta=float(str(avg_rework_delta_raw)) if avg_rework_delta_raw else None,
        outcome_correlation=str(outcome_correlation_raw) if outcome_correlation_raw else "",
        sync_hash=str(sync_hash_raw) if sync_hash_raw else "",
        sync_seq=int(str(sync_seq_raw)) if sync_seq_raw else 0,
        last_synced_at=parse_dt(last_synced_at_raw) if last_synced_at_raw else None,
        recall_count=int(str(recall_count_raw)) if recall_count_raw else 0,
        helpful_count=int(str(helpful_count_raw)) if helpful_count_raw else 0,
        unhelpful_count=int(str(unhelpful_count_raw)) if unhelpful_count_raw else 0,
    )


def entry_to_row(entry: MemoryEntry) -> tuple[object, ...]:
    """Convert a :class:`MemoryEntry` to an INSERT/REPLACE row tuple."""
    # Pydantic v2: use_enum_values=True converts enum instances to their string values,
    # so we use the fields directly without .value calls.
    return (
        entry.id,
        entry.content,
        entry.detail,
        json.dumps(entry.tags),
        json.dumps(entry.evidence),
        entry.importance,
        entry.status,
        entry.recurrence,
        entry.namespace,
        entry.created_at.isoformat(),
        entry.updated_at.isoformat(),
        entry.last_accessed_at.isoformat() if entry.last_accessed_at else None,
        entry.valid_from.isoformat(),
        entry.invalid_from.isoformat() if entry.invalid_from else None,
        entry.invalidated_by,
        entry.access_count,
        entry.session_count,
        entry.q_value,
        entry.q_observations,
        entry.source,
        entry.source_identity,
        entry.client_profile,
        entry.model_id,
        json.dumps(entry.merged_from),
        json.dumps(entry.consolidated_from),
        entry.consolidated_into,
        json.dumps(entry.metadata),
        json.dumps(entry.vector_clock),
        entry.remote_id,
        int(entry.published_to_platform),
        int(entry.pending_delete),
        int(entry.cross_validated),
        json.dumps(entry.outcome_history),
        json.dumps([a.model_dump() for a in entry.assertions]) if entry.assertions else "[]",
        json.dumps([a.model_dump() for a in entry.anchors]) if entry.anchors else "[]",
        entry.anchor_validity,
        entry.type,
        entry.nudge_line or "",
        entry.expires or "",
        entry.confidence,
        entry.task_type or "",
        json.dumps(entry.domain),
        entry.phase_origin or "",
        json.dumps(entry.phase_affinity),
        entry.team_origin or "",
        entry.protection_tier,
        entry.sessions_surfaced,
        str(entry.avg_rework_delta) if entry.avg_rework_delta is not None else None,
        entry.outcome_correlation or "",
        entry.sync_hash or "",
        entry.sync_seq,
        entry.last_synced_at.isoformat() if entry.last_synced_at else None,
        entry.recall_count,
        entry.helpful_count,
        entry.unhelpful_count,
    )
