"""Row-mapping helpers for the SQLite backend.

Converts between SQLite row tuples and :class:`MemoryEntry` model instances.
The column order is defined by :data:`trw_memory.storage._shared.ENTRY_COLUMNS`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
    parse_dt_safe,
    parse_json_dict_int,
    parse_json_dict_str,
    parse_json_list,
    parse_optional_float,
)
from trw_memory.storage._shared import VERIFICATION_STATUS_VALUES

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


def parse_verification_status(raw: object) -> Literal["verified", "stale"] | None:
    """Normalise a persisted ``verification_status`` value (PRD-CORE-231-FR02).

    Only the in-contract literals survive — ``"verified"`` and ``"stale"``
    (PRD-CORE-244-FR03). ``None``, empty strings, and any unknown value read
    back as ``None`` ("no verdict reached") so a legacy or hand-edited row can
    never make itself un-deserialisable.
    """
    if isinstance(raw, str):
        value = raw.strip()
        if value in VERIFICATION_STATUS_VALUES:
            # VERIFICATION_STATUS_VALUES is the single source of the vocabulary
            # (the write guard reads the same set), so the cast cannot drift
            # from the Literal without failing the write guard first.
            return cast('Literal["verified", "stale"]', value)
    return None


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
        sync_hash_raw,
        sync_seq_raw,
        last_synced_at_raw,
        recall_count_raw,
        helpful_count_raw,
        unhelpful_count_raw,
        verification_status_raw,
        verification_checked_at_raw,
    ) = row

    # Deserialise assertions from JSON (PRD-CORE-086).
    # strict=False is required because the JSON round-trip stores enum values
    # as strings, and Assertion has strict=True on the model.
    assertions = parse_model_list(assertions_json, Assertion, strict=False)

    # Fail-open timestamp parsing (mirrors yaml_backend): a single WAL-reset-
    # corrupted timestamp degrades that one field instead of raising and crashing
    # the whole row read (the 2026-06-10 corruption class that took down
    # list_entries). created_at/updated_at are required, so they fall back to now.
    _now = datetime.now(timezone.utc)
    created_at_val = parse_dt_safe(created_at_s, default=_now) or _now
    updated_at_val = parse_dt_safe(updated_at_s, default=_now) or _now

    metadata = parse_json_dict_str(metadata_json)
    type_value = str(type_ or "").strip()
    confidence_value = str(confidence or "").strip()
    try:
        canonical_type = MemoryType(type_value)
    except ValueError:
        metadata.setdefault("legacy_memory_type", type_value)
        canonical_type = MemoryType.INCIDENT if type_value == "gotcha" else MemoryType.PATTERN
    try:
        canonical_confidence = Confidence(confidence_value)
    except ValueError:
        metadata.setdefault("legacy_confidence", confidence_value)
        canonical_confidence = Confidence.UNVERIFIED

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
        created_at=created_at_val,
        updated_at=updated_at_val,
        last_accessed_at=parse_dt_safe(last_accessed_s, default=None) if last_accessed_s else None,
        # PRD-CORE-194: absent valid_from (pre-migration row) => open validity,
        # back-filled to created_at by the model ``mode="before"`` validator.
        valid_from=(parse_dt_safe(valid_from_s, default=created_at_val) or created_at_val)
        if valid_from_s
        else created_at_val,
        invalid_from=parse_dt_safe(invalid_from_s, default=None) if invalid_from_s else None,
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
        metadata=metadata,
        vector_clock=parse_json_dict_int(vector_clock_json),
        remote_id=str(remote_id) if remote_id else None,
        published_to_platform=bool(published_raw),
        pending_delete=bool(pending_del_raw),
        cross_validated=bool(cross_val_raw),
        outcome_history=parse_json_list(outcome_json),
        assertions=assertions,
        anchors=parse_model_list(anchors_json, Anchor, strict=True),
        # PRD-CORE-244-FR01: a SQL NULL is "never assessed" and stays None; it
        # is NOT coerced to a perfect 1.0.
        anchor_validity=parse_optional_float(anchor_validity),
        type=canonical_type,
        nudge_line=str(nudge_line) if nudge_line else "",
        expires=str(expires) if expires else "",
        confidence=canonical_confidence,
        task_type=str(task_type) if task_type else "",
        domain=parse_json_list(domain_json),
        phase_origin=str(phase_origin) if phase_origin else "",
        phase_affinity=parse_json_list(phase_affinity_json),
        team_origin=str(team_origin) if team_origin else "",
        protection_tier=ProtectionTier(protection_tier),
        sync_hash=str(sync_hash_raw) if sync_hash_raw else "",
        sync_seq=int(str(sync_seq_raw)) if sync_seq_raw else 0,
        last_synced_at=parse_dt_safe(last_synced_at_raw, default=None) if last_synced_at_raw else None,
        recall_count=int(str(recall_count_raw)) if recall_count_raw else 0,
        helpful_count=int(str(helpful_count_raw)) if helpful_count_raw else 0,
        unhelpful_count=int(str(unhelpful_count_raw)) if unhelpful_count_raw else 0,
        # PRD-CORE-231-FR02: an unrecognised persisted literal degrades to None
        # (no adverse verdict) rather than raising and quarantining the row.
        verification_status=parse_verification_status(verification_status_raw),
        verification_checked_at=str(verification_checked_at_raw) if verification_checked_at_raw else "",
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
        # mode="json" is load-bearing: Assertion carries datetime fields
        # (last_verified_at / first_failed_at) that a plain model_dump() leaves
        # as objects, making json.dumps raise TypeError and the whole store fail.
        json.dumps([a.model_dump(mode="json") for a in entry.assertions]) if entry.assertions else "[]",
        json.dumps([a.model_dump(mode="json") for a in entry.anchors]) if entry.anchors else "[]",
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
        entry.sync_hash or "",
        entry.sync_seq,
        entry.last_synced_at.isoformat() if entry.last_synced_at else None,
        entry.recall_count,
        entry.helpful_count,
        entry.unhelpful_count,
        entry.verification_status,
        entry.verification_checked_at or "",
    )
