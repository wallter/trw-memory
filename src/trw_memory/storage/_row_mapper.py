"""Row-mapping helpers for the SQLite backend.

Converts between SQLite row tuples and :class:`MemoryEntry` model instances.
The column order is defined by :data:`trw_memory.storage._shared.ENTRY_COLUMNS`.
"""

from __future__ import annotations

import json
from typing import Literal, cast

from trw_memory.models.memory import Assertion, MemoryEntry, MemoryStatus
from trw_memory.storage._parsing import (
    parse_dt,
    parse_json_dict_int,
    parse_json_dict_str,
    parse_json_list,
)

# Source provenance values accepted by MemoryEntry.
_SourceType = Literal["human", "agent", "tool", "consolidated"]


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
        access_count,
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
    ) = row

    # Deserialise assertions from JSON (PRD-CORE-086)
    # strict=False is required because the JSON round-trip stores enum values
    # as strings, and Assertion has strict=True on the model.
    assertions: list[Assertion] = []
    if assertions_json and assertions_json != "[]":
        try:
            assertions = [
                Assertion.model_validate(a, strict=False)
                for a in json.loads(str(assertions_json))
            ]
        except (json.JSONDecodeError, ValueError):
            assertions = []

    return MemoryEntry(
        id=str(id_),
        content=str(content),
        detail=str(detail) if detail else "",
        tags=parse_json_list(tags_json),
        evidence=parse_json_list(evidence_json),
        importance=float(str(importance)),
        status=MemoryStatus(str(status)),
        recurrence=int(str(recurrence)),
        namespace=str(namespace),
        created_at=parse_dt(created_at_s),
        updated_at=parse_dt(updated_at_s),
        last_accessed_at=parse_dt(last_accessed_s) if last_accessed_s else None,
        access_count=int(str(access_count)),
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
    )


def entry_to_row(entry: MemoryEntry) -> tuple[object, ...]:
    """Convert a :class:`MemoryEntry` to an INSERT/REPLACE row tuple."""
    # Pydantic v2: use_enum_values=True + strict=True can leave the field
    # as an enum instance in some code paths.  Safely extract the value.
    raw_status = entry.status
    status_val = raw_status.value if isinstance(raw_status, MemoryStatus) else str(raw_status)
    return (
        entry.id,
        entry.content,
        entry.detail,
        json.dumps(entry.tags),
        json.dumps(entry.evidence),
        entry.importance,
        status_val,
        entry.recurrence,
        entry.namespace,
        entry.created_at.isoformat(),
        entry.updated_at.isoformat(),
        entry.last_accessed_at.isoformat() if entry.last_accessed_at else None,
        entry.access_count,
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
    )
