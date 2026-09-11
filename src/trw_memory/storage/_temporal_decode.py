"""Reject temporally unusable decoded rows before full model construction.

UTF-8 decoding still runs first. Schema quarantine covers materialized
candidates, not a complete audit of temporally excluded records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.storage._parsing import parse_json_dict_str, parse_validity_fields
from trw_memory.storage._shared import ENTRY_COLUMNS

_INDEX = {name: index for index, name in enumerate(ENTRY_COLUMNS)}


@dataclass(frozen=True, slots=True)
class _Window:
    valid_from: datetime
    invalid_from: datetime | None
    expires: str


class TemporalMaterialization:
    """Per-attempt quota counts only successfully validated deferred entries."""

    def __init__(self, selection: TemporalSelection, limit: int) -> None:
        self.selection = selection
        self.limit = limit
        self.deferred = 0

    def retain(self, row: list[object]) -> bool:
        if self.selection.exclude_system_canaries:
            metadata = parse_json_dict_str(row[_INDEX["metadata"]])
            if metadata.get("system_canary") == "true":
                return False
        _, opened, closed = parse_validity_fields(
            row[_INDEX["created_at"]],
            row[_INDEX["valid_from"]],
            row[_INDEX["invalid_from"]],
            reference_time=self.selection.reference_time,
        )
        expiry = row[_INDEX["expires_at"]]
        view = _Window(opened, closed, str(expiry) if expiry else "")
        return self.selection.eligible(view) or (self.selection.include_superseded and self.deferred < self.limit)

    def accepted(self, entry: MemoryEntry) -> None:
        if not self.selection.eligible(entry):
            self.deferred += 1
