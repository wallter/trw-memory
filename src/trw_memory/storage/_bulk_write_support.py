"""The caller-facing bookkeeping around ``store_many``'s one write transaction.

Split out of ``_crud_ops`` (pure move, no behaviour change): stamping the batch's sync fields, the
carried-forward evidence check that runs inside the transaction (PRD-CORE-312,
CORE-312-STORE-MANY-TXN), and restoring the caller's entries when a batch is refused.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from trw_memory.models.memory import MemoryEntry
from trw_memory.security._evidence_invariant import refuse_new_violation, violates_evidence_invariant
from trw_memory.sync.delta import DeltaTracker

#: One entry's caller-visible state before a batch stamps it.
EntryState = tuple[object, object, int, str | None, object]


def stamp_sync_bookkeeping(entries: list[MemoryEntry]) -> list[EntryState]:
    """Give each entry its timestamps and sync fields; the returned states undo it (``restore_entry_state``)."""
    now = datetime.now(timezone.utc)
    prior: list[EntryState] = [(e.created_at, e.updated_at, e.sync_seq, e.sync_hash, e.last_synced_at) for e in entries]
    for entry in entries:
        if not entry.created_at:
            entry.created_at = now
        if not entry.updated_at:
            entry.updated_at = now
        entry.sync_seq = (entry.sync_seq or 0) + 1
        entry.sync_hash = DeltaTracker.compute_sync_hash(entry)
        entry.last_synced_at = None
    return prior


def restore_entry_state(entries: list[MemoryEntry], prior: list[EntryState]) -> None:
    """A refused batch leaves the caller's entries as they were given."""
    for entry, state in zip(entries, prior, strict=True):
        entry.created_at, entry.updated_at, entry.sync_seq, entry.sync_hash, entry.last_synced_at = state  # type: ignore[assignment]


def refuse_new_violations(
    entries: list[MemoryEntry], read_existing: Callable[[MemoryEntry], MemoryEntry | None]
) -> None:
    """Refuse a batch that introduces an evidence-invariant violation the replaced row did not carry."""
    for entry in entries:
        if violates_evidence_invariant(entry):
            refuse_new_violation(read_existing(entry), entry)
