"""Archive, restore and rollback for a consolidation cycle.

Belongs to the ``consolidation.py`` facade, which re-exports all three names —
``test_consolidation_helpers.py`` imports ``_archive_originals`` from there, and
so does ``consolidate_cycle``. Extracted for the 350 effective-LOC gate: the
0.17.0 train took ``consolidation.py`` from exactly 350 to 375.

These three are one unit on purpose. Consolidation writes a brand-new semantic
memory and THEN mutates the originals, so a failure between those two steps is
the dangerous state: the rollback path is what keeps a caller's original data
recoverable, and it must live beside the archival it undoes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryStatus

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

__all__ = ["_archive_originals", "_restore_originals", "_rollback_consolidation"]


def _archive_originals(
    cluster: list[MemoryEntry],
    consolidated_id: str,
    storage: StorageBackend,
    *,
    invalid_from: datetime | None = None,
) -> None:
    """Archive original cluster entries after consolidation.

    For each entry in *cluster*:
    1. Sets ``consolidated_into`` to the consolidated entry's ID.
    2. Sets ``status`` to ``"archived"``.
    3. PRD-CORE-194 FR04: CLOSES the validity window — sets ``invalid_from`` (the
       consolidation instant) + ``invalidated_by`` = the consolidated id. This is
       complementary to ``consolidated_into`` (the structural merge target): the
       same id names both the merge target and the window closer in the
       consolidation case. The row is RETAINED (status archived), never deleted —
       the findings-ledger retained-for-audit discipline.
    4. Updates via storage.update().

    On failure, logs ERROR and raises the exception (caller handles rollback).

    S4 fix: all per-entry archival updates run inside ONE ``storage.transaction()``
    so a crash mid-loop can never leave a cluster partially archived — either every
    original gets ``status=archived`` + ``consolidated_into`` or none do. The
    interface default ``transaction()`` is a no-op pass-through, so YAML/other
    backends keep their prior per-call-commit behaviour.

    Args:
        cluster: Original MemoryEntry objects being archived.
        consolidated_id: ID of the newly created consolidated entry.
        storage: StorageBackend for updating entries.
        invalid_from: The consolidation instant to close each original's validity
            window at. Defaults to ``now()`` when not supplied; callers thread the
            consolidated entry's ``valid_from`` so the close is gap-free (FR01
            half-open boundary: the superseding record opens exactly when the
            prior window closes).
    """
    archived_count = 0
    # The window-close instant. One shared instant for the whole cluster so all
    # originals close at the same moment the consolidated entry opens (gap-free).
    close_at = invalid_from if invalid_from is not None else datetime.now(timezone.utc)

    with storage.transaction():
        for entry in cluster:
            try:
                # FR04: close the prior window without ever clobbering a window
                # already closed by an earlier supersession (idempotent guard) —
                # an original that was already superseded keeps its first closer.
                close_fields: dict[str, object] = {
                    "consolidated_into": consolidated_id,
                    "status": MemoryStatus.ARCHIVED,
                    "updated_at": datetime.now(timezone.utc),
                }
                if entry.invalid_from is None:
                    close_fields["invalid_from"] = close_at
                    close_fields["invalidated_by"] = consolidated_id
                updated = storage.update(entry.id, namespace=entry.namespace, **close_fields)
                if updated is None:
                    raise StorageError(f"failed to archive original entry {entry.id!r}")
                archived_count += 1
            except (
                StorageError,
                ValueError,
                RuntimeError,
            ) as exc:  # per-item error handling: re-raise but log each failure individually
                logger.exception(
                    "consolidation_archive_failed",
                    entry_id=entry.id,
                    consolidated_id=consolidated_id,
                    error=str(exc),
                )
                raise

    logger.info(
        "consolidation_archive_complete",
        consolidated_id=consolidated_id,
        archived_count=archived_count,
    )


# ---------------------------------------------------------------------------
# FR06 — Dry-Run Mode + Helper
# ---------------------------------------------------------------------------


def _restore_originals(
    cluster: list[MemoryEntry],
    storage: StorageBackend,
) -> None:
    """Restore original entries after a failed consolidation attempt."""
    for entry in cluster:
        storage.store(entry)


def _rollback_consolidation(
    cluster: list[MemoryEntry],
    new_entry: MemoryEntry,
    storage: StorageBackend,
) -> None:
    """Undo a partially applied consolidation so callers keep original data.

    Consolidation creates a brand-new semantic memory and then mutates the
    originals. If archival fails after the new entry is written, leaving both
    sides in place would duplicate knowledge and silently mark only part of the
    cluster as archived. Roll back to the pre-cycle state instead.

    Restoring the originals to ACTIVE is the safety-critical half and must run
    even if deleting the new consolidated entry fails: on a YAML backend
    ``_archive_originals``' ``transaction()`` is a no-op, so a mid-loop failure
    can leave some originals already ``status=archived`` + ``consolidated_into``
    set while the consolidated entry survives. Restore the originals FIRST
    (idempotent ``store`` of their pre-cycle snapshots), then surface any
    new-entry delete failure. This guarantees a partial consolidation never
    leaves originals archived alongside a surviving consolidated entry.
    """
    _restore_originals(cluster, storage)
    deleted = storage.delete(new_entry.id, namespace=new_entry.namespace)
    if not deleted:
        raise StorageError(f"failed to delete partially consolidated entry {new_entry.id!r}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
