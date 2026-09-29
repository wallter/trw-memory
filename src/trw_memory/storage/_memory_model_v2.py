"""SQLite ``memory_model_v2_importance_type`` delta (PRD-CORE-181-FR06).

Wave 717B converted the persisted memory model from the legacy dual
``impact``/``importance`` vocabulary to a single canonical ``importance`` +
valid ``type`` shape. This module owns the SQLite forward-only delta
registered as ``_MIGRATIONS[2]`` in :mod:`trw_memory.storage._schema`
(:func:`migrate_sqlite_importance_type`). The one-time, quiesced YAML +
backup-API cutover orchestrator this delta originally shipped alongside
(the real 2026-07-12 maintenance-window run) had no upgrade-path caller —
`ensure_schema` alone runs on every store open — and was removed in the
trw-memory deletion wave (2026-09-26); see UPGRADE-NOTES-8.0.0.md.

Invariants (do not relax without updating PRD-CORE-181):

* Missing/empty ``type`` becomes ``"pattern"``.
* An invalid ``type`` value, or a conflicting ``impact`` vs ``importance``
  value on the same row/file, BLOCKS the cutover: the ``BEGIN IMMEDIATE``
  SQLite transaction rolls back, the staged YAML rewrites are discarded, and
  ``user_version`` is NOT bumped (no partial writes).
* The external learning-API ``impact``/``min_impact`` vocabulary lives ONLY in
  the versioned mapper in :mod:`trw_memory.sync._remote_common`; the storage /
  lifecycle readers are canonical ``importance`` after this cutover.
"""

from __future__ import annotations

import json
import sqlite3

from pydantic import BaseModel, Field

from trw_memory.models.memory import MemoryType

__all__ = [
    "ClassificationEntry",
    "MigrationBlocked",
    "migrate_sqlite_importance_type",
]

MIGRATION_KEY = "memory_model_v2_importance_type"

#: Canonical ``type`` vocabulary — the migration target enum (models own it).
_VALID_TYPES: frozenset[str] = frozenset(member.value for member in MemoryType)
_DEFAULT_TYPE = MemoryType.PATTERN.value


class ClassificationEntry(BaseModel):
    """One blocked path/row and why the cutover refused to migrate it."""

    kind: str = Field(description="sqlite_row | active_yaml | cold_yaml")
    ref: str = Field(description="row id or absolute file path")
    reason: str = Field(description="human-readable classification reason")


class MigrationBlocked(RuntimeError):
    """Raised when ambiguous/invalid legacy data blocks the v2 cutover.

    Carries the :class:`ClassificationEntry` list so the caller can emit a
    path/row classification report. When this propagates out of the SQLite
    delta the surrounding transaction rolls back with no ``user_version`` bump.
    """

    def __init__(self, report: list[ClassificationEntry]) -> None:
        self.report = report
        super().__init__(f"{MIGRATION_KEY} blocked: {len(report)} unmigratable item(s)")


# ---------------------------------------------------------------------------
# SQLite forward-only delta (registered as _MIGRATIONS[2])
# ---------------------------------------------------------------------------


def migrate_sqlite_importance_type(cursor: sqlite3.Cursor) -> None:
    """Convert/verify the SQLite ``importance``/``type`` columns (v1 -> v2).

    Runs INSIDE the :func:`trw_memory.storage._schema.ensure_schema`
    transaction, so raising :class:`MigrationBlocked` here causes that
    transaction to roll back with no ``user_version`` bump.

    Steps:
      1. If a legacy ``impact`` column still coexists with ``importance``
         (the v0 rename could not collapse them), block on any row whose two
         values disagree.
      2. Backfill missing/empty ``type`` to ``"pattern"``.
      3. Reject any surviving invalid ``type`` enum value.
    """
    columns = {str(row[1]) for row in cursor.execute("PRAGMA table_info(memories)").fetchall()}
    report: list[ClassificationEntry] = []

    if "impact" in columns and "importance" in columns:
        conflicts = cursor.execute(
            "SELECT id FROM memories WHERE impact IS NOT NULL AND importance IS NOT NULL AND impact <> importance"
        ).fetchall()
        report.extend(
            ClassificationEntry(
                kind="sqlite_row",
                ref=str(row[0]),
                reason="conflicting impact vs importance value",
            )
            for row in conflicts
        )

    # Missing/empty type -> pattern (canonical default).
    cursor.execute(
        f"UPDATE memories SET type = '{_DEFAULT_TYPE}' WHERE type IS NULL OR TRIM(type) = ''"  # noqa: S608
    )

    # ``gotcha`` was emitted by a historical TRW audit producer before the
    # canonical enum shipped. Audit findings are incidents in the documented
    # taxonomy. Preserve the original value in metadata before canonicalising
    # so v1 databases can progress to the lossless v3 compatibility migration.
    for entry_id, metadata_raw in cursor.execute("SELECT id, metadata FROM memories WHERE type = 'gotcha'").fetchall():
        try:
            parsed = json.loads(str(metadata_raw or "{}"))
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = {"legacy_metadata_raw": str(metadata_raw)}
        metadata = parsed if isinstance(parsed, dict) else {"legacy_metadata_raw": metadata_raw}
        metadata.setdefault("legacy_memory_type", "gotcha")
        cursor.execute(
            "UPDATE memories SET type = 'incident', metadata = ? WHERE id = ?",
            (json.dumps(metadata, sort_keys=True), str(entry_id)),
        )

    invalid_types = [
        (str(row[0]), str(row[1]))
        for row in cursor.execute("SELECT id, type FROM memories").fetchall()
        if str(row[1]) not in _VALID_TYPES
    ]
    report.extend(
        ClassificationEntry(kind="sqlite_row", ref=entry_id, reason=f"invalid type {type_value!r}")
        for entry_id, type_value in invalid_types
    )

    if report:
        raise MigrationBlocked(report)
