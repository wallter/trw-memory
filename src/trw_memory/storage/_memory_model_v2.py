"""Quiesced ``memory_model_v2_importance_type`` cutover orchestrator (PRD-CORE-181-FR06).

Wave 717B converts the persisted memory model from the legacy dual
``impact``/``importance`` vocabulary to a single canonical ``importance`` +
valid ``type`` shape. This module owns the *one-time, quiesced* orchestration:

* the SQLite forward-only delta registered as ``_MIGRATIONS[2]`` in
  :mod:`trw_memory.storage._schema` (:func:`migrate_sqlite_importance_type`);
* the full maintenance-window orchestrator that checkpoints the WAL, snapshots
  the database through the SQLite backup API, stages the active/cold YAML
  ``impact -> importance`` rewrite, and commits both atomically or rolls back
  with a path/row classification report (:func:`run_memory_model_v2_cutover`);
* the backup restore path exercised on interruption
  (:func:`restore_from_backup`).

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

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryType
from trw_memory.storage.persistence import read_yaml, write_yaml

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)

__all__ = [
    "ClassificationEntry",
    "CutoverReceipt",
    "MigrationBlocked",
    "migrate_sqlite_importance_type",
    "restore_from_backup",
    "run_memory_model_v2_cutover",
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


class CutoverReceipt(BaseModel):
    """Outcome of a :func:`run_memory_model_v2_cutover` invocation."""

    migrated: bool
    schema_version: int
    active_yaml_rewritten: int = 0
    cold_yaml_rewritten: int = 0
    backup_path: str | None = None
    report: list[ClassificationEntry] = Field(default_factory=list)


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


# ---------------------------------------------------------------------------
# YAML staging (active + cold tiers)
# ---------------------------------------------------------------------------


def _plan_yaml_rewrites(
    directory: Path | None,
    *,
    kind: str,
) -> tuple[dict[Path, dict[str, object]], list[ClassificationEntry]]:
    """Stage (never write) the canonical rewrite for every YAML under *directory*.

    Returns a ``{path: rewritten_dict}`` plan plus a classification report of
    files that cannot be migrated (conflicting values or an invalid ``type``).
    A file appearing in the report is NOT included in the plan, so a blocked
    run discards it entirely.
    """
    plan: dict[Path, dict[str, object]] = {}
    report: list[ClassificationEntry] = []
    if directory is None or not directory.exists():
        return plan, report

    for path in sorted(directory.rglob("*.yaml")):
        try:
            data = read_yaml(path)
        except StorageError:
            report.append(ClassificationEntry(kind=kind, ref=str(path), reason="unreadable YAML"))
            continue

        rewritten = _rewrite_yaml_data(data, kind=kind, ref=str(path), report=report)
        if rewritten is not None:
            plan[path] = rewritten

    return plan, report


def _rewrite_yaml_data(
    data: dict[str, object],
    *,
    kind: str,
    ref: str,
    report: list[ClassificationEntry],
) -> dict[str, object] | None:
    """Compute the canonical ``importance``/``type`` rewrite for one YAML dict.

    Appends to *report* and returns ``None`` when the file is unmigratable.
    """
    has_impact = "impact" in data
    has_importance = "importance" in data
    if has_impact and has_importance and data["impact"] != data["importance"]:
        report.append(ClassificationEntry(kind=kind, ref=ref, reason="conflicting impact vs importance value"))
        return None

    new_data = dict(data)
    if has_impact:
        legacy_value = new_data.pop("impact")
        new_data["importance"] = legacy_value if not has_importance else new_data["importance"]

    type_value = new_data.get("type")
    if type_value is None or str(type_value).strip() == "":
        new_data["type"] = _DEFAULT_TYPE
    elif str(type_value) not in _VALID_TYPES:
        report.append(ClassificationEntry(kind=kind, ref=ref, reason=f"invalid type {type_value!r}"))
        return None

    return new_data


def _apply_yaml_rewrites(plan: dict[Path, dict[str, object]]) -> int:
    """Atomically write every staged rewrite. Returns the count written."""
    for path, data in plan.items():
        write_yaml(path, data)
    return len(plan)


# ---------------------------------------------------------------------------
# SQLite backup / restore (backup API)
# ---------------------------------------------------------------------------


def _snapshot_backup(conn: sqlite3.Connection, backup_path: Path) -> None:
    """Create a SQLite backup-API snapshot of *conn* at *backup_path*."""
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    dest = sqlite3.connect(str(backup_path))
    try:
        conn.backup(dest)
    finally:
        dest.close()


def restore_from_backup(db_path: Path, backup_path: Path) -> None:
    """Restore *db_path* from a backup-API snapshot at *backup_path*.

    Used to recover from an interrupted cutover: the pre-migration snapshot is
    copied back over the live database via the SQLite backup API.
    """
    if not backup_path.exists():
        raise StorageError(f"backup snapshot not found: {backup_path}", path=str(backup_path))
    source = sqlite3.connect(str(backup_path))
    dest = sqlite3.connect(str(db_path))
    try:
        source.backup(dest)
        dest.commit()
    finally:
        source.close()
        dest.close()


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


# ---------------------------------------------------------------------------
# Full quiesced orchestrator
# ---------------------------------------------------------------------------


def run_memory_model_v2_cutover(
    db_path: Path,
    *,
    active_dir: Path | None = None,
    cold_dir: Path | None = None,
    backup_dir: Path | None = None,
) -> CutoverReceipt:
    """Run the one-time, quiesced ``memory_model_v2_importance_type`` cutover.

    Sequence:
      1. Stage (validate-only) the active + cold YAML ``impact -> importance``
         rewrites. Any conflicting/invalid file is classified, not written.
      2. Open the database, ``PRAGMA wal_checkpoint(TRUNCATE)``, and snapshot it
         through the SQLite backup API. The snapshot is MANDATORY — when
         *backup_dir* is omitted it defaults to
         ``<db_path parent>/backups/pre-v2-cutover/`` (FR06: the orchestrator
         "creates a SQLite backup-API snapshot", not "may create").
      3. If the YAML staging already found blockers, refuse before touching
         SQLite — ``user_version`` is untouched and no YAML is written.
      4. Otherwise apply the SQLite v1->v2 migration through ``ensure_schema``
         (``BEGIN IMMEDIATE`` + rollback-on-block). A block restores the
         snapshot and discards the staged YAML.
      5. On full success, atomically write the staged YAML rewrites.

    Args:
        db_path: Path to the SQLite ``memory.db``.
        active_dir: Directory of active-tier ``*.yaml`` entries (optional).
        cold_dir: Directory of cold-tier ``*.yaml`` archives (optional).
        backup_dir: Directory to hold the pre-migration snapshot. Defaults to
            ``<db_path parent>/backups/pre-v2-cutover`` — a snapshot is always
            taken before any migration attempt.

    Returns:
        A :class:`CutoverReceipt` describing the outcome. ``migrated=False``
        with a non-empty ``report`` means the cutover blocked without partial
        writes.
    """
    # Lazily import to avoid an import cycle: _schema imports this module to
    # register _MIGRATIONS[2].
    from trw_memory.storage._schema import SCHEMA_VERSION, ensure_schema

    active_plan, active_report = _plan_yaml_rewrites(active_dir, kind="active_yaml")
    cold_plan, cold_report = _plan_yaml_rewrites(cold_dir, kind="cold_yaml")
    yaml_report = active_report + cold_report

    conn = sqlite3.connect(str(db_path))
    backup_path: Path | None = None
    try:
        # WAL checkpoint before snapshot; a non-WAL db raises OperationalError.
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # FR06: the pre-migration snapshot is mandatory, never optional —
        # running without an explicit backup_dir must still leave a restore
        # point (gap observed on the real 2026-07-12 cutover, which ran
        # snapshot-less because the parameter defaulted to None).
        effective_backup_dir = backup_dir or (db_path.parent / "backups" / "pre-v2-cutover")
        backup_path = effective_backup_dir / f"{db_path.name}.v1-backup"
        _snapshot_backup(conn, backup_path)

        if yaml_report:
            # YAML ambiguity blocks before any SQLite version change.
            logger.warning(
                "memory_model_v2_blocked",
                migration=MIGRATION_KEY,
                phase="yaml_staging",
                blocked=len(yaml_report),
            )
            return CutoverReceipt(
                migrated=False,
                schema_version=_user_version(conn),
                backup_path=str(backup_path) if backup_path else None,
                report=yaml_report,
            )

        try:
            ensure_schema(conn)
        except MigrationBlocked as exc:
            # ensure_schema already rolled the SQLite transaction back; restore
            # the snapshot for defence-in-depth and discard the staged YAML.
            if backup_path is not None:
                restore_from_backup(db_path, backup_path)
            logger.warning(
                "memory_model_v2_blocked",
                migration=MIGRATION_KEY,
                phase="sqlite_migration",
                blocked=len(exc.report),
            )
            return CutoverReceipt(
                migrated=False,
                schema_version=_user_version(conn),
                backup_path=str(backup_path) if backup_path else None,
                report=exc.report,
            )

        active_written = _apply_yaml_rewrites(active_plan)
        cold_written = _apply_yaml_rewrites(cold_plan)

        logger.info(
            "memory_model_v2_complete",
            migration=MIGRATION_KEY,
            schema_version=SCHEMA_VERSION,
            active_yaml_rewritten=active_written,
            cold_yaml_rewritten=cold_written,
        )
        return CutoverReceipt(
            migrated=True,
            schema_version=SCHEMA_VERSION,
            active_yaml_rewritten=active_written,
            cold_yaml_rewritten=cold_written,
            backup_path=str(backup_path) if backup_path else None,
        )
    finally:
        conn.close()


# Static type-checker anchor: the schema module imports this callable to register
# _MIGRATIONS[2]. Keeping the alias explicit documents the wiring contract.
_MIGRATION_V2: Callable[[sqlite3.Cursor], None] = migrate_sqlite_importance_type
