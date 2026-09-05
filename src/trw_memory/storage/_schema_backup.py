"""Pre-migration snapshots for destructive schema deltas (PRD-CORE-245 NFR02/NFR04).

Schema 5 is the first delta that DROPS and RENAMES tables rather than adding
columns. ``ensure_schema`` runs it automatically on the first open by a new
build, which means the first process to start after an upgrade rewrites a store
that other, older processes may still hold open. The transaction guarantees the
rewrite is all-or-nothing, but it cannot give an operator a way back to the
pre-upgrade bytes — so a snapshot is taken first, and the migration refuses to
run if the snapshot cannot be written, or if whether one is needed cannot be
established. Both halves are load-bearing: a store whose row count cannot be
read is indistinguishable from an empty one, and "empty" is the one answer that
skips the snapshot entirely.

The snapshot uses SQLite's online backup API rather than a filesystem copy: it
reads through the same engine, so the WAL is included and the result is a
consistent single file even while another process is writing. A plain ``cp`` of
``memory.db`` without its ``-wal`` is exactly the silently-truncated backup this
module exists to avoid.

Snapshots are never pruned. A schema migration happens a handful of times in a
store's life, and a rotation count would be a tunable whose wrong value silently
discards the only copy of a user's memory; deleting one is an operator decision.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

__all__ = ["SchemaBackupError", "snapshot_before_migration"]

#: Directory (relative to the database file) that holds pre-migration snapshots.
BACKUP_DIR_NAME = "backups"


class SchemaBackupError(RuntimeError):
    """A pre-migration snapshot could not be written, so the migration is refused.

    Fail-closed by design: proceeding would rewrite the only copy of a store
    with no way back. The message names the destination that failed so an
    operator can free space or fix permissions and reopen.
    """


def _main_database_path(conn: sqlite3.Connection) -> Path | None:
    """Return the file backing the ``main`` schema, or ``None`` for in-memory."""
    for _seq, name, file in conn.execute("PRAGMA database_list").fetchall():
        if str(name) == "main":
            raw = str(file or "")
            return Path(raw) if raw else None
    return None


#: The only ``sqlite3`` failure that legitimately means "there is nothing to
#: protect". SQLite reports a missing relation as a generic ``OperationalError``
#: with no distinguishing error code, so its message is the sole discriminator.
_MISSING_TABLE_MESSAGE = "no such table"


def _has_rows(conn: sqlite3.Connection) -> bool:
    """Return whether ``memories`` exists and holds at least one row.

    Only an absent table counts as "empty". Every other failure -- a locked
    store, a disk I/O error, a corrupt page -- leaves the row count UNKNOWN,
    and an unknown row count is indistinguishable from a full store. Treating
    it as empty is how a populated store gets migrated with no snapshot at all,
    so it raises instead.

    Raises:
        SchemaBackupError: the probe failed for any reason other than the table
            not existing. The caller must NOT proceed with the migration.
    """
    try:
        row = conn.execute("SELECT 1 FROM memories LIMIT 1").fetchone()
    except sqlite3.OperationalError as exc:
        if _MISSING_TABLE_MESSAGE in str(exc).lower():
            return False
        raise SchemaBackupError(_probe_failure_message(exc)) from exc
    except sqlite3.Error as exc:
        raise SchemaBackupError(_probe_failure_message(exc)) from exc
    return row is not None


def _probe_failure_message(exc: BaseException) -> str:
    """Explain a failed row probe in terms of the decision it blocks."""
    return (
        f"could not determine whether this store holds data ({type(exc).__name__}: {exc}); "
        "refusing to migrate, because an unreadable store is indistinguishable from a full "
        "one and would be rewritten with no pre-migration snapshot. Resolve the underlying "
        "error (a concurrent writer holding the store, a disk fault, or corruption) and reopen."
    )


def _open_snapshot_source(db_path: Path) -> sqlite3.Connection:
    """Open a fresh sibling connection to *db_path* to serve as a backup source.

    ``sqlite3.Connection.backup()`` reads through the SAME connection you call
    it on. When that connection already holds an open write transaction — as
    ``ensure_schema``'s does, via the ``BEGIN IMMEDIATE`` migration lock — the
    backup step never returns: it retries SQLITE_BUSY forever against its own
    uncommitted transaction, with no bound and no other connection needed to
    reproduce it (verified: a lone connection with ``BEGIN IMMEDIATE`` open
    hangs in ``conn.backup(target)`` indefinitely, zero contention).

    A plain, freshly-opened connection to the same file has no transaction of
    its own, so it backs up cleanly. It does not race the migration: a
    RESERVED lock (what ``BEGIN IMMEDIATE`` holds) never blocks another
    connection's SHARED read lock, and under WAL — the mode ``ensure_schema``'s
    callers run in — a reader sees a consistent pre-transaction snapshot even
    while the writer's transaction is still open.
    """
    return sqlite3.connect(db_path, timeout=30)


def snapshot_before_migration(
    conn: sqlite3.Connection,
    *,
    from_version: int,
    to_version: int,
) -> Path | None:
    """Write a consistent copy of *conn*'s database before a destructive delta.

    Returns the snapshot path, or ``None`` when there is nothing to protect —
    an in-memory database, or a file with no ``memories`` rows yet (a fresh
    bootstrap has no prior state a rollback could want).

    Raises:
        SchemaBackupError: the snapshot could not be written, or whether one
            was needed could not be established. The caller must NOT proceed
            with the migration.
    """
    db_path = _main_database_path(conn)
    if db_path is None or not _has_rows(conn):
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = db_path.parent / BACKUP_DIR_NAME
    destination = backup_dir / f"{db_path.name}.pre-schema-{to_version}.{stamp}"
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        # Create the destination with owner-only permissions BEFORE any bytes
        # land in it — the snapshot carries the same learning content as the
        # store, so it inherits the store's 0600 posture rather than the
        # process umask.
        from trw_memory.storage._permissions import harden_db_file_mode, prepare_db_file_mode

        prepare_db_file_mode(destination)
        # ``with sqlite3.connect(...)`` is NOT a closing context manager — it
        # commits (or rolls back) and leaves the connection open. On the raise
        # path below the still-open connection is kept alive by the traceback
        # that references this frame, so it outlives the migration attempt and
        # holds a file descriptor on the half-written snapshot. ``closing``
        # makes the close explicit on every path, success and failure alike.
        with (
            contextlib.closing(sqlite3.connect(destination)) as target,
            contextlib.closing(_open_snapshot_source(db_path)) as source,
        ):
            source.backup(target)
            target.commit()
        harden_db_file_mode(destination)
    except (sqlite3.Error, OSError) as exc:
        raise SchemaBackupError(
            f"could not write the pre-migration snapshot to {destination}: {exc}; "
            f"refusing to migrate schema {from_version} -> {to_version} without one"
        ) from exc

    logger.info(
        "schema_migration_snapshot_written",
        database=str(db_path),
        backup=str(destination),
        from_version=from_version,
        to_version=to_version,
        bytes=destination.stat().st_size if destination.exists() else 0,
    )
    return destination
