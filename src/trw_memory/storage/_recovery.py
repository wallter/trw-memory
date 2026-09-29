"""SQLite recover_db — corrupt-DB salvage + cold-tier rebuild orchestration.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``SQLiteBackend.recover_db`` becomes a 1-line delegator
to ``recover_db()`` here.

Coordinates 6 steps:

1. PRD-CORE-139 FR01/FR03/FR04: timestamped corrupt-DB rotation +
   filename-based pruning (via ``_corrupt_backup`` helpers).
2. Stale WAL/SHM cleanup + cross-process sentinel write.
3. Primary salvage via in-process ``SELECT * FROM memories``.
4. PRD-CORE-138 FR04 fallback: sqlite3 ``.recover`` CLI dump salvage.
5. PRD-CORE-140 FR03 gated cold-tier rebuild before strict-refusal raise.
6. Strict-mode refusal on non-empty backup with zero salvaged rows OR
   legacy empty-DB creation under ``recovery_policy="empty_ok"``.

Extracted as PRD-DIST-245 Phase 1 batch 83.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any, Literal

import structlog

from trw_memory._store_lock import store_access
from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.storage._anchor_index import rebuild_anchor_postings
from trw_memory.storage._connection import (
    apply_open_pragmas,
    clear_journal_sidecars,
)
from trw_memory.storage._connection import (
    connect as _connection_connect,
)
from trw_memory.storage._corrupt_backup import find_rotated_backup
from trw_memory.storage._permissions import harden_db_file_mode, prepare_db_file_mode

# Bounded open-time preflight + advisory state sidecar extracted to
# _recovery_preflight.py (PRD-DIST-245 effective-LOC ratchet). Re-exported
# here so importers that resolve these names from ``_recovery`` keep working.
from trw_memory.storage._recovery_preflight import (
    RecoveryPreflight as RecoveryPreflight,
)
from trw_memory.storage._recovery_preflight import (
    _db_identity,
    clear_recovery_marker,
    read_recovery_marker,
    write_recovery_marker,
)
from trw_memory.storage._recovery_preflight import (
    classify_recovery_preflight as classify_recovery_preflight,
)
from trw_memory.storage._recovery_preflight import (
    recovery_state_path as recovery_state_path,
)
from trw_memory.storage._recovery_preflight import (
    write_recovery_state as write_recovery_state,
)
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage._schema_v5 import rebuild_memory_tags_postings
from trw_memory.storage._shared import ENTRY_COLUMNS
from trw_memory.storage._stale_handle_detector import write_sentinel

logger = structlog.get_logger(__name__)

_PAGE_SIZE = 4096


def _resolve_cold_rebuild_base_safe(db_path: Path) -> Path:
    """Look up _resolve_cold_rebuild_base via the parent sqlite_backend module."""
    from trw_memory.storage import sqlite_backend as _sqlite_backend_module

    return _sqlite_backend_module._resolve_cold_rebuild_base(db_path)


def _backend_corrupt_backup_helpers() -> Any:
    """Look up SQLiteBackend via parent module so test monkeypatches propagate.

    Tests routinely patch ``SQLiteBackend._salvage_via_recover_cli`` /
    ``_rotate_corrupt_backup`` / ``_prune_corrupt_backups``; routing
    through the class preserves those patches.
    """
    from trw_memory.storage import sqlite_backend as _sqlite_backend_module

    return _sqlite_backend_module.SQLiteBackend


_SALVAGE_INDEXES = (
    "idx_memories_status",
    "idx_memories_namespace",
    "idx_memories_sync_seq",
    "sqlite_autoindex_memories_1",
)


def _collect_salvage_rowids(conn: Any) -> list[int]:
    """Collect ``memories`` rowids by scanning a secondary INDEX btree.

    Walking an index avoids the corrupt table-leaf pages that abort a plain
    ``SELECT * FROM memories``. Falls back to a direct rowid scan when no index
    is usable (e.g. a healthy DB or index-only corruption).
    """
    for idx in _SALVAGE_INDEXES:
        rowids: list[int] = []
        try:
            cur = conn.execute(f"SELECT rowid FROM memories INDEXED BY {idx}")  # noqa: S608 - allowlisted index name
            while True:
                try:
                    row = cur.fetchone()
                except sqlite3.DatabaseError:
                    break
                if row is None:
                    break
                rowids.append(row[0])
            if rowids:
                return rowids
        except sqlite3.DatabaseError:
            continue
    rowids = []
    with contextlib.suppress(sqlite3.DatabaseError):
        for row in conn.execute("SELECT rowid FROM memories"):
            rowids.append(row[0])
    return rowids


def _attempt_primary_salvage(
    backup_path: Path,
    *,
    dbapi: Any,
) -> tuple[bool, list[Any]]:
    """Robustly salvage ``memories`` rows from a (possibly corrupt) backup.

    Walks rowids via a secondary index and fetches each row individually,
    skipping the rows that live on corrupt leaf pages. This recovers the
    maximum readable set instead of the prior behavior, which aborted at the
    first corrupt page and salvaged ZERO rows (the 2026-05-20 data-loss path).

    Returns ``(primary_failed, rows)`` — ``primary_failed`` is True only when
    nothing at all could be read.
    """
    try:
        old_conn = _connection_connect(
            backup_path,
            dbapi=dbapi,
            timeout=15.0,
            check_same_thread=True,
        )
    except sqlite3.DatabaseError:
        return True, []
    try:
        rowids = _collect_salvage_rowids(old_conn)
        rows: list[Any] = []
        page_failures = 0
        for rid in rowids:
            try:
                row = old_conn.execute("SELECT * FROM memories WHERE rowid=?", (rid,)).fetchone()
            except sqlite3.DatabaseError:
                page_failures += 1
                continue
            if row is not None:
                rows.append(row)
        if not rows:
            # No index path worked; last-ditch plain scan (healthy DBs).
            with contextlib.suppress(sqlite3.DatabaseError):
                rows = list(old_conn.execute("SELECT * FROM memories").fetchall())
        if page_failures:
            logger.warning(
                "db_salvage_partial",
                db=str(backup_path),
                salvaged=len(rows),
                page_failures=page_failures,
            )
        return (not rows), rows
    except sqlite3.DatabaseError:
        return True, []
    finally:
        with contextlib.suppress(sqlite3.Error):
            old_conn.close()


def _open_recovered_conn(
    db_path: Path,
    *,
    dbapi: Any,
) -> Any:
    """Open the post-recovery DB with WAL + ensure_schema."""
    prepare_db_file_mode(db_path)
    new_conn = _connection_connect(
        db_path,
        dbapi=dbapi,
        timeout=30.0,
        check_same_thread=False,
    )
    # Match the hardened open profile (busy_timeout + WAL + journal_size_limit)
    # so a recovered connection is configured identically to a normal open.
    apply_open_pragmas(new_conn)
    harden_db_file_mode(db_path)
    ensure_schema(new_conn)
    return new_conn


def _restore_rows(new_conn: Any, rows: list[Any], *, db_path: Path) -> None:
    """Project salvaged rows through ENTRY_COLUMNS allowlist + INSERT OR IGNORE."""
    if not rows:
        return
    raw_cols = list(rows[0].keys())
    safe_indices = [i for i, c in enumerate(raw_cols) if c in ENTRY_COLUMNS]
    safe_cols = [raw_cols[i] for i in safe_indices]
    dropped = [c for c in raw_cols if c not in ENTRY_COLUMNS]
    if dropped:
        logger.warning("db_recovery_dropped_unknown_columns", columns=dropped, db=str(db_path))
    if not safe_cols:
        return
    placeholders = ", ".join(["?"] * len(safe_cols))
    cols_sql = ", ".join(safe_cols)
    insert_sql = f"INSERT OR IGNORE INTO memories ({cols_sql}) VALUES ({placeholders})"  # noqa: S608
    failed = 0
    for row in rows:
        try:
            row_values = tuple(row)
            new_conn.execute(insert_sql, tuple(row_values[i] for i in safe_indices))
        except sqlite3.Error:
            failed += 1
    # The raw INSERT bypasses the write path, so re-derive the anchor and tag indexes before
    # committing (PRD-CORE-332 FR01, F2).
    rebuild_anchor_postings(new_conn, trigger="salvage_restore")
    rebuild_memory_tags_postings(new_conn.cursor(), trigger="salvage_restore")
    new_conn.commit()
    if failed:
        # Surface partial salvage: without this the caller logs rows_salvaged =
        # len(rows) (the backup-scan count), overstating the rows actually
        # committed and hiding data loss from an operator watching db_recovered.
        logger.warning(
            "db_recovery_insert_failures",
            db=str(db_path),
            attempted=len(rows),
            failed=failed,
        )


def _cleanup_strict_refuse(new_conn: Any, db_path: Path) -> None:
    """Close fresh conn + delete db_path + WAL/SHM sidecars on strict-refuse path."""
    if new_conn is None:
        return
    with contextlib.suppress(sqlite3.Error):
        new_conn.close()
    with contextlib.suppress(OSError):
        db_path.unlink(missing_ok=True)
    clear_journal_sidecars(db_path)


def recover_db(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    recovery_policy: Literal["strict", "empty_ok"] = "strict",
    corrupt_backup_keep: int = 5,
    rebuild_from_cold: bool = True,
) -> Any:
    """Recover from a corrupt database by salvaging rows into a fresh DB.

    Holds the store's ``recover`` op from the rotation to the restored rows
    (PRD-CORE-306): another process's open refuses it with ``StoreBusyError``,
    nothing moved. The caller holds no op on the store (an op never upgrades), and
    this process's other connections to it must close within the op's wait. The
    returned connection's ``open`` hold, taken inside the scope, outlives it.
    """
    with store_access(db_path, "recover"):
        _backend = _backend_corrupt_backup_helpers()
        write_recovery_marker(db_path)  # before the rotation: a kill from here on is resumed by the next open
        backup_path = _backend._rotate_corrupt_backup(db_path)
        _backend._prune_corrupt_backups(db_path.parent, keep_n=corrupt_backup_keep)

        write_sentinel(db_path, backup_path)
        return _salvage_into(
            db_path,
            backup_path,
            dbapi=dbapi,
            recovery_policy=recovery_policy,
            rebuild_from_cold=rebuild_from_cold,
            owns_store=True,  # the rotation emptied the path: any store there is one this call creates
        )


def resume_interrupted_recovery(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    recovery_policy: Literal["strict", "empty_ok"] = "strict",
    rebuild_from_cold: bool = True,
) -> Any:
    """Finish a recovery killed between its rotation and its restored rows (B71-133 (a)); None when none is due.

    Its marker names the rotated store's inode, which the backup kept. A store still at that inode was never
    rotated: the marker is dropped and the caller opens it as usual (a corrupt one is recovered again then).
    Otherwise the salvage runs again from that backup into the store now at the path, whatever it holds
    (``INSERT OR IGNORE``: rows restored before the kill, or written since, are kept). A store already at the
    path is never removed, even by a strict refusal: that leaves it, the backup and the marker in place.
    """
    if read_recovery_marker(db_path) is None:
        return None
    with store_access(db_path, "recover"):
        identity = read_recovery_marker(db_path)  # again under the op: another process may have finished it
        backup = find_rotated_backup(db_path, identity) if identity != _db_identity(db_path) else None
        if backup is None:
            if identity not in (None, _db_identity(db_path)):
                logger.error("db_recovery_resume_backup_missing", db=str(db_path))
            clear_recovery_marker(db_path)
            return None
        logger.warning("db_recovery_resumed", db=str(db_path), backup=str(backup))
        return _salvage_into(
            db_path,
            backup,
            dbapi=dbapi,
            recovery_policy=recovery_policy,
            rebuild_from_cold=rebuild_from_cold,
            owns_store=not db_path.exists(),  # checked under the op: nothing else can create it meanwhile
        )


def _salvage_into(
    db_path: Path,
    backup_path: Path,
    *,
    dbapi: Any,
    recovery_policy: Literal["strict", "empty_ok"],
    rebuild_from_cold: bool,
    owns_store: bool,
) -> Any:
    """Salvage *backup_path*'s rows into the store at *db_path* (the caller holds the ``recover`` op).

    *owns_store*: no store was at the path when the caller started, so a strict refusal may delete the one this
    call created. False (a resume into an existing store) never deletes it: its rows are the result so far.
    """
    salvage_primary_failed, rows = _attempt_primary_salvage(backup_path, dbapi=dbapi)

    salvage_cli_failed = False
    cli_used = False
    if not rows:
        cli_used = True
        rows = _backend_corrupt_backup_helpers()._salvage_via_recover_cli(backup_path, dbapi=dbapi)
        salvage_cli_failed = not rows

    recovered_rows = len(rows)

    try:
        backup_size = backup_path.stat().st_size
    except OSError:
        backup_size = 0

    strict_refuse = not rows and backup_size > _PAGE_SIZE and recovery_policy == "strict"
    cold_rebuild_attempted = False
    cold_rebuild_rows = 0
    new_conn: Any = None

    if strict_refuse and rebuild_from_cold:
        logger.debug(
            "cold_rebuild_gate_evaluated",
            policy=recovery_policy,
            knob=True,
            recovered_rows=recovered_rows,
            decision="run",
        )
        from trw_memory.storage._cold_rebuild import rebuild_from_cold as _rebuild_fn

        new_conn = _open_recovered_conn(db_path, dbapi=dbapi)
        rebuild_base = _resolve_cold_rebuild_base_safe(db_path)
        try:
            cold_rebuild_rows = _rebuild_fn(rebuild_base, new_conn)
        except Exception:
            logger.exception("cold_rebuild_failed", db=str(db_path), base_dir=str(rebuild_base))
            cold_rebuild_rows = 0
        cold_rebuild_attempted = True
        if cold_rebuild_rows > 0:
            strict_refuse = False
    elif strict_refuse:
        logger.debug(
            "cold_rebuild_gate_evaluated",
            policy=recovery_policy,
            knob=False,
            recovered_rows=recovered_rows,
            decision="skip_knob_off",
        )
    else:
        logger.debug(
            "cold_rebuild_gate_evaluated",
            policy=recovery_policy,
            knob=rebuild_from_cold,
            recovered_rows=recovered_rows,
            decision="skip_gate_not_met",
        )

    if strict_refuse and not owns_store:
        # Rows the store already holds (restored or rebuilt before a kill, or written since) are a recovered store.
        new_conn = new_conn if new_conn is not None else _open_recovered_conn(db_path, dbapi=dbapi)
        strict_refuse = new_conn.execute("SELECT NOT EXISTS (SELECT 1 FROM memories)").fetchone()[0] == 1
    if strict_refuse and owns_store:
        _cleanup_strict_refuse(new_conn, db_path)
        clear_recovery_marker(db_path)  # concluded: the caller records the verdict
    elif strict_refuse:  # the existing store, the backup and the marker all stay
        with contextlib.suppress(sqlite3.Error):
            new_conn.close()
    if strict_refuse:
        logger.error(
            "db_recovery_refused_strict",
            action="refuse_empty_fallback",
            db_path=str(db_path),
            backup_path=str(backup_path),
            backup_size_bytes=backup_size,
            salvage_primary_failed=salvage_primary_failed,
            salvage_cli_failed=salvage_cli_failed,
            cold_rebuild_attempted=cold_rebuild_attempted,
            cold_rebuild_rows=cold_rebuild_rows,
        )
        raise CorruptDatabaseUnsalvageableError(
            "database disk image is malformed and salvage yielded 0 rows",
            backup_path=str(backup_path),
        )

    if new_conn is None:
        new_conn = _open_recovered_conn(db_path, dbapi=dbapi)

    _restore_rows(new_conn, rows, db_path=db_path)
    clear_recovery_marker(db_path)

    if cold_rebuild_attempted and cold_rebuild_rows > 0:
        logger.warning(
            "db_recovered",
            db=str(db_path),
            backup=str(backup_path),
            rows_salvaged=recovered_rows,
            rebuilt_from_cold=cold_rebuild_rows,
            source="cold_rebuild",
        )
    elif cli_used and rows:
        logger.warning(
            "db_recovered_via_cli",
            db=str(db_path),
            backup=str(backup_path),
            rows_salvaged=recovered_rows,
            source="sqlite3_cli_dump",
        )
    else:
        logger.warning(
            "db_recovered",
            db=str(db_path),
            backup=str(backup_path),
            rows_salvaged=recovered_rows,
        )
    return new_conn
