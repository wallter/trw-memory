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

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.storage._connection import connect as _connection_connect
from trw_memory.storage._schema import ensure_schema
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


def _attempt_primary_salvage(
    backup_path: Path,
    *,
    dbapi: Any,
    sqlcipher_key_hex: str | None,
) -> tuple[bool, list[Any]]:
    """Try in-process ``SELECT * FROM memories`` salvage."""
    try:
        old_conn = _connection_connect(
            backup_path,
            dbapi=dbapi,
            timeout=5.0,
            check_same_thread=True,
            sqlcipher_key_hex=sqlcipher_key_hex,
        )
        try:
            rows = list(old_conn.execute("SELECT * FROM memories").fetchall())
            return False, rows
        except sqlite3.DatabaseError:
            return True, []
        finally:
            old_conn.close()
    except sqlite3.DatabaseError:
        return True, []


def _open_recovered_conn(
    db_path: Path,
    *,
    dbapi: Any,
    sqlcipher_key_hex: str | None,
) -> Any:
    """Open the post-recovery DB with WAL + ensure_schema."""
    new_conn = _connection_connect(
        db_path,
        dbapi=dbapi,
        timeout=30.0,
        check_same_thread=False,
        sqlcipher_key_hex=sqlcipher_key_hex,
    )
    new_conn.execute("PRAGMA journal_mode=WAL")
    new_conn.execute("PRAGMA synchronous=NORMAL")
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
    for row in rows:
        with contextlib.suppress(sqlite3.Error):
            row_values = tuple(row)
            new_conn.execute(insert_sql, tuple(row_values[i] for i in safe_indices))
    new_conn.commit()


def _cleanup_strict_refuse(new_conn: Any, db_path: Path) -> None:
    """Close fresh conn + delete db_path + WAL/SHM sidecars on strict-refuse path."""
    if new_conn is None:
        return
    with contextlib.suppress(sqlite3.Error):
        new_conn.close()
    with contextlib.suppress(OSError):
        db_path.unlink(missing_ok=True)
    for suffix in (".db-wal", ".db-shm"):
        sidecar = db_path.with_name(db_path.name.replace(".db", suffix))
        with contextlib.suppress(OSError):
            sidecar.unlink(missing_ok=True)


def recover_db(
    db_path: Path,
    *,
    dbapi: Any = sqlite3,
    sqlcipher_key_hex: str | None = None,
    recovery_policy: Literal["strict", "empty_ok"] = "strict",
    corrupt_backup_keep: int = 5,
    rebuild_from_cold: bool = True,
) -> Any:
    """Recover from a corrupt database by salvaging rows into a fresh DB."""
    _backend = _backend_corrupt_backup_helpers()
    backup_path = _backend._rotate_corrupt_backup(db_path)
    _backend._prune_corrupt_backups(db_path.parent, keep_n=corrupt_backup_keep)
    for suffix in (".db-wal", ".db-shm"):
        wal = db_path.with_name(db_path.name.replace(".db", suffix))
        with contextlib.suppress(OSError):
            wal.unlink()

    write_sentinel(db_path, backup_path)

    salvage_primary_failed, rows = _attempt_primary_salvage(
        backup_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex
    )

    salvage_cli_failed = False
    cli_used = False
    if not rows:
        cli_used = True
        rows = _backend._salvage_via_recover_cli(backup_path, dbapi=dbapi)
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

        new_conn = _open_recovered_conn(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)
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

    if strict_refuse:
        _cleanup_strict_refuse(new_conn, db_path)
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
        new_conn = _open_recovered_conn(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)

    _restore_rows(new_conn, rows, db_path=db_path)

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
