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
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import structlog

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.storage._connection import (
    apply_open_pragmas,
)
from trw_memory.storage._connection import (
    connect as _connection_connect,
)
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage._shared import ENTRY_COLUMNS
from trw_memory.storage._stale_handle_detector import write_sentinel

logger = structlog.get_logger(__name__)

_PAGE_SIZE = 4096
_RECOVERY_STATE_SUFFIX = ".recovery.json"


@dataclass(frozen=True)
class RecoveryPreflight:
    """Bounded open-time recovery classification."""

    classification: Literal["fast_open", "degraded_open_with_background_recovery", "hard_fail"]
    reason: str
    db_size_bytes: int
    state_path: str
    persisted_status: str = ""


def recovery_state_path(db_path: Path) -> Path:
    """Return the sidecar path used for persisted recovery state."""
    return db_path.with_name(f"{db_path.name}{_RECOVERY_STATE_SUFFIX}")


def write_recovery_state(db_path: Path, *, status: str, reason: str, db_size_bytes: int) -> None:
    """Persist additive recovery state for future bounded-open decisions."""
    state_path = recovery_state_path(db_path)
    payload = {
        "status": status,
        "reason": reason,
        "db_size_bytes": db_size_bytes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with contextlib.suppress(OSError):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(state_path.parent),
            prefix=f".{state_path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(state_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise


def _read_persisted_recovery_status(state_path: Path) -> str:
    """Read the advisory recovery-state sidecar status, fail-closed to ``""``.

    The sidecar is *advisory*: an absent, unreadable, non-UTF-8, malformed,
    non-object, or non-string-status file must never break bounded startup
    classification. Returns the ``status`` string only when the sidecar holds a
    JSON object whose ``status`` field is itself a string; otherwise ``""``.

    Reads raw bytes and decodes explicitly so non-UTF-8 content surfaces as a
    caught ``UnicodeDecodeError`` (a ``ValueError`` subclass that escapes
    ``suppress(OSError, JSONDecodeError)``) rather than crashing the caller.

    Diagnostics are content-free — ``reason``/``error_type`` only. The filesystem
    path, raw bytes, and decoded payload are never logged, so a poisoned or
    secret-bearing sidecar cannot leak through startup logs.
    """
    try:
        raw_bytes = state_path.read_bytes()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        logger.debug("recovery_state_unreadable", reason="read_failed", error_type=type(exc).__name__)
        return ""

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("recovery_state_unreadable", reason="non_utf8")
        return ""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("recovery_state_unreadable", reason="malformed_json")
        return ""

    if not isinstance(parsed, dict):
        logger.debug("recovery_state_unreadable", reason="non_object_json")
        return ""

    status = parsed.get("status", "")
    if not isinstance(status, str):
        logger.debug("recovery_state_unreadable", reason="non_string_status")
        return ""
    return status


def classify_recovery_preflight(db_path: Path, *, inline_max_bytes: int) -> RecoveryPreflight:
    """Classify whether startup can recover inline or should degrade/fail early."""
    state_path = recovery_state_path(db_path)
    db_size_bytes = 0
    with contextlib.suppress(OSError):
        db_size_bytes = db_path.stat().st_size

    persisted_status = _read_persisted_recovery_status(state_path)

    if persisted_status == "hard_fail":
        return RecoveryPreflight(
            classification="hard_fail",
            reason="previous_recovery_hard_fail",
            db_size_bytes=db_size_bytes,
            state_path=str(state_path),
            persisted_status=persisted_status,
        )

    if inline_max_bytes > 0 and db_size_bytes > inline_max_bytes:
        return RecoveryPreflight(
            classification="degraded_open_with_background_recovery",
            reason="db_exceeds_inline_recovery_budget",
            db_size_bytes=db_size_bytes,
            state_path=str(state_path),
            persisted_status=persisted_status,
        )

    if persisted_status in {"pending", "running", "degraded_open_with_background_recovery"}:
        return RecoveryPreflight(
            classification="degraded_open_with_background_recovery",
            reason="recovery_already_pending",
            db_size_bytes=db_size_bytes,
            state_path=str(state_path),
            persisted_status=persisted_status,
        )

    return RecoveryPreflight(
        classification="fast_open",
        reason="within_inline_recovery_budget",
        db_size_bytes=db_size_bytes,
        state_path=str(state_path),
        persisted_status=persisted_status,
    )


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
    sqlcipher_key_hex: str | None,
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
            sqlcipher_key_hex=sqlcipher_key_hex,
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
    # Match the hardened open profile (busy_timeout + WAL + journal_size_limit)
    # so a recovered connection is configured identically to a normal open.
    apply_open_pragmas(new_conn)
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
