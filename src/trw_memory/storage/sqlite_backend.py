"""SQLite + sqlite-vec storage backend.

Primary storage implementation for trw-memory.  Stores all :class:`MemoryEntry`
fields in a plain ``memories`` table, with an optional ``vec_memories`` virtual
table (sqlite-vec) for vector search.

Graceful degradation: if sqlite-vec is not installed, all metadata operations
continue to work; only :meth:`upsert_vector` and :meth:`search_vectors` become
no-ops.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import sqlite3
import struct
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

import structlog

from trw_memory.exceptions import (
    CorruptDatabaseUnsalvageableError,
    EncryptionUnavailableError,
    StaleConnectionError,
    StorageError,
)
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._row_mapper import entry_to_row, row_to_entry
from trw_memory.storage._schema import ensure_schema, ensure_vec_table
from trw_memory.storage._shared import (
    DICT_FIELDS,
    ENTRY_COLUMNS,
    IMMUTABLE_FIELDS,
    LIST_FIELDS,
    serialize_update_value,
    validate_update_fields,
)
from trw_memory.storage._stale_handle_detector import StaleHandleDetector, write_sentinel
from trw_memory.storage._utf8_validator import validate_utf8_fields
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync.delta import DeltaTracker

try:
    import sqlite_vec
except ImportError:  # pragma: no cover — optional dep
    sqlite_vec = None

logger = structlog.get_logger(__name__)
SQLCIPHER_REQUIRED_MESSAGE = (
    "SQLCipher is required when memory_encryption_enabled=True. Install with: pip install trw-memory[encryption]"
)
SQLCIPHER_CIPHER = "aes-256-cbc"
SQLCIPHER_CIPHER_PAGE_SIZE = 4096
SQLCIPHER_KDF_ITER = 256000


def _resolve_cold_rebuild_base(db_path: Path) -> Path:
    """Return the base directory whose ``memory/cold`` subtree should rebuild.

    ``rebuild_from_cold(base_dir, conn)`` intentionally reads
    ``base_dir / "memory" / "cold"``. Two layouts are live:

    - standalone ``trw-memory`` tests/CLI: ``<base>/memory.db`` and
      ``<base>/memory/cold``
    - ``trw-mcp`` runtime: ``<trw_dir>/memory/memory.db`` and
      ``<trw_dir>/memory/cold``

    The second layout was the 2026-04-28 incident: using ``db_path.parent``
    made recovery look under ``<trw_dir>/memory/memory/cold`` and rebuild
    zero rows. Prefer the production-shaped parent when it exists; otherwise
    preserve the standalone default.
    """
    standalone_base = db_path.parent
    candidates = [standalone_base]
    if db_path.parent.name == "memory" or (db_path.parent / "cold").exists():
        trw_dir_base = db_path.parent.parent
        if trw_dir_base != standalone_base:
            candidates.append(trw_dir_base)

    candidate_counts: list[tuple[Path, int]] = []
    for candidate in candidates:
        cold_dir = candidate / "memory" / "cold"
        yaml_count = sum(1 for _ in cold_dir.rglob("*.yaml")) if cold_dir.is_dir() else 0
        candidate_counts.append((candidate, yaml_count))

    non_empty = [item for item in candidate_counts if item[1] > 0]
    if non_empty:
        selected_base, selected_count = max(non_empty, key=lambda item: item[1])
    elif db_path.parent.name == "memory" and db_path.name == "memory.db":
        selected_base, selected_count = candidate_counts[-1]
    else:
        selected_base, selected_count = candidate_counts[0]

    logger.info(
        "cold_rebuild_base_selected",
        db_path=str(db_path),
        selected_base_dir=str(selected_base),
        selected_yaml_count=selected_count,
        candidates=[
            {
                "base_dir": str(candidate),
                "cold_dir": str(candidate / "memory" / "cold"),
                "yaml_count": yaml_count,
            }
            for candidate, yaml_count in candidate_counts
        ],
    )
    return selected_base


def _import_sqlcipher_driver() -> Any:
    """Return a SQLCipher DB-API module or raise the standard startup error."""
    for module_name in ("sqlcipher3.dbapi2", "pysqlcipher3.dbapi2"):
        with contextlib.suppress(ImportError):
            return importlib.import_module(module_name)
    raise EncryptionUnavailableError(SQLCIPHER_REQUIRED_MESSAGE)


def _apply_sqlcipher_pragmas(conn: Any) -> None:
    """Apply the explicit SQLCipher settings required by the PRD."""
    conn.execute(f"PRAGMA cipher = '{SQLCIPHER_CIPHER}'")
    conn.execute(f"PRAGMA cipher_page_size = {SQLCIPHER_CIPHER_PAGE_SIZE}")
    conn.execute(f"PRAGMA kdf_iter = {SQLCIPHER_KDF_ITER}")


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

_COLUMNS = ENTRY_COLUMNS
_SELECT_COLUMNS_SQL = ", ".join("expires_at AS expires" if column == "expires_at" else column for column in _COLUMNS)
_INSERT_COLUMNS_SQL = ", ".join(_COLUMNS)

# Allowlist for UPDATE: all columns except immutable ones.
_VALID_UPDATE_COLUMNS: frozenset[str] = (frozenset(_COLUMNS) - IMMUTABLE_FIELDS) | frozenset({"expires"})


# PRD-CORE-138/139: corrupt-backup rotation + CLI salvage extracted to
# _corrupt_backup.py (PRD-DIST-245 Phase 1 batch 81). Re-exports preserve
# the public API surface.
from trw_memory.storage._corrupt_backup import (
    _LEGACY_CORRUPT_NAMES as _LEGACY_CORRUPT_NAMES,
    _TIMESTAMPED_BACKUP_RE as _TIMESTAMPED_BACKUP_RE,
    prune_corrupt_backups as _prune_corrupt_backups_impl,
    rotate_corrupt_backup as _rotate_corrupt_backup_impl,
    salvage_via_recover_cli as _salvage_via_recover_cli_impl,
)
# Connection-management helpers extracted to _connection.py (PRD-DIST-245
# batch 82). Re-exports preserve the public API surface.
from trw_memory.storage._connection import (
    connect as _connection_connect,
    db_has_data as _connection_db_has_data,
    open_and_configure as _connection_open_and_configure,
    open_without_integrity_check as _connection_open_without_integrity_check,
)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class SQLiteBackend(StorageBackend):
    """SQLite-backed storage with optional sqlite-vec vector search.

    Args:
        db_path: Path to the SQLite database file.  Parent directories are
            created automatically.
        dim: Embedding dimension for the vec_memories virtual table.
            Defaults to 384 (all-MiniLM-L6-v2).
    """

    _DEFAULT_DIM: ClassVar[int] = 384

    def __init__(
        self,
        db_path: Path,
        dim: int = 384,
        *,
        sqlcipher_key_hex: str | None = None,
        recovery_policy: Literal["strict", "empty_ok"] = "strict",
        corrupt_backup_keep: int = 5,
        rebuild_from_cold: bool = True,
        integrity_check_interval_minutes: int = 0,
        concurrent_writer_warn_threshold: int = 4,
    ) -> None:
        self._db_path = db_path
        self._dim = dim
        self._vec_available = False
        self._lock = threading.Lock()
        # PRD-FIX-088 FR02: Open-transaction depth counter.
        # When >0, mutating methods (``update`` etc.) skip per-row ``commit()``
        # so a caller-controlled outer transaction can batch many writes.
        # Re-entrant by depth; only the outermost ``transaction()`` issues
        # ``BEGIN IMMEDIATE`` / ``COMMIT``.
        self._skip_commit_depth: int = 0
        self._dbapi: Any = _import_sqlcipher_driver() if sqlcipher_key_hex is not None else sqlite3
        self._sqlcipher_key_hex = sqlcipher_key_hex
        self._recovery_policy = recovery_policy
        self._corrupt_backup_keep = corrupt_backup_keep
        self._rebuild_from_cold = rebuild_from_cold
        self._integrity_check_interval_minutes = integrity_check_interval_minutes
        self._concurrent_writer_warn_threshold = concurrent_writer_warn_threshold
        # PRD-INFRA-063 (B2) + PRD-INFRA-064 (B3) hooks — populated after open.
        self._integrity_scheduler: Any = None
        self._writer_registry: Any = None

        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.recovered = False
        self.integrity_warning = False
        # P2 — per-row UTF-8 quarantine counter (incremented on each skipped row)
        self.quarantine_count_utf8: int = 0
        # P3 — reconnect counter (incremented on each stale-handle reopen)
        self.reconnect_count: int = 0
        try:
            if sqlcipher_key_hex is None:
                self._conn = self._open_and_configure(db_path)
            else:
                self._conn = self._open_and_configure(
                    db_path,
                    dbapi=self._dbapi,
                    sqlcipher_key_hex=sqlcipher_key_hex,
                )
        except sqlite3.DatabaseError:
            # quick_check failed — but this can be transient (WAL contention,
            # concurrent MCP server access). Check if DB actually has data
            # before destroying it with auto-recovery.
            if self._db_has_data(db_path, dbapi=self._dbapi, sqlcipher_key_hex=sqlcipher_key_hex):
                logger.warning(
                    "db_integrity_check_failed_but_has_data",
                    db=str(db_path),
                    action="open_anyway",
                    hint="quick_check failed but DB has rows — likely transient WAL contention, not corruption",
                )
                self._conn = self._open_without_integrity_check(
                    db_path,
                    dbapi=self._dbapi,
                    sqlcipher_key_hex=sqlcipher_key_hex,
                )
                self.integrity_warning = True
            else:
                logger.exception("db_corrupt_detected", db=str(db_path), action="auto_recover")
                self._conn = self.recover_db(
                    db_path,
                    dbapi=self._dbapi,
                    sqlcipher_key_hex=sqlcipher_key_hex,
                    recovery_policy=self._recovery_policy,
                    corrupt_backup_keep=self._corrupt_backup_keep,
                    rebuild_from_cold=self._rebuild_from_cold,
                )
                self.recovered = True

        ensure_schema(self._conn)

        # P3 — stale-handle detector (belt + suspenders: inode + sentinel).
        # Only meaningful for real files; :memory: paths get a no-op detector.
        self._stale_detector = StaleHandleDetector(db_path)

        vec_module = sqlite_vec
        if vec_module is not None:
            try:
                # AttributeError: Python built without SQLITE_ENABLE_LOAD_EXTENSION
                # (common on macOS system Python and some python.org builds).
                self._conn.enable_load_extension(True)
                vec_module.load(self._conn)
                self._conn.enable_load_extension(False)
                ensure_vec_table(self._conn, self._dim)
                self._vec_available = True
                logger.debug("sqlite_vec_loaded", db=str(db_path))
            except (sqlite3.Error, OSError, AttributeError) as exc:
                self._vec_available = False
                logger.warning(
                    "sqlite_vec_load_failed",
                    db=str(db_path),
                    reason=type(exc).__name__,
                    detail=str(exc),
                    hint="Python lacks SQLite load_extension support; vector search disabled, BM25 still works",
                )
        else:
            logger.debug("sqlite_vec_unavailable", reason="not_installed")

        # PRD-INFRA-064 (B3): multi-writer advisory registry — advisory ONLY,
        # registry failures MUST NOT raise (advisory-only invariant).
        try:
            from trw_memory.storage._writer_registry import WriterRegistry

            self._writer_registry = WriterRegistry(
                db_path,
                warn_threshold=self._concurrent_writer_warn_threshold,
            )
            self._writer_registry.register()
        except Exception:  # justified: advisory-only invariant — never block open
            logger.debug("writer_registry_unavailable", db=str(db_path), exc_info=True)
            self._writer_registry = None

        # PRD-INFRA-063 (B2): periodic integrity scheduler — opt-in via
        # interval > 0. Dedicated read-only connection; observability only.
        try:
            from trw_memory.storage._integrity_scheduler import IntegrityScheduler

            self._integrity_scheduler = IntegrityScheduler(
                db_path,
                interval_minutes=self._integrity_check_interval_minutes,
                on_regression=self._handle_integrity_regression,
            )
            self._integrity_scheduler.start()
        except Exception:  # justified: observability scheduler must not block open
            logger.debug("integrity_scheduler_unavailable", db=str(db_path), exc_info=True)
            self._integrity_scheduler = None

    def _handle_integrity_regression(self, _db_path: Path, _detail: str) -> None:
        """Callback invoked by :class:`IntegrityScheduler` on a failed probe.

        Sets :attr:`integrity_warning` so external code can observe the
        regression (same flag used by the transient-WAL-contention path).
        """
        self.integrity_warning = True

    # ------------------------------------------------------------------
    # P3 — Stale-handle detection + reconnect
    # ------------------------------------------------------------------

    def _reconnect(self) -> None:
        """Close the current connection and reopen against the current DB file.

        Called when :meth:`_ensure_connection_fresh` detects a stale handle.

        Raises:
            StaleConnectionError: If reopening the connection fails.
        """
        try:
            with contextlib.suppress(sqlite3.Error):
                self._conn.close()
            if self._sqlcipher_key_hex is not None:
                self._conn = self._open_and_configure(
                    self._db_path,
                    dbapi=self._dbapi,
                    sqlcipher_key_hex=self._sqlcipher_key_hex,
                )
            else:
                self._conn = self._open_and_configure(self._db_path)
            ensure_schema(self._conn)
            self._stale_detector.reset()
            self.reconnect_count += 1
            logger.info(
                "memory_stale_handle_reconnected",
                db_path=str(self._db_path),
                reconnect_count=self.reconnect_count,
            )
        except Exception as exc:
            raise StaleConnectionError(
                f"Failed to reopen stale connection to {self._db_path}: {exc}",
                path=str(self._db_path),
            ) from exc

    def _ensure_connection_fresh(self) -> None:
        """Check whether the connection has gone stale; reconnect if so.

        This is a best-effort probe — NOT a consistency guarantee.  The check
        is cached for ``TRW_MEMORY_STALE_HANDLE_CHECK_SECS`` (default 1s) so
        the steady-state cost on a warm kernel page cache is effectively zero.

        Raises:
            StaleConnectionError: If a stale handle is detected and reconnect
                fails.
        """
        if self._stale_detector.is_stale():
            self._reconnect()

    # ------------------------------------------------------------------
    # Integrity & recovery
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(
        db_path: Path,
        *,
        dbapi: Any,
        timeout: float,
        check_same_thread: bool,
        cached_statements: int | None = None,
        sqlcipher_key_hex: str | None = None,
    ) -> Any:
        """Delegate to ``_connection.connect`` (PRD-DIST-245 batch 82)."""
        return _connection_connect(
            db_path,
            dbapi=dbapi,
            timeout=timeout,
            check_same_thread=check_same_thread,
            cached_statements=cached_statements,
            sqlcipher_key_hex=sqlcipher_key_hex,
        )

    @staticmethod
    def _open_and_configure(
        db_path: Path,
        *,
        dbapi: Any = sqlite3,
        sqlcipher_key_hex: str | None = None,
    ) -> Any:
        """Delegate to ``_connection.open_and_configure``."""
        return _connection_open_and_configure(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)

    @staticmethod
    def _open_without_integrity_check(
        db_path: Path,
        *,
        dbapi: Any = sqlite3,
        sqlcipher_key_hex: str | None = None,
    ) -> Any:
        """Delegate to ``_connection.open_without_integrity_check``."""
        return _connection_open_without_integrity_check(
            db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex
        )

    @staticmethod
    def _db_has_data(
        db_path: Path,
        *,
        dbapi: Any = sqlite3,
        sqlcipher_key_hex: str | None = None,
    ) -> bool:
        """Delegate to ``_connection.db_has_data``."""
        return _connection_db_has_data(db_path, dbapi=dbapi, sqlcipher_key_hex=sqlcipher_key_hex)

    def _run_integrity_check(self) -> bool:
        """Run PRAGMA quick_check and return True if the database is healthy."""
        try:
            rows = self._conn.execute("PRAGMA quick_check").fetchall()
            return len(rows) == 1 and rows[0][0] == "ok"
        except sqlite3.DatabaseError:
            return False

    @staticmethod
    def _salvage_via_recover_cli(backup_path: Path, dbapi: Any = sqlite3) -> list[Any]:
        """Delegate to ``_corrupt_backup.salvage_via_recover_cli`` (PRD-CORE-138 FR04)."""
        return _salvage_via_recover_cli_impl(backup_path, dbapi=dbapi)

    @staticmethod
    def _rotate_corrupt_backup(db_path: Path) -> Path:
        """Delegate to ``_corrupt_backup.rotate_corrupt_backup`` (PRD-CORE-139 FR01)."""
        return _rotate_corrupt_backup_impl(db_path)

    @staticmethod
    def _prune_corrupt_backups(parent: Path, keep_n: int) -> None:
        """Delegate to ``_corrupt_backup.prune_corrupt_backups`` (PRD-CORE-139 FR03/FR04)."""
        _prune_corrupt_backups_impl(parent, keep_n)

    @staticmethod
    def recover_db(
        db_path: Path,
        *,
        dbapi: Any = sqlite3,
        sqlcipher_key_hex: str | None = None,
        recovery_policy: Literal["strict", "empty_ok"] = "strict",
        corrupt_backup_keep: int = 5,
        rebuild_from_cold: bool = True,
    ) -> Any:
        """Recover from a corrupt database by salvaging rows into a fresh DB.

        1. Move corrupt file to ``memory.db.corrupt.<UTC-ISO>.bak``
           (PRD-CORE-139 FR01) and prune old timestamped backups beyond
           ``corrupt_backup_keep`` (FR03/FR04). Legacy ``.corrupt.bak`` and
           ``.corrupt.bak.1`` files are preserved.
        2. Try salvage via in-process ``SELECT * FROM memories``.
        3. On ``DatabaseError`` (e.g., destroyed ``sqlite_master``), fall back
           to the ``sqlite3 .recover`` CLI (PRD-CORE-138 FR04).
        4. PRD-CORE-140 FR03: when both salvage paths yield zero rows, the
           backup is non-empty, ``recovery_policy == "strict"``, AND
           ``rebuild_from_cold`` is True, run the cold-tier rebuild into the
           freshly opened DB. If rebuild yields >0 rows, the strict-refusal
           raise is suppressed.
        5. If the salvage-and-rebuild combined still yielded zero rows AND
           the backup is non-empty AND ``recovery_policy == "strict"`` (default),
           raise :class:`CorruptDatabaseUnsalvageableError` preserving the backup.
        6. Under ``recovery_policy == "empty_ok"`` preserve legacy behavior:
           create a fresh empty DB and log ``rows_salvaged=0``.

        Returns:
            A new :class:`sqlite3.Connection` to the recovered database.

        Raises:
            CorruptDatabaseUnsalvageableError: Under ``strict`` policy when
                salvage + cold-rebuild both yield zero rows from a non-empty
                ``.corrupt.bak``.
        """
        # PRD-CORE-139 FR01/FR03/FR04: timestamped rotation + filename-based pruning.
        backup_path = SQLiteBackend._rotate_corrupt_backup(db_path)
        SQLiteBackend._prune_corrupt_backups(db_path.parent, keep_n=corrupt_backup_keep)
        # Also remove stale WAL/SHM files for the corrupt DB
        for suffix in (".db-wal", ".db-shm"):
            wal = db_path.with_name(db_path.name.replace(".db", suffix))
            with contextlib.suppress(OSError):
                wal.unlink()

        # P3 — write cross-process sentinel so peer consumers detect stale FDs.
        # Written early (before the fresh DB exists) so any peer doing a precheck
        # between now and the fresh-DB creation will see the sentinel and
        # schedule a reconnect on their next read.
        write_sentinel(db_path, backup_path)

        salvage_primary_failed = False
        rows: list[Any] = []
        try:
            # Attempt to salvage rows from the corrupt database.
            old_conn = SQLiteBackend._connect(
                backup_path,
                dbapi=dbapi,
                timeout=5.0,
                check_same_thread=True,
                sqlcipher_key_hex=sqlcipher_key_hex,
            )
            try:
                rows = list(old_conn.execute("SELECT * FROM memories").fetchall())
            except sqlite3.DatabaseError:
                salvage_primary_failed = True
                rows = []
            old_conn.close()
        except sqlite3.DatabaseError:
            salvage_primary_failed = True
            rows = []

        # FR04: second salvage path via sqlite3 CLI .recover when primary yielded no rows.
        salvage_cli_failed = False
        cli_used = False
        if not rows:
            cli_used = True
            rows = SQLiteBackend._salvage_via_recover_cli(backup_path, dbapi=dbapi)
            salvage_cli_failed = not rows

        recovered_rows = len(rows)

        # FR03: strict-mode refusal on non-empty backup with zero salvaged rows.
        # Page-size heuristic: SQLite default page is 4096 bytes. Backups smaller
        # than one page cannot contain real row data — treat them as genuinely
        # empty and fall through to the legacy empty-DB creation path.
        page_size = 4096
        try:
            backup_size = backup_path.stat().st_size
        except OSError:
            backup_size = 0

        # PRD-CORE-140 FR03: gated cold-tier rebuild before strict-refusal raise.
        # Gate: strict policy AND knob enabled AND salvage yielded zero rows AND
        # backup is non-empty. When all four hold, open the new DB early, run the
        # rebuild, and only re-raise the strict error if rebuild also yielded zero.
        strict_refuse = not rows and backup_size > page_size and recovery_policy == "strict"
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
            # Lazy import: avoids circular-import at module-load time.
            from trw_memory.storage._cold_rebuild import rebuild_from_cold as _rebuild_fn

            new_conn = SQLiteBackend._connect(
                db_path,
                dbapi=dbapi,
                timeout=30.0,
                check_same_thread=False,
                sqlcipher_key_hex=sqlcipher_key_hex,
            )
            new_conn.execute("PRAGMA journal_mode=WAL")
            new_conn.execute("PRAGMA synchronous=NORMAL")
            ensure_schema(new_conn)
            rebuild_base = _resolve_cold_rebuild_base(db_path)
            try:
                cold_rebuild_rows = _rebuild_fn(rebuild_base, new_conn)
            except Exception:
                logger.exception(
                    "cold_rebuild_failed",
                    db=str(db_path),
                    base_dir=str(rebuild_base),
                )
                cold_rebuild_rows = 0
            cold_rebuild_attempted = True
            if cold_rebuild_rows > 0:
                # Strict gate satisfied by rebuild — suppress the raise.
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
            # Close the fresh conn opened for an unsuccessful rebuild (if any)
            # so we don't leak a handle on the raise path. Also delete the
            # just-created DB file and its WAL sidecars so the caller observes
            # the same post-state as pre-PRD-CORE-140 (backup preserved; no
            # stub DB at db_path).
            if new_conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    new_conn.close()
                with contextlib.suppress(OSError):
                    db_path.unlink(missing_ok=True)
                for suffix in (".db-wal", ".db-shm"):
                    sidecar = db_path.with_name(db_path.name.replace(".db", suffix))
                    with contextlib.suppress(OSError):
                        sidecar.unlink(missing_ok=True)
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
            new_conn = SQLiteBackend._connect(
                db_path,
                dbapi=dbapi,
                timeout=30.0,
                check_same_thread=False,
                sqlcipher_key_hex=sqlcipher_key_hex,
            )
            new_conn.execute("PRAGMA journal_mode=WAL")
            new_conn.execute("PRAGMA synchronous=NORMAL")
            ensure_schema(new_conn)

        if rows:
            # Security: recovered row keys come from a corrupt/attacker-
            # influenced DB dump. Splicing them into SQL without validation
            # is an injection vector. Keep only columns in the current schema
            # allowlist; project each row by positional index (works with
            # both sqlite3.Row and the iterable row objects used in tests).
            raw_cols = list(rows[0].keys())
            safe_indices = [i for i, c in enumerate(raw_cols) if c in ENTRY_COLUMNS]
            safe_cols = [raw_cols[i] for i in safe_indices]
            dropped = [c for c in raw_cols if c not in ENTRY_COLUMNS]
            if dropped:
                logger.warning(
                    "db_recovery_dropped_unknown_columns",
                    columns=dropped,
                    db=str(db_path),
                )
            if safe_cols:
                placeholders = ", ".join(["?"] * len(safe_cols))
                cols_sql = ", ".join(safe_cols)
                insert_sql = f"INSERT OR IGNORE INTO memories ({cols_sql}) VALUES ({placeholders})"  # noqa: S608
                for row in rows:
                    with contextlib.suppress(sqlite3.Error):
                        row_values = tuple(row)
                        new_conn.execute(insert_sql, tuple(row_values[i] for i in safe_indices))
                new_conn.commit()

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

    @staticmethod
    def check_integrity(
        db_path: Path,
        *,
        dbapi: Any = sqlite3,
        sqlcipher_key_hex: str | None = None,
    ) -> dict[str, object]:
        """Public utility: check database integrity without opening a full backend.

        Returns:
            Dict with ``ok`` (bool), ``detail`` (str), and ``db_path``.
        """
        try:
            conn = SQLiteBackend._connect(
                db_path,
                dbapi=dbapi,
                timeout=5.0,
                check_same_thread=True,
                sqlcipher_key_hex=sqlcipher_key_hex,
            )
            rows = conn.execute("PRAGMA quick_check").fetchall()
            conn.close()
            healthy = len(rows) == 1 and rows[0][0] == "ok"
            return {"ok": healthy, "detail": rows[0][0] if rows else "empty", "db_path": str(db_path)}
        except sqlite3.DatabaseError as exc:
            return {"ok": False, "detail": str(exc), "db_path": str(db_path)}

    # ------------------------------------------------------------------
    # Public property
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SQLiteBackend(db_path={self._db_path!r}, vec={self._vec_available})"

    @property
    def vec_available(self) -> bool:
        """``True`` when sqlite-vec is loaded and the virtual table exists."""
        return self._vec_available

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # P2 — Resilient row materialisation
    # ------------------------------------------------------------------

    def _fetch_rows_resilient(
        self,
        cursor: Any,
        *,
        table: str = "memories",
    ) -> list[MemoryEntry]:
        """Iterate a cursor row-by-row, quarantining bad-UTF-8 rows.

        The sqlite3.OperationalError("Could not decode to UTF-8 column ...")
        is raised by the C extension during row fetch.  We use a fallback
        connection with ``text_factory=bytes`` to read raw rows when the
        primary cursor fails, then decode each column individually, replacing
        undecidable bytes with the Unicode replacement character and marking
        the row as quarantined.

        Strategy:
          1. Attempt ``fetchall()`` on the primary cursor.
          2. On UTF-8 decode failure, fall back to a secondary bytes-mode
             connection that re-executes the same SQL and reads all rows as
             bytes objects.  For each bytes row we attempt str decode; rows
             that cannot be decoded at all are quarantined.

        Per-row errors:
          - Increment ``self.quarantine_count_utf8``.
          - Emit a structlog WARN event with action="memory_row_utf8_quarantined".
          - Skip the row from the return list.

        Non-UTF-8 errors (e.g. schema errors) are re-raised so callers can
        convert them to ``StorageError`` as before.

        Args:
            cursor: Active sqlite3 cursor after ``execute()``.
            table: Table name for logging context.

        Returns:
            List of successfully materialised MemoryEntry objects.
        """
        # Fast path: all rows are clean UTF-8.
        try:
            raw_rows = cursor.fetchall()
        except (sqlite3.OperationalError, UnicodeDecodeError) as exc:
            err_str = str(exc)
            if "UTF-8" not in err_str and "decode" not in err_str.lower() and not isinstance(exc, UnicodeDecodeError):
                raise  # non-UTF-8 error — propagate

            # Slow path: re-read rows via a bytes-mode connection.
            return self._fetch_rows_via_bytes_fallback(cursor, table=table)

        results: list[MemoryEntry] = []
        for idx, raw_row in enumerate(raw_rows):
            try:
                entry = row_to_entry(tuple(raw_row))
                results.append(entry)
            except (UnicodeDecodeError, UnicodeEncodeError) as exc:
                self.quarantine_count_utf8 += 1
                row_id: str | None = None
                with contextlib.suppress(Exception):
                    row_id = str(raw_row[0])
                logger.warning(
                    "db_bad_utf8_row_quarantined",
                    action="memory_row_utf8_quarantined",
                    row_id=row_id,
                    column="detail",
                    db_path=str(self._db_path),
                    table=table,
                    row_index=idx,
                    error=str(exc),
                )
        return results

    def _fetch_rows_via_bytes_fallback(
        self,
        cursor: Any,
        *,
        table: str = "memories",
    ) -> list[MemoryEntry]:
        """Fallback: re-execute via bytes-mode connection to isolate bad rows.

        Opens a second connection with ``text_factory=bytes`` so we get raw
        bytes per column.  We then attempt UTF-8 decode per column; rows where
        any column cannot decode at all are quarantined (counted + logged) and
        omitted from the result.

        This is the slow path — invoked only when the primary cursor fails.
        """
        # Reconstruct the SQL and parameters from the cursor description.
        # Since we can't re-read them, use the table scan approach: read all
        # ids via a bytes connection and then re-fetch individually.
        # Simpler: re-query the full table via bytes connection and filter.
        results: list[MemoryEntry] = []
        try:
            raw_conn = self._dbapi.connect(str(self._db_path))
            raw_conn.text_factory = bytes
            raw_rows = raw_conn.execute(
                f"SELECT {_SELECT_COLUMNS_SQL} FROM {table} ORDER BY updated_at DESC"  # noqa: S608
            ).fetchall()
            raw_conn.close()
        except Exception:
            # Can't even open the fallback — return empty so callers don't crash.
            logger.warning(
                "db_utf8_fallback_failed",
                action="memory_row_utf8_quarantined",
                db_path=str(self._db_path),
                table=table,
            )
            return []

        for idx, raw_row in enumerate(raw_rows):
            # Decode each bytes value to str; skip row if any column fails entirely.
            decoded: list[object] = []
            row_bad = False
            bad_col: str | None = None
            for col_idx, val in enumerate(raw_row):
                if isinstance(val, bytes):
                    try:
                        decoded.append(val.decode("utf-8", errors="strict"))
                    except (UnicodeDecodeError, ValueError):
                        row_bad = True
                        with contextlib.suppress(Exception):
                            bad_col = cursor.description[col_idx][0] if cursor.description else None
                        break
                else:
                    decoded.append(val)

            if row_bad:
                self.quarantine_count_utf8 += 1
                row_id_str: str | None = None
                with contextlib.suppress(Exception):
                    first = raw_row[0]
                    row_id_str = first.decode("utf-8", errors="replace") if isinstance(first, bytes) else str(first)
                logger.warning(
                    "db_bad_utf8_row_quarantined",
                    action="memory_row_utf8_quarantined",
                    row_id=row_id_str,
                    column=bad_col or "unknown",
                    db_path=str(self._db_path),
                    table=table,
                    row_index=idx,
                )
                continue

            try:
                entry = row_to_entry(tuple(decoded))
                results.append(entry)
            except Exception:
                # Unexpected mapping failure — quarantine silently.
                self.quarantine_count_utf8 += 1
                logger.warning(
                    "db_bad_utf8_row_quarantined",
                    action="memory_row_utf8_quarantined",
                    row_id=None,
                    column="row_to_entry",
                    db_path=str(self._db_path),
                    table=table,
                    row_index=idx,
                )

        return results

    @staticmethod
    def _build_filter_clause(
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        min_importance: float = 0.0,
    ) -> tuple[str, list[object]]:
        """Build a WHERE clause fragment from common filter parameters.

        Returns:
            A ``(where_sql, params)`` tuple.  *where_sql* is ``"1"`` when no
            filters are active, otherwise the clauses joined with ``AND``.
        """
        clauses: list[str] = []
        params: list[object] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)

        if min_importance > 0.0:
            clauses.append("importance >= ?")
            params.append(min_importance)

        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)

        where_sql = " AND ".join(clauses) if clauses else "1"
        return where_sql, params

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        """INSERT OR REPLACE the entry into the memories table.

        Args:
            entry: Memory entry to persist.

        Raises:
            Utf8ValidationError: If any TEXT-column field contains invalid UTF-8.
            StorageError: If the write fails.
        """
        # P1 — write-time UTF-8 validation (prevention layer).
        # Validate before computing sync_hash so a bad entry never mutates state.
        validate_utf8_fields(
            {
                "id": entry.id,
                "content": entry.content,
                "detail": entry.detail,
                "nudge_line": entry.nudge_line,
                "type": entry.type,
                "namespace": entry.namespace,
                "source": entry.source,
                "source_identity": entry.source_identity,
                "client_profile": entry.client_profile,
                "model_id": entry.model_id,
                "consolidated_into": entry.consolidated_into,
                "remote_id": entry.remote_id,
                "expires_at": entry.expires,
                "task_type": entry.task_type,
                "phase_origin": entry.phase_origin,
                "team_origin": entry.team_origin,
                "outcome_correlation": entry.outcome_correlation,
                "sync_hash": entry.sync_hash,
            }
        )

        # Auto dirty-mark for sync pipeline (PRD-INFRA-051)
        entry.sync_seq = (entry.sync_seq or 0) + 1
        entry.sync_hash = DeltaTracker.compute_sync_hash(entry)
        entry.last_synced_at = None

        placeholders = ", ".join(["?"] * len(_COLUMNS))
        sql = f"INSERT OR REPLACE INTO memories ({_INSERT_COLUMNS_SQL}) VALUES ({placeholders})"  # noqa: S608 — _INSERT_COLUMNS_SQL is a static constant (no user input); values are parameterized
        try:
            with self._lock:
                self._conn.execute(sql, entry_to_row(entry))
                self._conn.commit()
            logger.debug("memory_stored", entry_id=entry.id)
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise StorageError(
                f"Failed to store entry {entry.id}: {exc}",
                path=str(self._db_path),
            ) from exc

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve an entry by id.

        Args:
            entry_id: Target entry id.

        Returns:
            :class:`MemoryEntry` or ``None`` if not found.

        Raises:
            StorageError: If the query fails.
        """
        sql = f"SELECT {_SELECT_COLUMNS_SQL} FROM memories WHERE id = ?"  # noqa: S608 — _SELECT_COLUMNS_SQL is a static constant; entry_id is a parameterized ?
        try:
            with self._lock:
                row = self._conn.execute(sql, (entry_id,)).fetchone()
            if row is None:
                return None
            return row_to_entry(tuple(row))
        except (sqlite3.Error, ValueError, KeyError) as exc:
            raise StorageError(
                f"Failed to get entry {entry_id}: {exc}",
                path=str(self._db_path),
            ) from exc

    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
        """Apply a partial update to an existing entry.

        Args:
            entry_id: Target entry id.
            **fields: Fields to update.

        Returns:
            Updated :class:`MemoryEntry` or ``None`` if not found.

        Raises:
            StorageError: If the update fails.
        """
        if not fields:
            return self.get(entry_id)

        existing = self.get(entry_id)
        if existing is None:
            return None

        try:
            # Serialise list/dict fields to JSON for SQLite
            set_parts: list[str] = []
            values: list[object] = []

            # Auto-set updated_at when not explicitly provided
            field_dict: dict[str, object] = dict(fields)
            if "updated_at" not in field_dict:
                field_dict["updated_at"] = datetime.now(timezone.utc)

            try:
                validate_update_fields(field_dict, _VALID_UPDATE_COLUMNS)
            except ValueError as ve:
                raise StorageError(
                    f"Invalid update field: {ve.args[0]!r}",
                    path=str(self._db_path),
                ) from None

            # Mark dirty for sync pipeline (PRD-INFRA-051)
            # Skip when the caller is explicitly setting sync bookkeeping fields
            # (for example ``mark_synced()`` only updates ``last_synced_at``).
            if not {"sync_seq", "sync_hash", "last_synced_at"} & field_dict.keys():
                updated_entry = existing.model_copy(deep=True)
                for key, val in field_dict.items():
                    setattr(updated_entry, key, val)
                next_sync_seq = (existing.sync_seq or 0) + 1
                field_dict["sync_seq"] = next_sync_seq
                updated_entry.sync_seq = next_sync_seq
                field_dict["sync_hash"] = DeltaTracker.compute_sync_hash(updated_entry)
                field_dict["last_synced_at"] = None

            for key, val in field_dict.items():
                sql_key = "expires_at" if key == "expires" else key
                set_parts.append(f"{sql_key} = ?")
                normalised = serialize_update_value(key, val)
                # SQLite needs JSON strings for list/dict columns
                if (key in LIST_FIELDS and isinstance(normalised, list)) or (
                    key in DICT_FIELDS and isinstance(normalised, dict)
                ):
                    values.append(json.dumps(normalised))
                else:
                    values.append(normalised)

            values.append(entry_id)
            sql = f"UPDATE memories SET {', '.join(set_parts)} WHERE id = ?"  # noqa: S608 — column names come from the validated UPDATABLE_FIELDS whitelist; values are parameterized
            with self._lock:
                self._conn.execute(sql, values)
                if self._skip_commit_depth == 0:
                    self._conn.commit()
            return self.get(entry_id)
        except StorageError:
            raise
        except (sqlite3.Error, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise StorageError(
                f"Failed to update entry {entry_id}: {exc}",
                path=str(self._db_path),
            ) from exc

    @contextlib.contextmanager
    def transaction(self) -> Iterator[SQLiteBackend]:
        """Context manager that batches multiple writes into one SQLite transaction.

        PRD-FIX-088 FR02: When a caller wraps a series of ``update()`` (or other
        mutating) calls in ``with backend.transaction():``, the per-call
        ``commit()`` is suppressed and a single ``BEGIN IMMEDIATE`` / ``COMMIT``
        bracket is used instead.  This collapses N implicit transactions to one
        explicit transaction, which is cheaper and avoids holding the SQLite
        write-lock across N round-trips.

        Re-entrant by depth: nested ``transaction()`` calls do not re-issue
        ``BEGIN``; only the outermost issues ``BEGIN IMMEDIATE`` / ``COMMIT``.
        Inner exceptions still propagate; the outermost handler issues
        ``ROLLBACK`` and re-raises.

        Yields:
            ``self`` for fluent chaining (callers normally ignore the value).
        """
        is_outer = self._skip_commit_depth == 0
        if is_outer:
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
        self._skip_commit_depth += 1
        try:
            yield self
            if is_outer:
                with self._lock:
                    self._conn.commit()
        except BaseException:
            if is_outer:
                try:
                    with self._lock:
                        self._conn.rollback()
                except sqlite3.Error:
                    logger.exception("transaction_rollback_failed", db=str(self._db_path))
            raise
        finally:
            self._skip_commit_depth -= 1

    def increment_session_counts(self, entry_ids: list[str], *, updated_at: datetime | None = None) -> int:
        """Increment ``session_count`` for multiple entries in one transaction."""
        if not entry_ids:
            return 0

        now = updated_at or datetime.now(timezone.utc)
        values = [(now.isoformat(), entry_id) for entry_id in entry_ids]

        try:
            sql = """
                UPDATE memories
                SET session_count = COALESCE(session_count, 0) + 1,
                    updated_at = ?,
                    sync_seq = COALESCE(sync_seq, 0) + 1,
                    last_synced_at = NULL
                WHERE id = ?
            """
            with self._lock:
                before = self._conn.total_changes
                self._conn.executemany(sql, values)
                self._conn.commit()
                return int(self._conn.total_changes - before)
        except sqlite3.Error as exc:
            raise StorageError(
                f"Failed to increment session counts: {exc}",
                path=str(self._db_path),
            ) from exc

    def increment_access_counts(self, entry_ids: list[str], *, accessed_at: datetime | None = None) -> int:
        """Increment ``access_count`` and ``last_accessed_at`` for entries in one transaction."""
        if not entry_ids:
            return 0

        now = accessed_at or datetime.now(timezone.utc)
        values = [(now.isoformat(), now.isoformat(), entry_id) for entry_id in entry_ids]

        try:
            sql = """
                UPDATE memories
                SET access_count = COALESCE(access_count, 0) + 1,
                    last_accessed_at = ?,
                    updated_at = ?,
                    sync_seq = COALESCE(sync_seq, 0) + 1,
                    last_synced_at = NULL
                WHERE id = ?
            """
            with self._lock:
                before = self._conn.total_changes
                self._conn.executemany(sql, values)
                self._conn.commit()
                return int(self._conn.total_changes - before)
        except sqlite3.Error as exc:
            raise StorageError(
                f"Failed to increment access counts: {exc}",
                path=str(self._db_path),
            ) from exc

    def delete(self, entry_id: str) -> bool:
        """Remove an entry from the memories table (and vec_index if present).

        Args:
            entry_id: Target entry id.

        Returns:
            ``True`` if deleted, ``False`` if not found.

        Raises:
            StorageError: If the deletion fails.
        """
        try:
            with self._lock:
                cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
                deleted = cursor.rowcount > 0
                if deleted and self._vec_available:
                    self._delete_vector(entry_id)
                self._conn.commit()
            logger.debug("memory_deleted", entry_id=entry_id, existed=deleted)
            return bool(deleted)
        except sqlite3.Error as exc:
            raise StorageError(
                f"Failed to delete entry {entry_id}: {exc}",
                path=str(self._db_path),
            ) from exc

    def _delete_vector(self, entry_id: str) -> None:
        """Remove the vector row for *entry_id* (no-op if absent)."""
        row = self._conn.execute("SELECT rowid FROM vec_index WHERE entry_id = ?", (entry_id,)).fetchone()
        if row is None:
            return
        rowid: int = row[0]
        self._conn.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM vec_index WHERE rowid = ?", (rowid,))

    def delete_vector(self, entry_id: str) -> bool:
        """Public vector-row deletion helper for warm-tier maintenance."""
        if not self._vec_available:
            return False
        with self._lock:
            before = self._conn.total_changes
            self._delete_vector(entry_id)
            self._conn.commit()
            return bool(self._conn.total_changes > before)

    def vector_exists(self, entry_id: str) -> bool:
        """Return whether vec_index currently contains *entry_id*."""
        if not self._vec_available:
            return False
        row = self._conn.execute("SELECT 1 FROM vec_index WHERE entry_id = ?", (entry_id,)).fetchone()
        return row is not None

    def existing_vector_ids(self) -> set[str]:
        """Return the set of entry IDs that currently have a stored vector.

        Empty set when sqlite-vec is unavailable. Single-query bulk lookup so
        callers (e.g. backfill loops) can skip already-embedded entries
        without paying per-entry round-trips.
        """
        if not self._vec_available:
            return set()
        try:
            with self._lock:
                rows = self._conn.execute("SELECT entry_id FROM vec_index").fetchall()
        except sqlite3.Error:
            logger.debug("existing_vector_ids_query_failed", exc_info=True)
            return set()
        return {str(r[0]) for r in rows}

    def search(
        self,
        query: str,
        *,
        top_k: int = 25,
        tags: list[str] | None = None,
        status: MemoryStatus | None = None,
        min_importance: float = 0.0,
        namespace: str | None = None,
    ) -> list[MemoryEntry]:
        """Keyword LIKE search on content + detail + tags with filters.

        Args:
            query: Free-text search term (case-insensitive substring match).
            top_k: Maximum results to return.
            tags: If provided, entries must contain ALL listed tags.
            status: If provided, restrict to this status.
            min_importance: Lower bound on importance (inclusive).
            namespace: If provided, restrict to this namespace.

        Returns:
            Up to *top_k* matching :class:`MemoryEntry` objects.

        Raises:
            StorageError: If the query fails.
        """
        # Keyword match — LIKE on id, content, detail, tags JSON
        # Escape LIKE metacharacters to prevent unintended wildcard expansion
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_term = f"%{escaped}%"
        like_clause = (
            "(id LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\' "
            "OR detail LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')"
        )
        like_params: list[object] = [like_term, like_term, like_term, like_term]

        filter_sql, filter_params = self._build_filter_clause(
            status=status,
            namespace=namespace,
            min_importance=min_importance,
        )

        # Combine LIKE clause with filter clauses
        if filter_sql == "1":
            where_sql = like_clause
        else:
            where_sql = f"{like_clause} AND {filter_sql}"
        params: list[object] = like_params + filter_params

        sql = (
            f"SELECT {_SELECT_COLUMNS_SQL} FROM memories WHERE {where_sql} "  # noqa: S608 — _SELECT_COLUMNS_SQL and where_sql are built from static constants and ? placeholders only
            f"ORDER BY importance DESC, updated_at DESC LIMIT ?"
        )
        params.append(top_k)

        # P3 — stale-handle precheck
        self._ensure_connection_fresh()

        try:
            with self._lock:
                cursor = self._conn.execute(sql, params)
                # P2 — resilient cursor iteration
                results = self._fetch_rows_resilient(cursor)

            # Post-filter by tags (all-of semantics)
            if tags:
                required = set(tags)
                results = [e for e in results if required.issubset(set(e.tags))]

            return results[:top_k]
        except (sqlite3.Error, ValueError, KeyError) as exc:
            raise StorageError(
                f"Failed to search memories: {exc}",
                path=str(self._db_path),
            ) from exc

    def count(self, namespace: str | None = None) -> int:
        """Return the number of stored entries.

        Args:
            namespace: If provided, count only this namespace.

        Returns:
            Entry count.

        Raises:
            StorageError: If the count query fails.
        """
        try:
            with self._lock:
                if namespace is not None:
                    row = self._conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE namespace = ?",
                        (namespace,),
                    ).fetchone()
                else:
                    row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error as exc:
            raise StorageError(
                f"Failed to count memories: {exc}",
                path=str(self._db_path),
            ) from exc

    def entries_with_assertions(self) -> list[MemoryEntry]:
        """Return all entries that have non-empty assertions (PRD-CORE-086 FR07).

        Used by ``trw_session_start`` to compute assertion health summary
        from cached ``last_result`` fields without running verification I/O.

        Returns:
            List of MemoryEntry objects that have at least one assertion.
        """
        # P3 — stale-handle precheck
        self._ensure_connection_fresh()

        try:
            with self._lock:
                cursor = self._conn.execute(
                    "SELECT * FROM memories WHERE assertions IS NOT NULL AND assertions != '[]'",
                )
                # P2 — resilient cursor iteration
                return self._fetch_rows_resilient(cursor)
        except sqlite3.Error as exc:
            logger.debug("entries_with_assertions_query_failed", exc_info=True)
            raise StorageError(
                f"Failed to query entries with assertions: {exc}",
                path=str(self._db_path),
            ) from exc

    def count_with_assertions(self) -> list[MemoryEntry]:
        """Backward-compatible alias for PRD-CORE-086 FR07 traceability."""
        return self.entries_with_assertions()

    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Return entries with optional filters, ordered by updated_at desc.

        Args:
            status: If provided, filter by this status.
            namespace: If provided, filter by this namespace.
            limit: Maximum entries to return.

        Returns:
            Up to *limit* matching :class:`MemoryEntry` objects.

        Raises:
            StorageError: If the query fails.
        """
        where_sql, params = self._build_filter_clause(
            status=status,
            namespace=namespace,
        )
        sql = (
            f"SELECT {_SELECT_COLUMNS_SQL} FROM memories WHERE {where_sql} "  # noqa: S608 — _SELECT_COLUMNS_SQL and where_sql are built from static constants and ? placeholders only
            f"ORDER BY updated_at DESC LIMIT ?"
        )
        params.append(limit)

        # P3 — stale-handle precheck (cached; cheap on warm path)
        self._ensure_connection_fresh()

        try:
            with self._lock:
                cursor = self._conn.execute(sql, params)
                # P2 — resilient cursor iteration: skip bad-UTF-8 rows
                return self._fetch_rows_resilient(cursor)
        except (sqlite3.Error, ValueError, KeyError) as exc:
            raise StorageError(
                f"Failed to list entries: {exc}",
                path=str(self._db_path),
            ) from exc

    # ------------------------------------------------------------------
    # Namespace operations
    # ------------------------------------------------------------------

    def list_namespaces(self) -> list[str]:
        """Return all distinct namespaces that have stored entries.

        Returns:
            Sorted list of namespace strings.

        Raises:
            StorageError: If the query fails.
        """
        try:
            with self._lock:
                rows = self._conn.execute("SELECT DISTINCT namespace FROM memories ORDER BY namespace").fetchall()
            return [str(row[0]) for row in rows]
        except sqlite3.Error as exc:
            raise StorageError(
                f"Failed to list namespaces: {exc}",
                path=str(self._db_path),
            ) from exc

    def delete_by_namespace(self, namespace: str) -> int:
        """Delete all entries in a namespace.

        Args:
            namespace: Namespace to clear.

        Returns:
            Number of entries deleted.

        Raises:
            StorageError: If the deletion fails.
        """
        try:
            with self._lock:
                cursor = self._conn.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
                deleted = cursor.rowcount
                self._conn.commit()
            logger.debug(
                "namespace_deleted",
                namespace=namespace,
                entries_deleted=deleted,
            )
            return int(deleted)
        except sqlite3.Error as exc:
            raise StorageError(
                f"Failed to delete namespace {namespace!r}: {exc}",
                path=str(self._db_path),
            ) from exc

    def close(self) -> None:
        """Close the database connection."""
        # PRD-INFRA-063: stop the integrity scheduler BEFORE closing the
        # connection so the background thread unwinds cleanly.
        if self._integrity_scheduler is not None:
            with contextlib.suppress(Exception):
                self._integrity_scheduler.stop(timeout=2.0)
            self._integrity_scheduler = None
        # PRD-INFRA-064: remove our writer-registry lockfile.
        if self._writer_registry is not None:
            with contextlib.suppress(Exception):
                self._writer_registry.close()
            self._writer_registry = None
        with contextlib.suppress(sqlite3.Error):
            self._conn.close()
        logger.debug("sqlite_backend_closed", db=str(self._db_path))

    def __enter__(self) -> SQLiteBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Vector operations (sqlite-vec)
    # ------------------------------------------------------------------

    def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
        """Insert or update a vector in vec_memories.

        No-op when sqlite-vec is not available.

        Args:
            entry_id: The corresponding memory entry id.
            embedding: Dense float vector (length must equal *dim*).
        """
        if not self._vec_available:
            return

        emb_bytes = struct.pack(f"{self._dim}f", *embedding)

        with self._lock:
            # Ensure a vec_index row exists and get its rowid
            self._conn.execute("INSERT OR IGNORE INTO vec_index(entry_id) VALUES(?)", (entry_id,))
            row = self._conn.execute("SELECT rowid FROM vec_index WHERE entry_id = ?", (entry_id,)).fetchone()
            rowid: int = row[0]

            # Delete old vector then insert fresh (idempotent upsert)
            self._conn.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
            self._conn.execute(
                "INSERT INTO vec_memories(rowid, embedding) VALUES(?, ?)",
                (rowid, emb_bytes),
            )
            self._conn.commit()
        logger.debug("vector_upserted", entry_id=entry_id)

    def search_vectors(self, query_embedding: list[float], top_k: int = 25) -> list[tuple[str, float]]:
        """KNN search in vec_memories.

        No-op (returns empty list) when sqlite-vec is not available.

        Args:
            query_embedding: Query vector (length must equal *dim*).
            top_k: Number of nearest neighbours to return.

        Returns:
            List of ``(entry_id, distance)`` pairs sorted by distance ascending.
        """
        if not self._vec_available:
            return []

        query_bytes = struct.pack(f"{self._dim}f", *query_embedding)
        try:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT vi.entry_id, vm.distance
                    FROM vec_memories vm
                    JOIN vec_index vi ON vm.rowid = vi.rowid
                    WHERE vm.embedding MATCH ? AND k = ?
                    ORDER BY vm.distance
                    """,
                    (query_bytes, top_k),
                ).fetchall()
            return [(str(r[0]), float(r[1])) for r in rows]
        except sqlite3.Error:
            logger.debug("vector_search_error", exc_info=True)
            return []

    def get_stored_embeddings(self, entry_ids: list[str]) -> dict[str, list[float]]:
        """Load stored vectors for the requested entry IDs.

        Returns an empty dict when sqlite-vec is unavailable or when none of the
        requested IDs currently have persisted vectors.
        """
        if not self._vec_available or not entry_ids:
            return {}

        placeholders = ", ".join(["?"] * len(entry_ids))
        sql = f"""
            SELECT vi.entry_id, vm.embedding
            FROM vec_memories vm
            JOIN vec_index vi ON vm.rowid = vi.rowid
            WHERE vi.entry_id IN ({placeholders})
        """  # noqa: S608 — placeholder count is derived from entry_ids length only; values bound separately
        try:
            with self._lock:
                rows = self._conn.execute(sql, entry_ids).fetchall()
        except sqlite3.Error:
            logger.debug("vector_load_error", exc_info=True)
            return {}

        embeddings: dict[str, list[float]] = {}
        for row in rows:
            raw = row[1]
            if raw is None:
                continue
            # sqlite-vec stores float arrays as packed blobs; unpack them here so
            # the retrieval layer receives the same list[float] shape as embed().
            blob = bytes(raw)
            if len(blob) % 4 != 0:
                logger.debug(
                    "vector_load_skipped_invalid_blob",
                    entry_id=str(row[0]),
                    blob_len=len(blob),
                )
                continue
            dim = len(blob) // 4
            embeddings[str(row[0])] = list(struct.unpack(f"{dim}f", blob))
        return embeddings
