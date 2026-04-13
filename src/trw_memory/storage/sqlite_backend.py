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
import re
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

import structlog

from trw_memory.exceptions import (
    CorruptDatabaseUnsalvageableError,
    EncryptionUnavailableError,
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
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync.delta import DeltaTracker

try:
    import sqlite_vec
except ImportError:  # pragma: no cover — optional dep
    sqlite_vec = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)
SQLCIPHER_REQUIRED_MESSAGE = (
    "SQLCipher is required when memory_encryption_enabled=True. Install with: pip install trw-memory[encryption]"
)
SQLCIPHER_CIPHER = "aes-256-cbc"
SQLCIPHER_CIPHER_PAGE_SIZE = 4096
SQLCIPHER_KDF_ITER = 256000


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
_SELECT_COLUMNS_SQL = ", ".join(
    "expires_at AS expires" if column == "expires_at" else column for column in _COLUMNS
)
_INSERT_COLUMNS_SQL = ", ".join(_COLUMNS)

# Allowlist for UPDATE: all columns except immutable ones.
_VALID_UPDATE_COLUMNS: frozenset[str] = (frozenset(_COLUMNS) - IMMUTABLE_FIELDS) | frozenset({"expires"})


# ---------------------------------------------------------------------------
# PRD-CORE-139: timestamped corrupt-backup rotation
# ---------------------------------------------------------------------------
# Matches memory.db.corrupt.<ISO-UTC>.bak with optional -N collision suffix.
# Group 1 is the timestamp used for lexicographic (== chronological) sort.
_TIMESTAMPED_BACKUP_RE: re.Pattern[str] = re.compile(
    r"^memory\.db\.corrupt\.(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)(?:-\d+)?\.bak$"
)
# Pre-PRD-CORE-139 filenames. These are sacred: count against the keep budget
# but are never pruning victims, so forensic evidence captured before the
# rotation change is never silently destroyed on upgrade.
_LEGACY_CORRUPT_NAMES: frozenset[str] = frozenset(
    {"memory.db.corrupt.bak", "memory.db.corrupt.bak.1"}
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
    ) -> None:
        self._db_path = db_path
        self._dim = dim
        self._vec_available = False
        self._lock = threading.Lock()
        self._dbapi: Any = _import_sqlcipher_driver() if sqlcipher_key_hex is not None else sqlite3
        self._recovery_policy = recovery_policy
        self._corrupt_backup_keep = corrupt_backup_keep

        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.recovered = False
        self.integrity_warning = False
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
                )
                self.recovered = True

        ensure_schema(self._conn)

        vec_module = sqlite_vec
        if vec_module is not None:
            try:
                self._conn.enable_load_extension(True)
                vec_module.load(self._conn)
                self._conn.enable_load_extension(False)
                ensure_vec_table(self._conn, self._dim)
                self._vec_available = True
                logger.debug("sqlite_vec_loaded", db=str(db_path))
            except (sqlite3.Error, OSError):
                self._vec_available = False
                logger.debug("sqlite_vec_load_failed", db=str(db_path), exc_info=True)
        else:
            logger.debug("sqlite_vec_unavailable", reason="not_installed")

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
        kwargs: dict[str, object] = {
            "timeout": timeout,
            "check_same_thread": check_same_thread,
        }
        if cached_statements is not None:
            kwargs["cached_statements"] = cached_statements
        conn = dbapi.connect(str(db_path), **kwargs)
        conn.row_factory = sqlite3.Row
        if sqlcipher_key_hex is not None:
            if len(sqlcipher_key_hex) != 64 or any(ch not in "0123456789abcdef" for ch in sqlcipher_key_hex):
                raise ValueError("sqlcipher_key_hex must be a 64-character lowercase hex string")
            conn.execute(f'PRAGMA key = "x\'{sqlcipher_key_hex}\'"')  # noqa: S608
            _apply_sqlcipher_pragmas(conn)
            conn.execute("SELECT count(*) FROM sqlite_master")
        return conn

    @staticmethod
    def _open_and_configure(
        db_path: Path,
        *,
        dbapi: Any = sqlite3,
        sqlcipher_key_hex: str | None = None,
    ) -> Any:
        """Open a connection with WAL mode and run a quick integrity check.

        Retries once on quick_check failure to handle transient WAL contention
        (e.g., MCP server mid-checkpoint while trw-maintain opens the DB).

        Raises:
            sqlite3.DatabaseError: If the database fails integrity check twice.
        """
        import time

        conn = SQLiteBackend._connect(
            db_path,
            dbapi=dbapi,
            timeout=30.0,
            check_same_thread=False,
            cached_statements=0,
            sqlcipher_key_hex=sqlcipher_key_hex,
        )
        # busy_timeout prevents SQLITE_BUSY under multi-process contention.
        conn.execute("PRAGMA busy_timeout = 30000")

        wal_result = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if wal_result and wal_result[0] != "wal":
            logger.warning("wal_mode_not_enabled", got=wal_result[0])
        sync_result = conn.execute("PRAGMA synchronous=NORMAL").fetchone()
        if sync_result and sync_result[0] not in ("1", 1):
            logger.warning("synchronous_normal_not_set", got=sync_result[0] if sync_result else None)

        # Quick integrity probe with retry — transient WAL contention can
        # cause false positives on the first attempt.
        for attempt in range(2):
            rows = conn.execute("PRAGMA quick_check").fetchall()
            if len(rows) == 1 and rows[0][0] == "ok":
                return conn
            if attempt == 0:
                logger.warning(
                    "integrity_check_retry",
                    db=str(db_path),
                    detail=rows[0][0] if rows else "empty",
                )
                time.sleep(1.0)  # Allow WAL checkpoint to complete

        conn.close()
        raise sqlite3.DatabaseError("database disk image is malformed (quick_check failed twice)")

    @staticmethod
    def _open_without_integrity_check(
        db_path: Path,
        *,
        dbapi: Any = sqlite3,
        sqlcipher_key_hex: str | None = None,
    ) -> Any:
        """Open a connection skipping integrity check — used when DB has data
        but quick_check fails (likely transient WAL contention)."""
        conn = SQLiteBackend._connect(
            db_path,
            dbapi=dbapi,
            timeout=30.0,
            check_same_thread=False,
            cached_statements=0,
            sqlcipher_key_hex=sqlcipher_key_hex,
        )
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @staticmethod
    def _db_has_data(
        db_path: Path,
        *,
        dbapi: Any = sqlite3,
        sqlcipher_key_hex: str | None = None,
    ) -> bool:
        """Check if a database file has any rows, without integrity check.

        Used to prevent auto-recovery from destroying a DB that has data
        but fails quick_check due to transient WAL contention.
        """
        try:
            conn = SQLiteBackend._connect(
                db_path,
                dbapi=dbapi,
                timeout=5.0,
                check_same_thread=True,
                sqlcipher_key_hex=sqlcipher_key_hex,
            )
            try:
                count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
                conn.close()
                return bool(count > 0)
            except sqlite3.Error:
                conn.close()
                return False
        except sqlite3.Error:
            return False

    def _run_integrity_check(self) -> bool:
        """Run PRAGMA quick_check and return True if the database is healthy."""
        try:
            rows = self._conn.execute("PRAGMA quick_check").fetchall()
            return len(rows) == 1 and rows[0][0] == "ok"
        except sqlite3.DatabaseError:
            return False

    @staticmethod
    def _salvage_via_recover_cli(backup_path: Path, dbapi: Any = sqlite3) -> list[Any]:
        """Attempt `sqlite3 <backup> .recover` CLI salvage.

        PRD-CORE-138 FR04 — second salvage path for cases where the in-process
        ``SELECT * FROM memories`` raises :class:`sqlite3.DatabaseError` because
        ``sqlite_master`` is damaged but B-tree pages still hold rows.

        Returns an empty list on any failure: CLI absent, non-zero exit,
        ``subprocess.TimeoutExpired``, ``FileNotFoundError``, ``sqlite3.Error``
        during dump load, or empty output. The caller treats an empty list as
        "salvage failed" and consults the recovery policy.
        """
        try:
            completed = subprocess.run(  # noqa: S603 — args passed as list, shell=False
                ["sqlite3", str(backup_path), ".recover"],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return []

        if completed.returncode != 0 or not completed.stdout:
            return []

        dump_sql = completed.stdout.decode("utf-8", errors="replace")
        if not dump_sql.strip():
            return []

        # Load dump into a temp DB and extract rows.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "recover.db"
            try:
                tmp_conn = dbapi.connect(str(tmp_db))
                tmp_conn.row_factory = sqlite3.Row
                try:
                    tmp_conn.executescript(dump_sql)
                    rows = tmp_conn.execute("SELECT * FROM memories").fetchall()
                    return list(rows)
                except sqlite3.Error:
                    return []
                finally:
                    tmp_conn.close()
            except sqlite3.Error:
                return []

    @staticmethod
    def _rotate_corrupt_backup(db_path: Path) -> Path:
        """Move ``db_path`` to ``memory.db.corrupt.<UTC-ISO>.bak`` (PRD-CORE-139 FR01).

        Uses ``datetime.now(timezone.utc)`` at call time. Colons in the time
        portion are replaced with hyphens for Windows/shell safety. On a
        same-second collision with an existing backup (NFR02), appends a
        ``-1``, ``-2``, ... suffix — names remain parseable by
        :data:`_TIMESTAMPED_BACKUP_RE`.

        Returns:
            The new path the corrupt DB was moved to.
        """
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        candidate = db_path.with_name(f"memory.db.corrupt.{ts}.bak")
        i = 1
        while candidate.exists():
            candidate = db_path.with_name(f"memory.db.corrupt.{ts}-{i}.bak")
            i += 1
        shutil.move(str(db_path), str(candidate))
        return candidate

    @staticmethod
    def _prune_corrupt_backups(parent: Path, keep_n: int) -> None:
        """Keep at most ``keep_n`` corruption backup files in ``parent`` (FR03/FR04).

        Pruning candidates are restricted to files matching
        :data:`_TIMESTAMPED_BACKUP_RE`. Order is determined by parsing the
        timestamp out of the filename — never the filesystem mtime, which is
        trampled by ``shutil.move`` and so is not authoritative for move-time.

        Legacy ``memory.db.corrupt.bak`` and ``memory.db.corrupt.bak.1`` files
        count against ``keep_n`` (so total disk usage stays bounded across an
        upgrade) but are NEVER selected for deletion. If the legacy count alone
        exceeds the budget, no deletion occurs and a single WARNING
        (``corrupt_backup_budget_exceeded_legacy_only``) is emitted.

        Deletion failures are swallowed via :func:`contextlib.suppress`
        following the existing style at the WAL/SHM cleanup block.
        """
        timestamped: list[tuple[str, Path]] = []
        legacy_count = 0
        try:
            children = list(parent.iterdir())
        except OSError:
            return
        for child in children:
            name = child.name
            if name in _LEGACY_CORRUPT_NAMES:
                legacy_count += 1
                continue
            match = _TIMESTAMPED_BACKUP_RE.fullmatch(name)
            if match is not None:
                timestamped.append((match.group(1), child))
        # Lexicographic sort over the ISO-8601 prefix == chronological sort.
        timestamped.sort(key=lambda pair: pair[0])
        excess = len(timestamped) + legacy_count - keep_n
        pruned = 0
        while excess > 0 and timestamped:
            _, victim = timestamped.pop(0)
            with contextlib.suppress(OSError):
                victim.unlink()
                pruned += 1
            excess -= 1
        if excess > 0:
            # Only legacy files remain; cannot meet the budget without
            # touching legacy artifacts (FR04 forbids).
            logger.warning(
                "corrupt_backup_budget_exceeded_legacy_only",
                keep=keep_n,
                legacy_count=legacy_count,
            )
        logger.info(
            "corrupt_backup_rotated",
            parent=str(parent),
            keep=keep_n,
            total_timestamped=len(timestamped),
            total_legacy=legacy_count,
            pruned=pruned,
        )

    @staticmethod
    def recover_db(
        db_path: Path,
        *,
        dbapi: Any = sqlite3,
        sqlcipher_key_hex: str | None = None,
        recovery_policy: Literal["strict", "empty_ok"] = "strict",
        corrupt_backup_keep: int = 5,
    ) -> Any:
        """Recover from a corrupt database by salvaging rows into a fresh DB.

        1. Move corrupt file to ``memory.db.corrupt.<UTC-ISO>.bak``
           (PRD-CORE-139 FR01) and prune old timestamped backups beyond
           ``corrupt_backup_keep`` (FR03/FR04). Legacy ``.corrupt.bak`` and
           ``.corrupt.bak.1`` files are preserved.
        2. Try salvage via in-process ``SELECT * FROM memories``.
        3. On ``DatabaseError`` (e.g., destroyed ``sqlite_master``), fall back
           to the ``sqlite3 .recover`` CLI (PRD-CORE-138 FR04).
        4. If both salvage paths yield zero rows AND the backup is non-empty
           AND ``recovery_policy == "strict"`` (default), raise
           :class:`CorruptDatabaseUnsalvageableError` preserving the backup
           (FR03).
        5. Under ``recovery_policy == "empty_ok"`` preserve legacy behavior:
           create a fresh empty DB and log ``rows_salvaged=0`` (FR05).

        Returns:
            A new :class:`sqlite3.Connection` to the recovered database.

        Raises:
            CorruptDatabaseUnsalvageableError: Under ``strict`` policy when
                salvage yields zero rows from a non-empty ``.corrupt.bak``.
        """
        # PRD-CORE-139 FR01/FR03/FR04: timestamped rotation + filename-based pruning.
        backup_path = SQLiteBackend._rotate_corrupt_backup(db_path)
        SQLiteBackend._prune_corrupt_backups(db_path.parent, keep_n=corrupt_backup_keep)
        # Also remove stale WAL/SHM files for the corrupt DB
        for suffix in (".db-wal", ".db-shm"):
            wal = db_path.with_name(db_path.name.replace(".db", suffix))
            with contextlib.suppress(OSError):
                wal.unlink()

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

        if not rows and backup_size > page_size and recovery_policy == "strict":
            logger.error(
                "db_recovery_refused_strict",
                action="refuse_empty_fallback",
                db_path=str(db_path),
                backup_path=str(backup_path),
                backup_size_bytes=backup_size,
                salvage_primary_failed=salvage_primary_failed,
                salvage_cli_failed=salvage_cli_failed,
            )
            raise CorruptDatabaseUnsalvageableError(
                "database disk image is malformed and salvage yielded 0 rows",
                backup_path=str(backup_path),
            )

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
            cols = rows[0].keys()
            placeholders = ", ".join(["?"] * len(cols))
            cols_sql = ", ".join(cols)
            insert_sql = f"INSERT OR IGNORE INTO memories ({cols_sql}) VALUES ({placeholders})"  # noqa: S608
            for row in rows:
                with contextlib.suppress(sqlite3.Error):
                    new_conn.execute(insert_sql, tuple(row))
            new_conn.commit()

        if cli_used and rows:
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
            StorageError: If the write fails.
        """
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
                self._conn.commit()
            return self.get(entry_id)
        except StorageError:
            raise
        except (sqlite3.Error, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise StorageError(
                f"Failed to update entry {entry_id}: {exc}",
                path=str(self._db_path),
            ) from exc

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

        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            results = [row_to_entry(tuple(r)) for r in rows]

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
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE assertions IS NOT NULL AND assertions != '[]'",
                ).fetchall()
            return [row_to_entry(row) for row in rows]
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

        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            return [row_to_entry(tuple(r)) for r in rows]
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
        sql = f"""  -- noqa: S608 - placeholder count is derived from entry_ids length only
            SELECT vi.entry_id, vm.embedding
            FROM vec_memories vm
            JOIN vec_index vi ON vm.rowid = vi.rowid
            WHERE vi.entry_id IN ({placeholders})
        """
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
