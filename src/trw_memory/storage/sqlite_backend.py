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
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import structlog

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._shared import (
    ENTRY_COLUMNS,
    IMMUTABLE_FIELDS,
)
# SQLCipher driver/pragma + cold-rebuild base extracted to _sqlcipher_setup.py
# (PRD-DIST-245 batch 90). Re-exports preserve back-compat names.
from trw_memory.storage._sqlcipher_setup import (
    SQLCIPHER_CIPHER as SQLCIPHER_CIPHER,
    SQLCIPHER_CIPHER_PAGE_SIZE as SQLCIPHER_CIPHER_PAGE_SIZE,
    SQLCIPHER_KDF_ITER as SQLCIPHER_KDF_ITER,
    SQLCIPHER_REQUIRED_MESSAGE as SQLCIPHER_REQUIRED_MESSAGE,
    apply_sqlcipher_pragmas as _apply_sqlcipher_pragmas,
    import_sqlcipher_driver as _import_sqlcipher_driver,
    resolve_cold_rebuild_base as _resolve_cold_rebuild_base,
)
from trw_memory.storage._stale_handle_detector import StaleHandleDetector
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)
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
# recover_db extracted to _recovery.py (PRD-DIST-245 batch 83).
from trw_memory.storage._recovery import recover_db as _recovery_recover_db
# Resilient row materialisation extracted to _resilient_fetch.py
# (PRD-DIST-245 batch 84).
from trw_memory.storage._resilient_fetch import (
    fetch_rows_resilient as _resilient_fetch_rows_resilient,
    fetch_rows_via_bytes_fallback as _resilient_fetch_rows_via_bytes_fallback,
)
# Vector operations extracted to _vector_ops.py (PRD-DIST-245 batch 85).
from trw_memory.storage._vector_ops import (
    delete_vector as _vec_ops_delete_vector,
    delete_vector_internal as _vec_ops_delete_vector_internal,
    existing_vector_ids as _vec_ops_existing_vector_ids,
    get_stored_embeddings as _vec_ops_get_stored_embeddings,
    search_vectors as _vec_ops_search_vectors,
    upsert_vector as _vec_ops_upsert_vector,
    vector_exists as _vec_ops_vector_exists,
)
# Query / list / namespace operations extracted to _query_ops.py
# (PRD-DIST-245 batch 86).
from trw_memory.storage._query_ops import (
    count as _query_ops_count,
    delete_by_namespace as _query_ops_delete_by_namespace,
    entries_with_assertions as _query_ops_entries_with_assertions,
    list_entries as _query_ops_list_entries,
    list_namespaces as _query_ops_list_namespaces,
    search as _query_ops_search,
)
# CRUD ops extracted to _crud_ops.py (PRD-DIST-245 batch 87).
from trw_memory.storage._crud_ops import (
    delete as _crud_ops_delete,
    get as _crud_ops_get,
    increment_access_counts as _crud_ops_increment_access_counts,
    increment_session_counts as _crud_ops_increment_session_counts,
    store as _crud_ops_store,
    update as _crud_ops_update,
)
# __init__ helpers extracted to _init_helpers.py (PRD-DIST-245 batch 88).
from trw_memory.storage._init_helpers import (
    load_vec_extension as _init_load_vec_extension,
    open_connection_with_recovery as _init_open_connection_with_recovery,
    register_writer_registry as _init_register_writer_registry,
    start_integrity_scheduler as _init_start_integrity_scheduler,
)
# Stale-handle + integrity-check helpers extracted to _stale_handle.py
# (PRD-DIST-245 batch 89).
from trw_memory.storage._stale_handle import (
    ensure_connection_fresh as _stale_handle_ensure_fresh,
    handle_integrity_regression as _stale_handle_integrity_regression,
    reconnect as _stale_handle_reconnect,
    run_integrity_check as _stale_handle_run_integrity_check,
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

        # Connection open + auto-recovery (PRD-DIST-245 batch 88)
        self._conn, self.integrity_warning, self.recovered = _init_open_connection_with_recovery(
            self,
            db_path,
            dbapi=self._dbapi,
            sqlcipher_key_hex=sqlcipher_key_hex,
            recovery_policy=self._recovery_policy,
            corrupt_backup_keep=self._corrupt_backup_keep,
            rebuild_from_cold=self._rebuild_from_cold,
        )

        # P3 — stale-handle detector (belt + suspenders: inode + sentinel).
        self._stale_detector = StaleHandleDetector(db_path)

        # sqlite-vec extension load (fail-open)
        self._vec_available = _init_load_vec_extension(self._conn, db_path, self._dim)

        # PRD-INFRA-064 (B3): multi-writer advisory registry (fail-open)
        self._writer_registry = _init_register_writer_registry(
            db_path, self._concurrent_writer_warn_threshold
        )

        # PRD-INFRA-063 (B2): periodic integrity scheduler (fail-open)
        self._integrity_scheduler = _init_start_integrity_scheduler(
            db_path,
            interval_minutes=self._integrity_check_interval_minutes,
            on_regression=self._handle_integrity_regression,
        )

    def _handle_integrity_regression(self, _db_path: Path, _detail: str) -> None:
        """Delegate to ``_stale_handle.handle_integrity_regression``."""
        _stale_handle_integrity_regression(self)

    def _reconnect(self) -> None:
        """Delegate to ``_stale_handle.reconnect``."""
        _stale_handle_reconnect(self)

    def _ensure_connection_fresh(self) -> None:
        """Delegate to ``_stale_handle.ensure_connection_fresh``."""
        _stale_handle_ensure_fresh(self)

    # ------------------------------------------------------------------
    # Integrity & recovery
    # ------------------------------------------------------------------

    # Connection-mgmt + corrupt-backup + recovery delegators are direct
    # staticmethod aliases — the parent module imports each helper as
    # `_connection_*` / `_*_impl` / `_recovery_*` and re-exposes them
    # under their existing class-method names.
    _connect = staticmethod(_connection_connect)
    _open_and_configure = staticmethod(_connection_open_and_configure)
    _open_without_integrity_check = staticmethod(_connection_open_without_integrity_check)
    _db_has_data = staticmethod(_connection_db_has_data)
    _salvage_via_recover_cli = staticmethod(_salvage_via_recover_cli_impl)
    _rotate_corrupt_backup = staticmethod(_rotate_corrupt_backup_impl)
    _prune_corrupt_backups = staticmethod(_prune_corrupt_backups_impl)
    recover_db = staticmethod(_recovery_recover_db)

    def _run_integrity_check(self) -> bool:
        """Delegate to ``_stale_handle.run_integrity_check``."""
        return _stale_handle_run_integrity_check(self)

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
    # P2 — Resilient row materialisation (extracted to _resilient_fetch.py
    # PRD-DIST-245 batch 84)
    # ------------------------------------------------------------------

    def _fetch_rows_resilient(
        self,
        cursor: Any,
        *,
        table: str = "memories",
    ) -> list[MemoryEntry]:
        """Delegate to ``_resilient_fetch.fetch_rows_resilient``."""
        results, delta = _resilient_fetch_rows_resilient(
            cursor,
            db_path=self._db_path,
            dbapi=self._dbapi,
            select_columns_sql=_SELECT_COLUMNS_SQL,
            table=table,
        )
        self.quarantine_count_utf8 += delta
        return results

    def _fetch_rows_via_bytes_fallback(
        self,
        cursor: Any,
        *,
        table: str = "memories",
    ) -> list[MemoryEntry]:
        """Delegate to ``_resilient_fetch.fetch_rows_via_bytes_fallback``."""
        results, delta = _resilient_fetch_rows_via_bytes_fallback(
            cursor,
            db_path=self._db_path,
            dbapi=self._dbapi,
            select_columns_sql=_SELECT_COLUMNS_SQL,
            table=table,
        )
        self.quarantine_count_utf8 += delta
        return results

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
    # StorageBackend interface — CRUD ops delegated to _crud_ops.py
    # (PRD-DIST-245 batch 87)
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        """INSERT OR REPLACE the entry into the memories table."""
        _crud_ops_store(self, _INSERT_COLUMNS_SQL, _COLUMNS, entry)

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve an entry by id."""
        return _crud_ops_get(self, _SELECT_COLUMNS_SQL, entry_id)

    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
        """Apply a partial update to an existing entry."""
        return _crud_ops_update(self, _SELECT_COLUMNS_SQL, _VALID_UPDATE_COLUMNS, entry_id, **fields)

    def increment_session_counts(
        self, entry_ids: list[str], *, updated_at: datetime | None = None
    ) -> int:
        """Increment session_count for multiple entries in one transaction."""
        return _crud_ops_increment_session_counts(self, entry_ids, updated_at=updated_at)

    def increment_access_counts(
        self, entry_ids: list[str], *, accessed_at: datetime | None = None
    ) -> int:
        """Increment access_count and last_accessed_at in one transaction."""
        return _crud_ops_increment_access_counts(self, entry_ids, accessed_at=accessed_at)

    def delete(self, entry_id: str) -> bool:
        """Remove an entry from memories (and vec_index when available)."""
        return _crud_ops_delete(self, entry_id)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[SQLiteBackend]:
        """PRD-FIX-088 FR02: batch N writes into one BEGIN IMMEDIATE / COMMIT.

        Re-entrant by depth — only the outermost ``transaction()`` issues
        BEGIN/COMMIT; inner exceptions propagate; outermost issues ROLLBACK.
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

    # ------------------------------------------------------------------
    # Query / list / namespace ops (delegated to _query_ops.py
    # — PRD-DIST-245 batch 86)
    # ------------------------------------------------------------------

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
        """Keyword LIKE search on content + detail + tags with filters."""
        return _query_ops_search(
            self,
            _SELECT_COLUMNS_SQL,
            query=query,
            top_k=top_k,
            tags=tags,
            status=status,
            min_importance=min_importance,
            namespace=namespace,
        )

    def count(self, namespace: str | None = None) -> int:
        """Return the number of stored entries."""
        return _query_ops_count(self, namespace)

    def entries_with_assertions(self) -> list[MemoryEntry]:
        """PRD-CORE-086 FR07 query for assertion-health summary."""
        return _query_ops_entries_with_assertions(self)

    def count_with_assertions(self) -> list[MemoryEntry]:
        """Backward-compatible alias for PRD-CORE-086 FR07 traceability."""
        return _query_ops_entries_with_assertions(self)

    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Return entries with optional filters, ordered by updated_at desc."""
        return _query_ops_list_entries(
            self,
            _SELECT_COLUMNS_SQL,
            status=status,
            namespace=namespace,
            limit=limit,
        )

    def list_namespaces(self) -> list[str]:
        """Return all distinct namespaces that have stored entries."""
        return _query_ops_list_namespaces(self)

    def delete_by_namespace(self, namespace: str) -> int:
        """Delete all entries in a namespace."""
        return _query_ops_delete_by_namespace(self, namespace)

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
    # Vector operations (delegated to _vector_ops.py — PRD-DIST-245 batch 85)
    # ------------------------------------------------------------------

    def _delete_vector(self, entry_id: str) -> None:
        """Internal vector deletion (caller holds the lock)."""
        _vec_ops_delete_vector_internal(self._conn, entry_id)

    def delete_vector(self, entry_id: str) -> bool:
        """Public vector-row deletion."""
        return _vec_ops_delete_vector(
            self._conn, self._lock, vec_available=self._vec_available, entry_id=entry_id
        )

    def vector_exists(self, entry_id: str) -> bool:
        """Single-row vec_index probe."""
        return _vec_ops_vector_exists(self._conn, vec_available=self._vec_available, entry_id=entry_id)

    def existing_vector_ids(self) -> set[str]:
        """Bulk set of entry IDs with stored vectors."""
        return _vec_ops_existing_vector_ids(
            self._conn, self._lock, vec_available=self._vec_available
        )

    def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
        """Insert or update a vector in vec_memories."""
        _vec_ops_upsert_vector(
            self._conn,
            self._lock,
            vec_available=self._vec_available,
            dim=self._dim,
            entry_id=entry_id,
            embedding=embedding,
        )

    def search_vectors(self, query_embedding: list[float], top_k: int = 25) -> list[tuple[str, float]]:
        """KNN search in vec_memories."""
        return _vec_ops_search_vectors(
            self._conn,
            self._lock,
            vec_available=self._vec_available,
            dim=self._dim,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def get_stored_embeddings(self, entry_ids: list[str]) -> dict[str, list[float]]:
        """Bulk lookup of packed embedding blobs."""
        return _vec_ops_get_stored_embeddings(
            self._conn, self._lock, vec_available=self._vec_available, entry_ids=entry_ids
        )
