"""SQLite + sqlite-vec storage backend.

Primary storage implementation for trw-memory.  Stores all :class:`MemoryEntry`
fields in a plain ``memories`` table, with an optional ``vec_memories`` virtual
table (sqlite-vec) for vector search.

Graceful degradation: if sqlite-vec is not installed, all metadata operations
continue to work; only :meth:`upsert_vector` and :meth:`search_vectors` become
no-ops.
"""
# ruff: noqa: E402,I001 - facade re-export imports are intentionally grouped near delegator comments.

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

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
from trw_memory.storage._wal_checkpoint import (
    CheckpointMode as CheckpointMode,
    CheckpointResult as CheckpointResult,
    run_checkpoint as run_checkpoint,
)
from trw_memory.storage.interface import StorageBackend

if TYPE_CHECKING:
    from trw_memory.wiki.storage import StoredWikiReference

logger = structlog.get_logger(__name__)
__all__ = [
    "CheckpointMode",
    "CheckpointResult",
    "SQLiteBackend",
    "_apply_sqlcipher_pragmas",
    "_import_sqlcipher_driver",
    "_resolve_cold_rebuild_base",
    "run_checkpoint",
]
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
    check_integrity as _connection_check_integrity,
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
    FetchQuery,
    _CursorLike,
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
    find_active_by_content as _query_ops_find_active_by_content,
    list_entries as _query_ops_list_entries,
    list_namespaces as _query_ops_list_namespaces,
    search as _query_ops_search,
)

# CRUD ops extracted to _crud_ops.py (PRD-DIST-245 batch 87).
from trw_memory.storage._crud_ops import (
    delete as _crud_ops_delete,
    get as _crud_ops_get,
    increment_access_counts as _crud_ops_increment_access_counts,
    increment_recall_access as _crud_ops_increment_recall_access,
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
from trw_memory.storage._transaction import transaction as _transaction_impl


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
        recovery_inline_max_bytes: int = 64 * 1024 * 1024,
        integrity_check_interval_minutes: int = 0,
        concurrent_writer_warn_threshold: int = 4,
    ) -> None:
        self._db_path = db_path
        self._dim = dim
        self._lock = threading.Lock()
        # PRD-FIX-088 FR02: open-transaction depth — mutating methods skip
        # per-row commit when >0 so caller-controlled outer transaction
        # batches N writes into one BEGIN IMMEDIATE / COMMIT.
        self._skip_commit_depth: int = 0
        # S8 follow-on: serialize OUTER transaction() bodies so two threads
        # cannot both hold an open BEGIN IMMEDIATE on the single shared
        # connection (which raises "cannot start a transaction within a
        # transaction"). Re-entrant so same-thread nested transaction() calls
        # do not self-deadlock; held only by the outermost transaction, and is
        # a DIFFERENT lock from ``_lock`` (which inner delegated writes acquire),
        # so no deadlock arises between the two.
        self._txn_serializer = threading.RLock()
        self._dbapi: Any = _import_sqlcipher_driver() if sqlcipher_key_hex is not None else sqlite3
        self._sqlcipher_key_hex = sqlcipher_key_hex
        self._recovery_policy = recovery_policy
        self._corrupt_backup_keep = corrupt_backup_keep
        self._rebuild_from_cold = rebuild_from_cold
        self._integrity_check_interval_minutes = integrity_check_interval_minutes
        self._concurrent_writer_warn_threshold = concurrent_writer_warn_threshold
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # P2 — per-row UTF-8 quarantine counter (incremented on each skipped row)
        self.quarantine_count_utf8: int = 0
        # P3 — reconnect counter (incremented on each stale-handle reopen)
        self.reconnect_count: int = 0
        self.recovery_preflight: Any = None

        # Connection open + auto-recovery (PRD-DIST-245 batch 88)
        self._conn, self.integrity_warning, self.recovered = _init_open_connection_with_recovery(
            self,
            db_path,
            dbapi=self._dbapi,
            sqlcipher_key_hex=sqlcipher_key_hex,
            recovery_policy=self._recovery_policy,
            corrupt_backup_keep=self._corrupt_backup_keep,
            rebuild_from_cold=self._rebuild_from_cold,
            recovery_inline_max_bytes=recovery_inline_max_bytes,
        )

        # WAL-reset-bug safety gate. The active SQLite engine carries the
        # WAL-reset corruption bug (sqlite.org/wal.html §walresetbug) unless it
        # is >= 3.51.3 (or backport 3.44.6 / 3.50.7). When unsafe we cannot fix
        # the engine (no fixed pysqlite3 wheel exists yet) so we warn loudly and
        # rely on single-connection checkpoint serialization (see checkpoint_wal)
        # to avoid the two-connection race that detonates the bug.
        from trw_memory.storage import _dbapi as _driver

        self.wal_reset_safe: bool = _driver.is_wal_reset_safe()
        if not self.wal_reset_safe:
            logger.warning(
                "sqlite_wal_reset_unsafe",
                sqlite_version=_driver.sqlite_version(),
                driver=_driver.backend(),
                detail=(
                    "Active SQLite predates the 3.51.3 WAL-reset fix; WAL "
                    "checkpoints are serialized on the single owning connection "
                    "to avoid the corruption race. Upgrade to a pysqlite3 build "
                    "bundling SQLite>=3.51.3 when one is published."
                ),
            )

        # P3 — stale-handle detector (belt + suspenders: inode + sentinel).
        self._stale_detector = StaleHandleDetector(db_path)

        # sqlite-vec extension load (fail-open)
        self._vec_available = _init_load_vec_extension(self._conn, db_path, self._dim)

        # PRD-INFRA-064 (B3): multi-writer advisory registry (fail-open)
        self._writer_registry = _init_register_writer_registry(db_path, self._concurrent_writer_warn_threshold)

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

    # Public integrity probe delegated to ``_connection.check_integrity``;
    # kept as a staticmethod alias so ``SQLiteBackend.check_integrity`` callers
    # and test patches resolve unchanged.
    check_integrity = staticmethod(_connection_check_integrity)

    # ------------------------------------------------------------------
    # Public property
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SQLiteBackend(db_path={self._db_path!r}, vec={self._vec_available})"

    @property
    def vec_available(self) -> bool:
        """``True`` when sqlite-vec is loaded and the virtual table exists."""
        return self._vec_available

    def supports_vectors(self) -> bool:
        """Explicit vector-capability signal (see :class:`StorageBackend`).

        Reflects the real ``sqlite-vec`` load state so callers can skip
        embedding work that would otherwise no-op through ``upsert_vector``.
        """
        return self._vec_available

    @property
    def db_path(self) -> Path:
        """Filesystem path of the backing database file."""
        return self._db_path

    # ------------------------------------------------------------------
    # P2 — Resilient row materialisation (extracted to _resilient_fetch.py
    # PRD-DIST-245 batch 84)
    # ------------------------------------------------------------------

    def _fetch_query(
        self,
        *,
        where_sql: str = "1",
        params: Sequence[object] = (),
        order_by: str = "updated_at DESC",
        limit: int | None = None,
        table: str = "memories",
    ) -> FetchQuery:
        """Build the :class:`FetchQuery` the resilient path re-executes."""
        return FetchQuery(
            select_columns_sql=_SELECT_COLUMNS_SQL,
            table=table,
            where_sql=where_sql,
            params=tuple(params),
            order_by=order_by,
            limit=limit,
        )

    def _fetch_rows_resilient(
        self,
        cursor: _CursorLike,
        *,
        query: FetchQuery | None = None,
    ) -> list[MemoryEntry]:
        """Delegate to ``_resilient_fetch.fetch_rows_resilient``."""
        results, delta = _resilient_fetch_rows_resilient(
            cursor,
            db_path=self._db_path,
            dbapi=self._dbapi,
            query=query if query is not None else self._fetch_query(),
        )
        self.quarantine_count_utf8 += delta
        return results

    def _fetch_rows_via_bytes_fallback(
        self,
        *,
        query: FetchQuery | None = None,
    ) -> list[MemoryEntry]:
        """Delegate to ``_resilient_fetch.fetch_rows_via_bytes_fallback``."""
        results, delta = _resilient_fetch_rows_via_bytes_fallback(
            db_path=self._db_path,
            dbapi=self._dbapi,
            query=query if query is not None else self._fetch_query(),
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
        from trw_memory.wiki.storage import replace_wiki_refs_for_entry

        replace_wiki_refs_for_entry(self, entry)

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve an entry by id."""
        return _crud_ops_get(self, _SELECT_COLUMNS_SQL, entry_id)

    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
        """Apply a partial update to an existing entry."""
        updated = _crud_ops_update(self, _SELECT_COLUMNS_SQL, _VALID_UPDATE_COLUMNS, entry_id, **fields)
        if updated is not None and "metadata" in fields:
            from trw_memory.wiki.storage import replace_wiki_refs_for_entry

            replace_wiki_refs_for_entry(self, updated)
        return updated

    def increment_session_counts(self, entry_ids: list[str], *, updated_at: datetime | None = None) -> int:
        """Increment session_count for multiple entries in one transaction."""
        return _crud_ops_increment_session_counts(self, entry_ids, updated_at=updated_at)

    def increment_access_counts(self, entry_ids: list[str], *, accessed_at: datetime | None = None) -> int:
        """Increment access_count and last_accessed_at in one transaction."""
        return _crud_ops_increment_access_counts(self, entry_ids, accessed_at=accessed_at)

    def increment_recall_access(self, entry_ids: list[str], *, accessed_at: datetime | None = None) -> int:
        """F-008: increment access_count + recall_count + last_accessed_at in ONE commit."""
        return _crud_ops_increment_recall_access(self, entry_ids, accessed_at=accessed_at)

    def delete(self, entry_id: str) -> bool:
        """Remove an entry from memories (and vec_index when available)."""
        deleted = _crud_ops_delete(self, entry_id)
        if deleted:
            from trw_memory.wiki.storage import purge_wiki_refs_for_entry

            purge_wiki_refs_for_entry(self, entry_id)
        return deleted

    @contextlib.contextmanager
    def transaction(self) -> Iterator[SQLiteBackend]:
        """PRD-FIX-088 FR02: batch N writes into one BEGIN IMMEDIATE / COMMIT.

        Implementation lives in ``storage._transaction.transaction`` so the
        concurrency-sensitive transaction seam is local and testable while this
        backend remains the public adapter.
        """
        with _transaction_impl(self) as txn:
            yield txn

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

    def find_active_by_content(
        self,
        content: str,
        detail: str,
        *,
        namespace: str = "default",
    ) -> str | None:
        """Return the id of an ACTIVE entry with exactly matching content + detail.

        Embedding-independent exact-content dedup lookup (PRD-CORE-042).
        Read-only; namespace-scoped. Returns None when no exact duplicate exists.
        """
        return _query_ops_find_active_by_content(self, content, detail, namespace=namespace)

    def count(self, namespace: str | None = None) -> int:
        """Return the number of stored entries."""
        return _query_ops_count(self, namespace)

    def entries_with_assertions(
        self,
        *,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        namespace: str | None = None,
        limit: int = 500,
    ) -> list[MemoryEntry]:
        """PRD-CORE-086 FR07 query for assertion-health summary.

        F7: defaults to active-only so obsolete entries' stale assertions
        don't pollute the session-start summary. ``status=None`` = all statuses.

        ``namespace`` scopes the query to one namespace (omit for all). ``limit``
        caps the scan (default 500) so the summary never triggers an unbounded
        full-table scan on a large store.
        """
        return _query_ops_entries_with_assertions(
            self, _SELECT_COLUMNS_SQL, status=status, namespace=namespace, limit=limit
        )

    # Backward-compat alias for PRD-CORE-086 FR07 traceability.
    count_with_assertions = entries_with_assertions

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
        """Delete all entries in a namespace atomically.

        S8 fix: the entry-row DELETE, the companion ``wiki_refs`` cleanup, and
        per-entry vector removal all run inside ONE ``transaction()`` so a crash
        can never leave orphan wiki refs or vector rows pointing at deleted
        entries — it is all-or-nothing.
        """
        with self.transaction():
            # Snapshot the namespace's entry IDs INSIDE the BEGIN IMMEDIATE txn so
            # vec_index rows (keyed on entry_id, not namespace) are cleaned up for
            # exactly the rows the DELETE removes. Reading the IDs before the txn
            # left a TOCTOU: a concurrent INSERT between the SELECT and BEGIN got
            # deleted from `memories` (DELETE WHERE namespace) but its vec rows
            # were missed (stale id list) → orphan vec hits. The write lock here
            # blocks concurrent writers, so the snapshot matches the delete.
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id FROM memories WHERE namespace = ?",
                    (namespace,),
                ).fetchall()
            entry_ids = [str(row[0]) for row in rows]
            if not entry_ids:
                return 0
            deleted = _query_ops_delete_by_namespace(self, namespace)
            with self._lock:
                self._conn.execute("DELETE FROM wiki_refs WHERE namespace = ?", (namespace,))
                if self._vec_available:
                    for entry_id in entry_ids:
                        self._delete_vector(entry_id)
        return deleted

    def query_wiki_outbound_refs(self, source_slug: str, *, namespace: str | None = None) -> list[StoredWikiReference]:
        """Return deterministic persisted outbound wiki refs for ``source_slug``."""
        from trw_memory.wiki.storage import query_wiki_outbound_refs

        return list(query_wiki_outbound_refs(self, source_slug, namespace=namespace))

    def query_wiki_inbound_refs(self, target_slug: str, *, namespace: str | None = None) -> list[StoredWikiReference]:
        """Return deterministic persisted inbound wiki refs for ``target_slug``."""
        from trw_memory.wiki.storage import query_wiki_inbound_refs

        return list(query_wiki_inbound_refs(self, target_slug, namespace=namespace))

    def close(self) -> None:
        """Stop integrity scheduler + writer registry, then close connection."""
        if self._integrity_scheduler is not None:
            with contextlib.suppress(Exception):
                self._integrity_scheduler.stop(timeout=2.0)
            self._integrity_scheduler = None
        if self._writer_registry is not None:
            with contextlib.suppress(Exception):
                self._writer_registry.close()
            self._writer_registry = None
        with contextlib.suppress(sqlite3.Error):
            self._conn.close()
        logger.debug("sqlite_backend_closed", db=str(self._db_path))

    def __enter__(self) -> SQLiteBackend:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def checkpoint_wal(self, mode: str = "TRUNCATE") -> CheckpointResult:
        """Checkpoint the WAL on the single owning connection under the lock.

        This is the structural fix for the SQLite WAL-reset corruption bug
        (sqlite.org/wal.html §walresetbug, fixed in 3.51.3): the bug requires
        TWO OR MORE connections checkpointing the same WAL concurrently. By
        routing every checkpoint through ``self._conn`` while holding
        ``self._lock`` — the same connection and lock all writes use — there is
        never a second checkpointer to race, so the bug cannot fire even on an
        unpatched engine.

        On an unsafe engine a resetting checkpoint (TRUNCATE/RESTART) is
        downgraded to PASSIVE; otherwise TRUNCATE reclaims WAL file space and
        falls back to PASSIVE when readers still hold pages (``busy=1``). The
        mode coercion + fallback live in :mod:`_wal_checkpoint`; this method
        only owns the connection + lock. Fail-open: never raises.

        Returns:
            A :class:`CheckpointResult` (a ``dict`` at runtime) — exported from
            ``trw_memory.storage`` so consumers share a precise contract.
        """
        with self._lock:
            return run_checkpoint(
                lambda sql: self._conn.execute(sql).fetchone(),
                mode,
                wal_reset_safe=self.wal_reset_safe,
                db_path=str(self._db_path),
            )

    # ------------------------------------------------------------------
    # Vector operations (delegated to _vector_ops.py — PRD-DIST-245 batch 85)
    # ------------------------------------------------------------------

    def _delete_vector(self, entry_id: str) -> None:
        """Internal vector deletion (caller holds the lock)."""
        _vec_ops_delete_vector_internal(self._conn, entry_id)

    def delete_vector(self, entry_id: str) -> bool:
        """Public vector-row deletion.

        Defers the commit when inside a ``transaction()`` block so the vector
        delete batches into the caller's outermost COMMIT (atomicity parity with
        ``delete()``/``store()``).
        """
        return _vec_ops_delete_vector(
            self._conn,
            self._lock,
            vec_available=self._vec_available,
            entry_id=entry_id,
            skip_commit=self._skip_commit_depth != 0,
        )

    def vector_exists(self, entry_id: str) -> bool:
        """Single-row vec_index probe."""
        return _vec_ops_vector_exists(self._conn, vec_available=self._vec_available, entry_id=entry_id)

    def existing_vector_ids(self) -> set[str]:
        """Bulk set of entry IDs with stored vectors."""
        return _vec_ops_existing_vector_ids(self._conn, self._lock, vec_available=self._vec_available)

    def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
        """Insert or update a vector in vec_memories."""
        _vec_ops_upsert_vector(
            self._conn,
            self._lock,
            vec_available=self._vec_available,
            dim=self._dim,
            entry_id=entry_id,
            embedding=embedding,
            skip_commit=self._skip_commit_depth != 0,
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
