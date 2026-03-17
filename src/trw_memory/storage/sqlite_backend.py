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
import json
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._parsing import (
    parse_dt,
    parse_json_dict_int,
    parse_json_dict_str,
    parse_json_list,
)
from trw_memory.storage._shared import (
    DICT_FIELDS,
    ENTRY_COLUMNS,
    IMMUTABLE_FIELDS,
    LIST_FIELDS,
    serialize_update_value,
    validate_update_fields,
)
from trw_memory.storage.interface import StorageBackend

try:
    import sqlite_vec  # type: ignore[import-untyped]

    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    detail            TEXT DEFAULT '',
    tags              TEXT DEFAULT '[]',
    evidence          TEXT DEFAULT '[]',
    importance        REAL DEFAULT 0.5,
    status            TEXT DEFAULT 'active',
    recurrence        INTEGER DEFAULT 1,
    namespace         TEXT DEFAULT 'default',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    last_accessed_at  TEXT,
    access_count      INTEGER DEFAULT 0,
    q_value           REAL DEFAULT 0.5,
    q_observations    INTEGER DEFAULT 0,
    source            TEXT DEFAULT 'agent',
    source_identity   TEXT DEFAULT '',
    merged_from       TEXT DEFAULT '[]',
    consolidated_from TEXT DEFAULT '[]',
    consolidated_into TEXT,
    metadata          TEXT DEFAULT '{}',
    vector_clock      TEXT DEFAULT '{}',
    remote_id         TEXT,
    published_to_platform INTEGER DEFAULT 0,
    pending_delete    INTEGER DEFAULT 0,
    cross_validated   INTEGER DEFAULT 0,
    outcome_history   TEXT DEFAULT '[]'
)
"""

_CREATE_IDX_NAMESPACE = "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)"
_CREATE_IDX_STATUS = "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)"

_CREATE_GRAPH_EDGES = """
CREATE TABLE IF NOT EXISTS memory_graph_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    created_at  TEXT NOT NULL,
    UNIQUE (source_id, target_id, edge_type)
)
"""

_CREATE_IDX_MGE_SOURCE = "CREATE INDEX IF NOT EXISTS idx_mge_source ON memory_graph_edges(source_id, edge_type)"
_CREATE_IDX_MGE_TARGET = "CREATE INDEX IF NOT EXISTS idx_mge_target ON memory_graph_edges(target_id, edge_type)"

_CREATE_NAMESPACES = """
CREATE TABLE IF NOT EXISTS memory_namespaces (
    namespace_id  TEXT PRIMARY KEY,
    team_id       TEXT,
    created_at    TEXT NOT NULL,
    expires_at    TEXT,
    status        TEXT NOT NULL DEFAULT 'active'
)
"""

_CREATE_IDX_MN_STATUS = "CREATE INDEX IF NOT EXISTS idx_mn_status ON memory_namespaces(status, expires_at)"

# ---------------------------------------------------------------------------
# Row <-> MemoryEntry helpers
# ---------------------------------------------------------------------------

_COLUMNS = ENTRY_COLUMNS
_COLUMNS_SQL = ", ".join(_COLUMNS)

# Allowlist for UPDATE: all columns except immutable ones.
_VALID_UPDATE_COLUMNS: frozenset[str] = frozenset(_COLUMNS) - IMMUTABLE_FIELDS


def _row_to_entry(row: tuple[object, ...]) -> MemoryEntry:
    """Convert a SQLite row tuple to a :class:`MemoryEntry`.

    The column order must match :data:`_COLUMNS`.
    """
    (
        id_,
        content,
        detail,
        tags_json,
        evidence_json,
        importance,
        status,
        recurrence,
        namespace,
        created_at_s,
        updated_at_s,
        last_accessed_s,
        access_count,
        q_value,
        q_obs,
        source,
        source_identity,
        merged_json,
        cons_from_json,
        consolidated_into,
        metadata_json,
        vector_clock_json,
        remote_id,
        published_raw,
        pending_del_raw,
        cross_val_raw,
        outcome_json,
    ) = row

    return MemoryEntry(
        id=str(id_),
        content=str(content),
        detail=str(detail) if detail else "",
        tags=parse_json_list(tags_json),
        evidence=parse_json_list(evidence_json),
        importance=float(str(importance)),
        status=MemoryStatus(str(status)),
        recurrence=int(str(recurrence)),
        namespace=str(namespace),
        created_at=parse_dt(created_at_s),
        updated_at=parse_dt(updated_at_s),
        last_accessed_at=parse_dt(last_accessed_s) if last_accessed_s else None,
        access_count=int(str(access_count)),
        q_value=float(str(q_value)),
        q_observations=int(str(q_obs)),
        source=str(source),
        source_identity=str(source_identity) if source_identity else "",
        merged_from=parse_json_list(merged_json),
        consolidated_from=parse_json_list(cons_from_json),
        consolidated_into=str(consolidated_into) if consolidated_into else None,
        metadata=parse_json_dict_str(metadata_json),
        vector_clock=parse_json_dict_int(vector_clock_json),
        remote_id=str(remote_id) if remote_id else None,
        published_to_platform=bool(int(str(published_raw))) if published_raw else False,
        pending_delete=bool(int(str(pending_del_raw))) if pending_del_raw else False,
        cross_validated=bool(int(str(cross_val_raw))) if cross_val_raw else False,
        outcome_history=parse_json_list(outcome_json),
    )


def _entry_to_row(entry: MemoryEntry) -> tuple[object, ...]:
    """Convert a :class:`MemoryEntry` to an INSERT/REPLACE row tuple."""
    # Pydantic v2: use_enum_values=True + strict=True can leave the field
    # as an enum instance in some code paths.  Safely extract the value.
    raw_status = entry.status
    status_val = raw_status.value if isinstance(raw_status, MemoryStatus) else str(raw_status)
    return (
        entry.id,
        entry.content,
        entry.detail,
        json.dumps(entry.tags),
        json.dumps(entry.evidence),
        entry.importance,
        status_val,
        entry.recurrence,
        entry.namespace,
        entry.created_at.isoformat(),
        entry.updated_at.isoformat(),
        entry.last_accessed_at.isoformat() if entry.last_accessed_at else None,
        entry.access_count,
        entry.q_value,
        entry.q_observations,
        entry.source,
        entry.source_identity,
        json.dumps(entry.merged_from),
        json.dumps(entry.consolidated_from),
        entry.consolidated_into,
        json.dumps(entry.metadata),
        json.dumps(entry.vector_clock),
        entry.remote_id,
        int(entry.published_to_platform),
        int(entry.pending_delete),
        int(entry.cross_validated),
        json.dumps(entry.outcome_history),
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

    def __init__(self, db_path: Path, dim: int = 384) -> None:
        self._db_path = db_path
        self._dim = dim
        self._vec_available = False
        self._lock = threading.Lock()

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row

        # WAL mode: concurrent reads are not serialized behind writes.
        # synchronous=NORMAL is safe with WAL and avoids fsync on every commit.
        wal_result = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if wal_result and wal_result[0] != "wal":
            logger.warning("wal_mode_not_enabled", got=wal_result[0])
        sync_result = self._conn.execute("PRAGMA synchronous=NORMAL").fetchone()
        if sync_result and sync_result[0] not in ("1", 1):
            logger.warning("synchronous_normal_not_set", got=sync_result[0] if sync_result else None)

        self._ensure_schema()

        if _SQLITE_VEC_AVAILABLE:
            try:
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
                self._ensure_vec_table()
                self._vec_available = True
                logger.debug("sqlite_vec_loaded", db=str(db_path))
            except (sqlite3.Error, OSError):
                self._vec_available = False
                logger.debug("sqlite_vec_load_failed", db=str(db_path), exc_info=True)
        else:
            logger.debug("sqlite_vec_unavailable", reason="not_installed")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute(_CREATE_MEMORIES)
            cursor.execute(_CREATE_IDX_NAMESPACE)
            cursor.execute(_CREATE_IDX_STATUS)
            cursor.execute(_CREATE_GRAPH_EDGES)
            cursor.execute(_CREATE_IDX_MGE_SOURCE)
            cursor.execute(_CREATE_IDX_MGE_TARGET)
            cursor.execute(_CREATE_NAMESPACES)
            cursor.execute(_CREATE_IDX_MN_STATUS)
            # Migration: add new columns for sync + graph (Sprint 37)
            _migrate_cols = [
                ("vector_clock", "TEXT DEFAULT '{}'"),
                ("remote_id", "TEXT"),
                ("published_to_platform", "INTEGER DEFAULT 0"),
                ("pending_delete", "INTEGER DEFAULT 0"),
                ("cross_validated", "INTEGER DEFAULT 0"),
                ("outcome_history", "TEXT DEFAULT '[]'"),
            ]
            for col_name, col_def in _migrate_cols:
                with contextlib.suppress(sqlite3.OperationalError):
                    cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
            self._conn.commit()
        finally:
            cursor.close()

    def _ensure_vec_table(self) -> None:
        self._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(embedding float[{self._dim}])")
        # Companion table to map rowid <-> entry_id
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_index ("
            "rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
            "entry_id TEXT UNIQUE NOT NULL)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public property
    # ------------------------------------------------------------------

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
        placeholders = ", ".join(["?"] * len(_COLUMNS))
        sql = f"INSERT OR REPLACE INTO memories ({_COLUMNS_SQL}) VALUES ({placeholders})"  # noqa: S608 — _COLUMNS_SQL is a static constant (no user input); values are parameterized
        try:
            with self._lock:
                self._conn.execute(sql, _entry_to_row(entry))
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
        sql = f"SELECT {_COLUMNS_SQL} FROM memories WHERE id = ?"  # noqa: S608 — _COLUMNS_SQL is a static constant; entry_id is a parameterized ?
        try:
            with self._lock:
                row = self._conn.execute(sql, (entry_id,)).fetchone()
            if row is None:
                return None
            return _row_to_entry(tuple(row))
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

            for key, val in field_dict.items():
                set_parts.append(f"{key} = ?")
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
            return deleted
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
            f"SELECT {_COLUMNS_SQL} FROM memories WHERE {where_sql} "  # noqa: S608 — _COLUMNS_SQL and where_sql are built from static constants and ? placeholders only
            f"ORDER BY importance DESC, updated_at DESC LIMIT ?"
        )
        params.append(top_k)

        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            results = [_row_to_entry(tuple(r)) for r in rows]

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
            f"SELECT {_COLUMNS_SQL} FROM memories WHERE {where_sql} "  # noqa: S608 — _COLUMNS_SQL and where_sql are built from static constants and ? placeholders only
            f"ORDER BY updated_at DESC LIMIT ?"
        )
        params.append(limit)

        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            return [_row_to_entry(tuple(r)) for r in rows]
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
            return deleted
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
