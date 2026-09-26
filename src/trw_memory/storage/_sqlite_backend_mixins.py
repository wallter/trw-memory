"""WAL checkpoint and vector-operation mixins for ``SQLiteBackend``."""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

from trw_memory.embeddings.provenance import EmbeddingSpace, StoredVector, VectorProvenance
from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._change_feed import change_token, entries_changed_since
from trw_memory.storage._crud_index_ops import insert_edges, read_edges
from trw_memory.storage._vector_ops import (
    delete_hype_siblings,
    delete_vector,
    delete_vector_internal,
    existing_vector_ids,
    get_stored_embeddings,
    get_vector_records,
    hype_sibling_ids,
    search_vectors,
    upsert_vector,
    vector_exists,
    vector_space_census,
)
from trw_memory.storage._wal_checkpoint import CheckpointResult
from trw_memory.storage.interface import GraphEdge, NamespaceChangeToken

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


class SQLiteCheckpointVectorMixin:
    """Operations independent of core metadata CRUD and query behavior."""

    _conn: Any
    _db_path: Path
    _dbapi: Any
    _dim: int
    _lock: Any
    _skip_commit_depth: int
    _vec_available: bool
    wal_reset_safe: bool

    def _fresh_connection(self) -> contextlib.AbstractContextManager[None]:
        raise NotImplementedError

    def transaction(self) -> contextlib.AbstractContextManager[Any]:
        raise NotImplementedError

    def checkpoint_wal(self, mode: str = "TRUNCATE") -> CheckpointResult:
        """Checkpoint the owning connection under the backend lock; fail open.

        A resetting *mode* is honoured only on an engine carrying the SQLite
        3.51.3 WAL-reset fix; below that it is coerced to ``PASSIVE`` with no
        caller-supplied escape (see ``_wal_checkpoint``'s module docstring for
        why a sole-writer certification is not sufficient). ``PASSIVE`` writes
        frames back but never truncates, so on an unsafe engine the file stays
        at ``_connection.WAL_JOURNAL_SIZE_LIMIT_BYTES`` (64 MiB) once it gets
        there — well above trw-mcp's 10 MB ``wal_checkpoint_threshold_mb``
        trigger, which is why that trigger keeps firing to no visible effect.
        """
        # Resolve through the public facade at call time. Besides retaining the
        # long-standing monkeypatch seam, this keeps embedders that instrument
        # checkpoint/lock behavior compatible with the mixin extraction.
        from trw_memory.storage import sqlite_backend as facade

        try:
            checkpoint_lock = (
                contextlib.nullcontext()
                if str(self._db_path) == ":memory:"
                else facade.lock_for_rmw(Path(f"{self._db_path.resolve(strict=False)}.checkpoint"))
            )
            with checkpoint_lock, self._fresh_connection():
                return facade.run_checkpoint(
                    lambda sql: self._conn.execute(sql).fetchone(),
                    mode,
                    wal_reset_safe=self.wal_reset_safe,
                    db_path=str(self._db_path),
                    db_error=self._dbapi.Error,
                )
        except OSError as exc:
            logger.warning("wal_checkpoint_lock_failed", error_type=type(exc).__name__, db=str(self._db_path))
            return CheckpointResult(busy=1, checkpointed=0, log_frames=0, mode="error")

    def graph_edges(self, namespace: str) -> list[GraphEdge]:
        with self._fresh_connection(), self._lock:
            return read_edges(cast("SQLiteBackend", self), namespace)

    def graph_edge_count(self, namespace: str) -> int:
        with self._fresh_connection(), self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM memory_graph_edges WHERE namespace = ?", (namespace,))
            return int(row.fetchone()[0])

    def ids_by_source(self, namespace: str, source_identity: str, limit: int) -> list[str]:
        with self._fresh_connection(), self._lock:
            query = "SELECT id FROM memories WHERE namespace = ? AND source_identity = ? LIMIT ?"
            return [str(row[0]) for row in self._conn.execute(query, (namespace, source_identity, limit))]

    def add_graph_edges(self, namespace: str, edges: Sequence[GraphEdge]) -> None:
        with self._fresh_connection(), self._lock:
            insert_edges(cast("SQLiteBackend", self), namespace, edges)
            if self._skip_commit_depth == 0:
                self._conn.commit()

    def _delete_vector(self, entry_id: str, namespace: str) -> None:
        delete_vector_internal(self._conn, entry_id, namespace)

    def delete_vector(self, entry_id: str, *, namespace: str) -> bool:
        with self._fresh_connection():
            return delete_vector(
                self._conn,
                self._lock,
                vec_available=self._vec_available,
                entry_id=entry_id,
                namespace=namespace,
                skip_commit=self._skip_commit_depth != 0,
            )

    def vector_exists(self, entry_id: str, *, namespace: str) -> bool:
        with self._fresh_connection():
            return vector_exists(self._conn, vec_available=self._vec_available, entry_id=entry_id, namespace=namespace)

    def existing_vector_ids(self, namespace: str | None = None) -> set[str]:
        with self._fresh_connection():
            return existing_vector_ids(self._conn, self._lock, vec_available=self._vec_available, namespace=namespace)

    def upsert_vector(
        self, entry_id: str, embedding: list[float], *, namespace: str, provenance: VectorProvenance | None = None
    ) -> None:
        with self._fresh_connection():
            upsert_vector(
                self._conn,
                self._lock,
                vec_available=self._vec_available,
                dim=self._dim,
                entry_id=entry_id,
                namespace=namespace,
                embedding=embedding,
                skip_commit=self._skip_commit_depth != 0,
                provenance=provenance,
            )

    def search_vectors(
        self, query_embedding: list[float], top_k: int = 25, namespace: str | None = None
    ) -> list[tuple[str, float]]:
        with self._fresh_connection():
            return search_vectors(
                self._conn,
                self._lock,
                vec_available=self._vec_available,
                dim=self._dim,
                query_embedding=query_embedding,
                top_k=top_k,
                namespace=namespace,
            )

    def get_stored_embeddings(self, entry_ids: list[str], *, namespace: str | None = None) -> dict[str, list[float]]:
        with self._fresh_connection():
            return get_stored_embeddings(
                self._conn, self._lock, vec_available=self._vec_available, entry_ids=entry_ids, namespace=namespace
            )

    def get_vector_records(self, entry_ids: list[str], *, namespace: str) -> dict[str, StoredVector]:
        with self._fresh_connection():
            return get_vector_records(
                self._conn, self._lock, vec_available=self._vec_available, entry_ids=entry_ids, namespace=namespace
            )

    def vector_records_or_raise(self, entry_ids: list[str], *, namespace: str) -> dict[str, StoredVector]:
        """``get_vector_records``, but a failed read raises ``StorageError`` instead of returning none."""
        try:
            with self._fresh_connection():
                return get_vector_records(
                    self._conn,
                    self._lock,
                    vec_available=self._vec_available,
                    entry_ids=entry_ids,
                    namespace=namespace,
                    strict=True,
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to read vectors of {namespace}: {exc}", path=str(self._db_path)) from exc

    def vector_space_census(self, *, namespace: str) -> dict[EmbeddingSpace | None, int] | None:
        with self._fresh_connection():
            return vector_space_census(self._conn, self._lock, vec_available=self._vec_available, namespace=namespace)

    def recent_vector_records(self, *, namespace: str, limit: int) -> dict[str, StoredVector]:
        """``StorageBackend.recent_vector_records`` selecting ids only (same order as ``list_entries``)."""
        if not self._vec_available or limit <= 0:
            return {}
        try:
            with self._fresh_connection(), self._lock:
                rows = self._conn.execute(
                    "SELECT id FROM memories WHERE namespace = ? AND status = ? "
                    "ORDER BY updated_at DESC, id DESC LIMIT ?",
                    (namespace, MemoryStatus.ACTIVE.value, limit),
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to list recent entry ids: {exc}", path=str(self._db_path)) from exc
        return self.get_vector_records([str(row[0]) for row in rows], namespace=namespace)

    def hype_sibling_ids(self, parent_id: str, *, namespace: str) -> list[str]:
        with self._fresh_connection():
            return hype_sibling_ids(
                self._conn, self._lock, vec_available=self._vec_available, parent_id=parent_id, namespace=namespace
            )

    def delete_hype_siblings(self, parent_id: str, *, namespace: str) -> int:
        with self.transaction():
            return delete_hype_siblings(
                self._conn,
                self._lock,
                vec_available=self._vec_available,
                parent_id=parent_id,
                namespace=namespace,
                skip_commit=self._skip_commit_depth != 0,
            )

    def namespace_change_token(self, namespace: str) -> NamespaceChangeToken:
        """``StorageBackend.namespace_change_token``: two index seeks (see ``_change_feed``)."""
        return change_token(cast("SQLiteBackend", self), namespace)

    def entries_changed_since(
        self, namespace: str, token: NamespaceChangeToken, *, limit: int
    ) -> list[MemoryEntry] | None:
        """``StorageBackend.entries_changed_since`` over the rowid and updated_at indexes."""
        return entries_changed_since(cast("SQLiteBackend", self), namespace, token, limit=limit)
