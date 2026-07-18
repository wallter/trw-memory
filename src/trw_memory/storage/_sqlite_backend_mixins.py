"""WAL checkpoint and vector-operation mixins for ``SQLiteBackend``."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import structlog

from trw_memory.storage._vector_ops import (
    delete_hype_siblings,
    delete_vector,
    delete_vector_internal,
    existing_vector_ids,
    get_stored_embeddings,
    hype_sibling_ids,
    search_vectors,
    upsert_vector,
    vector_exists,
)
from trw_memory.storage._wal_checkpoint import CheckpointResult

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

    def checkpoint_wal(self, mode: str = "TRUNCATE") -> CheckpointResult:
        """Checkpoint the owning connection under the backend lock; fail open."""
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
            return CheckpointResult(busy=1, checkpointed=0, mode="error")

    def _delete_vector(self, entry_id: str) -> None:
        delete_vector_internal(self._conn, entry_id)

    def delete_vector(self, entry_id: str) -> bool:
        with self._fresh_connection():
            return delete_vector(
                self._conn,
                self._lock,
                vec_available=self._vec_available,
                entry_id=entry_id,
                skip_commit=self._skip_commit_depth != 0,
            )

    def vector_exists(self, entry_id: str) -> bool:
        with self._fresh_connection():
            return vector_exists(self._conn, vec_available=self._vec_available, entry_id=entry_id)

    def existing_vector_ids(self, namespace: str | None = None) -> set[str]:
        with self._fresh_connection():
            return existing_vector_ids(self._conn, self._lock, vec_available=self._vec_available, namespace=namespace)

    def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:
        with self._fresh_connection():
            upsert_vector(
                self._conn,
                self._lock,
                vec_available=self._vec_available,
                dim=self._dim,
                entry_id=entry_id,
                embedding=embedding,
                skip_commit=self._skip_commit_depth != 0,
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

    def get_stored_embeddings(self, entry_ids: list[str]) -> dict[str, list[float]]:
        with self._fresh_connection():
            return get_stored_embeddings(self._conn, self._lock, vec_available=self._vec_available, entry_ids=entry_ids)

    def hype_sibling_ids(self, parent_id: str) -> list[str]:
        with self._fresh_connection():
            return hype_sibling_ids(self._conn, self._lock, vec_available=self._vec_available, parent_id=parent_id)

    def delete_hype_siblings(self, parent_id: str) -> int:
        with self._fresh_connection():
            return delete_hype_siblings(
                self._conn,
                self._lock,
                vec_available=self._vec_available,
                parent_id=parent_id,
                skip_commit=self._skip_commit_depth != 0,
            )
