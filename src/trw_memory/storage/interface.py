"""StorageBackend ABC — defines the contract for all storage implementations.

Any class that subclasses :class:`StorageBackend` can be used as a drop-in
replacement.  Two implementations are provided:

- :class:`~trw_memory.storage.sqlite_backend.SQLiteBackend` — primary, fast
- :class:`~trw_memory.storage.yaml_backend.YAMLBackend` — portable fallback
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

from trw_memory.models.memory import MemoryEntry, MemoryStatus


class StorageBackend(ABC):
    """Abstract base class for memory storage backends.

    All mutating operations are synchronous.  Backends are responsible for
    their own thread-safety; callers must not share backend instances across
    threads without external synchronisation.
    """

    @abstractmethod
    def store(self, entry: MemoryEntry) -> None:
        """Persist a new entry (or replace an existing one with the same id).

        Args:
            entry: The memory entry to persist.

        Raises:
            StorageError: If the write fails.
        """
        ...

    @abstractmethod
    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve an entry by its unique identifier.

        Args:
            entry_id: The entry's ``id`` field.

        Returns:
            The :class:`MemoryEntry`, or ``None`` if not found.

        Raises:
            StorageError: If the read fails.
        """
        ...

    @abstractmethod
    def update(self, entry_id: str, **fields: object) -> MemoryEntry | None:
        """Apply a partial update to an existing entry.

        Args:
            entry_id: Target entry identifier.
            **fields: Field names and new values.  Only supplied fields are
                changed; all others retain their current values.

        Returns:
            The updated :class:`MemoryEntry`, or ``None`` if not found.

        Raises:
            StorageError: If the update fails.
        """
        ...

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Remove an entry from storage.

        Args:
            entry_id: The entry to remove.

        Returns:
            ``True`` if the entry existed and was deleted, ``False`` otherwise.

        Raises:
            StorageError: If the deletion fails.
        """
        ...

    @abstractmethod
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
        """Keyword search over content and detail fields.

        Args:
            query: Free-text search string.
            top_k: Maximum number of results to return.
            tags: If provided, entries must contain ALL of these tags.
            status: If provided, filter to entries with this status.
            min_importance: Lower bound on ``importance`` (inclusive).
            namespace: If provided, restrict to this namespace.

        Returns:
            Up to *top_k* matching entries, ordered by relevance (descending).

        Raises:
            StorageError: If the query fails.
        """
        ...

    @abstractmethod
    def count(self, namespace: str | None = None) -> int:
        """Return the total number of stored entries.

        Args:
            namespace: If provided, count only entries in this namespace.

        Returns:
            Number of entries.

        Raises:
            StorageError: If the count query fails.
        """
        ...

    @abstractmethod
    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Return entries with optional filters.

        Args:
            status: If provided, only return entries with this status.
            namespace: If provided, only return entries in this namespace.
            limit: Maximum number of entries to return.

        Returns:
            Up to *limit* entries ordered by ``updated_at`` descending.

        Raises:
            StorageError: If the query fails.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any held resources (file handles, DB connections, etc.).

        Safe to call multiple times.
        """
        ...

    # -- Non-abstract extension points (safe defaults) ----------------------

    def list_namespaces(self) -> list[str]:
        """Return all distinct namespaces that have stored entries.

        Subclasses that support multi-namespace storage should override this
        to query their underlying store.  The default returns an empty list,
        which is safe for single-namespace or in-memory backends.

        Returns:
            Sorted list of unique namespace strings.  Empty if the backend
            does not track namespaces or has no entries.
        """
        return []

    def delete_by_namespace(self, namespace: str) -> int:
        """Delete all entries belonging to *namespace*.

        Subclasses that support bulk-delete should override this for
        efficiency.  The default returns ``0`` (no entries deleted) and
        performs no I/O.

        Args:
            namespace: The namespace whose entries should be removed.

        Returns:
            Number of entries actually deleted.  ``0`` if the namespace
            does not exist or the backend does not support this operation.
        """
        return 0

    def increment_session_counts(self, entry_ids: list[str], *, updated_at: datetime | None = None) -> int:
        """Increment ``session_count`` for multiple entries.

        Backends that support bulk mutation should override this to perform the
        work in a single transaction. The default is a safe no-op.

        Args:
            entry_ids: Distinct entry ids to increment.
            updated_at: Optional timestamp to stamp onto updated rows.

        Returns:
            Number of rows updated.
        """
        return 0

    def increment_recall_access(self, entry_ids: list[str], *, accessed_at: datetime | None = None) -> int:
        """Increment ``access_count`` and ``recall_count`` for recalled entries.

        F-008: backends that support bulk mutation should override this to do
        the work in a single statement / commit. The default falls back to a
        per-entry get+update loop so non-SQLite backends keep correct
        semantics (each distinct id incremented once).

        Args:
            entry_ids: Entry ids that were surfaced by recall (may contain dups).
            accessed_at: Timestamp to stamp onto ``last_accessed_at``.

        Returns:
            Number of entries updated.
        """
        seen: set[str] = set()
        updated = 0
        for entry_id in entry_ids:
            if entry_id in seen:
                continue
            seen.add(entry_id)
            entry = self.get(entry_id)
            if entry is None:
                continue
            self.update(
                entry_id,
                access_count=entry.access_count + 1,
                recall_count=entry.recall_count + 1,
                last_accessed_at=accessed_at,
            )
            updated += 1
        return updated

    @contextlib.contextmanager
    def transaction(self) -> Iterator[StorageBackend]:
        """Optional batching context — backends that support transactions
        should override.

        PRD-FIX-088 FR02: callers wrap a series of writes in
        ``with backend.transaction(): ...`` to collapse N implicit
        per-call commits into one explicit commit.  The default
        implementation is a no-op pass-through so callers don't need
        ``hasattr`` guards; non-supporting backends still see N implicit
        commits, which is correct (just slower).
        """
        yield self

    def upsert_vector(self, entry_id: str, embedding: list[float]) -> None:  # noqa: B027
        """Insert or update a dense vector associated with *entry_id*.

        Backends that support vector search (e.g. via ``sqlite-vec``) should
        override this.  The default is a silent no-op so that callers do not
        need to guard against missing vector support.

        Args:
            entry_id: The memory entry id to associate the vector with.
            embedding: Dense float vector.  Length must match the backend's
                configured dimensionality.

        Raises:
            StorageError: If the upsert fails (only in overriding backends).
        """

    def delete_vector(self, entry_id: str) -> bool:
        """Delete the dense vector associated with *entry_id*.

        Backends without vector support return ``False``.
        """
        return False

    def vector_exists(self, entry_id: str) -> bool:
        """Return whether a dense vector currently exists for *entry_id*."""
        return False

    def existing_vector_ids(self) -> set[str]:
        """Return the set of entry IDs that currently have a stored vector.

        Default returns an empty set so callers can opt into batch backfill
        skipping without branching on backend capabilities.
        """
        return set()

    def search_vectors(
        self,
        query_embedding: list[float],
        top_k: int = 25,
    ) -> list[tuple[str, float]]:
        """KNN search over stored dense vectors.

        Backends that support vector search should override this.  The default
        returns an empty list so that callers can always call this method
        without checking for vector support.

        Args:
            query_embedding: Query vector.  Length must match the backend's
                configured dimensionality.
            top_k: Maximum number of nearest neighbours to return.

        Returns:
            List of ``(entry_id, distance)`` tuples sorted by distance
            ascending (closest first).  Empty if the backend has no vector
            support or no vectors are stored.

        Raises:
            StorageError: If the search fails (only in overriding backends).
        """
        return []

    def get_stored_embeddings(self, entry_ids: list[str]) -> dict[str, list[float]]:
        """Return stored dense vectors for the requested entry IDs.

        Backends with vector persistence should override this. The default
        returns an empty mapping so callers can opt into dense retrieval
        without branching on backend capabilities.
        """
        return {}

    # -- Context manager (non-abstract) ------------------------------------

    def __enter__(self) -> StorageBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
