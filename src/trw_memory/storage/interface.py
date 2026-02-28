"""StorageBackend ABC — defines the contract for all storage implementations.

Any class that subclasses :class:`StorageBackend` can be used as a drop-in
replacement.  Two implementations are provided:

- :class:`~trw_memory.storage.sqlite_backend.SQLiteBackend` — primary, fast
- :class:`~trw_memory.storage.yaml_backend.YAMLBackend` — portable fallback
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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
