"""VSCode extension interface contract.

Defines the :class:`VSCodeMemoryInterface` protocol that the VSCode extension
will implement against via REST.  Also provides :class:`LocalMemoryAdapter`,
a concrete implementation using the local storage backend for testing.

This module has **no external framework dependencies** and can be imported in
a base ``trw-memory`` install.

Usage::

    from trw_memory.integrations.vscode import VSCodeMemoryInterface, LocalMemoryAdapter

    adapter = LocalMemoryAdapter(namespace="project:my-app")
    results = adapter.get_relevant("/path/to/file.py", limit=5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from trw_memory.integrations._mixin import BackendOwnerMixin

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend


@runtime_checkable
class VSCodeMemoryInterface(Protocol):
    """Interface contract for the VSCode memory extension.

    Four operations that map to core trw-memory capabilities:

    - ``get_relevant``: Retrieve memories relevant to a source file.
    - ``store_selection``: Store a user-selected snippet or note.
    - ``search``: Full-text memory search.
    - ``get_status``: Memory store health metrics.
    """

    def get_relevant(
        self,
        file_path: str,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """Return memories relevant to the given source file path.

        Args:
            file_path: Absolute or relative path to a source file.
            limit: Maximum number of results.

        Returns:
            List of memory dicts with ``memory_id``, ``content``, ``score``.
        """
        ...

    def store_selection(
        self,
        content: str,
        file_path: str,
        tags: list[str],
    ) -> dict[str, str]:
        """Store a user-selected code snippet or note.

        Args:
            content: The text content to store.
            file_path: Source file the selection came from.
            tags: User-assigned categorisation tags.

        Returns:
            Dict with ``memory_id``, ``status``.
        """
        ...

    def search(
        self,
        query: str,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """Full-text memory search.

        Args:
            query: Free-text search string.
            namespace: Namespace to search within.  Defaults to the
                adapter's own namespace if ``None``.
            limit: Maximum number of results.

        Returns:
            List of memory dicts ordered by relevance.
        """
        ...

    def get_status(self) -> dict[str, object]:
        """Return memory store health metrics.

        Returns:
            Dict with ``entry_count``, ``namespace``, ``storage_backend``.
        """
        ...


class LocalMemoryAdapter(BackendOwnerMixin):
    """Concrete :class:`VSCodeMemoryInterface` using the local storage backend.

    Intended for local development and testing of the VSCode extension.

    Args:
        namespace: trw-memory namespace for storage isolation.
        storage_path: Override for the storage directory.
        backend: Pre-existing backend (for testing).
    """

    def __init__(
        self,
        namespace: str = "default",
        *,
        storage_path: str | None = None,
        backend: StorageBackend | None = None,
    ) -> None:
        from trw_memory.integrations._backend import resolve_backend

        self._namespace = namespace
        self._backend, self._owns_backend = resolve_backend(
            namespace, storage_path, backend,
        )

    def get_relevant(
        self,
        file_path: str,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """Return memories relevant to *file_path*."""
        entries = self._backend.search(
            file_path,
            top_k=limit,
            namespace=self._namespace,
        )
        return [
            {
                "memory_id": e.id,
                "content": e.content,
                "tags": list(e.tags),
                "score": e.importance,
            }
            for e in entries
        ]

    def store_selection(
        self,
        content: str,
        file_path: str,
        tags: list[str],
    ) -> dict[str, str]:
        """Store a code snippet or note from *file_path*."""
        from trw_memory.integrations._backend import make_entry

        entry = make_entry(
            content=content,
            namespace=self._namespace,
            tags=[*tags, f"file:{file_path}"],
            importance=0.6,
            source="human",
        )
        self._backend.store(entry)
        return {"memory_id": entry.id, "status": "stored"}

    def search(
        self,
        query: str,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """Full-text search across memories."""
        effective_ns = namespace if namespace is not None else self._namespace
        entries = self._backend.search(
            query,
            top_k=limit,
            namespace=effective_ns,
        )
        return [
            {
                "memory_id": e.id,
                "content": e.content,
                "tags": list(e.tags),
                "score": e.importance,
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ]

    def get_status(self) -> dict[str, object]:
        """Return memory store health metrics."""
        count = self._backend.count(namespace=self._namespace)
        return {
            "entry_count": count,
            "namespace": self._namespace,
            "storage_backend": type(self._backend).__name__,
        }

    # Resource management inherited from BackendOwnerMixin.
