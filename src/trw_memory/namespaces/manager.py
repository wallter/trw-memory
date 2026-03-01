"""NamespaceManager — high-level namespace operations backed by a storage backend."""

from __future__ import annotations

from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.sqlite_backend import SQLiteBackend


class NamespaceManager:
    """Manage namespace lifecycle (list, register, delete) against a storage backend.

    Args:
        backend: The SQLiteBackend to operate on.
    """

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend

    def register(self, ns: str) -> str:
        """Validate and register a namespace.

        This is idempotent — registering an existing namespace is a no-op.

        Args:
            ns: Namespace to register.

        Returns:
            The validated namespace string.

        Raises:
            ConfigError: If the namespace pattern is invalid.
        """
        return validate_namespace(ns)

    def list_namespaces(self) -> list[str]:
        """Return all distinct namespaces with stored entries.

        Returns:
            Sorted list of namespace strings.
        """
        return self._backend.list_namespaces()

    def delete(self, ns: str) -> int:
        """Delete all entries in a namespace.

        Args:
            ns: Namespace to clear.

        Returns:
            Number of entries deleted.

        Raises:
            ConfigError: If the namespace pattern is invalid.
        """
        validate_namespace(ns)
        return self._backend.delete_by_namespace(ns)

    def count(self, ns: str) -> int:
        """Return the number of entries in a namespace.

        Args:
            ns: Namespace to count.

        Returns:
            Entry count.

        Raises:
            ConfigError: If the namespace pattern is invalid.
        """
        validate_namespace(ns)
        return self._backend.count(namespace=ns)
