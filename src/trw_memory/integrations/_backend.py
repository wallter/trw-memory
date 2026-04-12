"""Shared sync backend bridge for integration adapters.

Adapters need sync operations but :class:`~trw_memory.client.MemoryClient` is
async.  This module provides a thin sync wrapper around
:class:`~trw_memory.storage.interface.StorageBackend` that all adapters share.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from trw_memory.exceptions import EncryptionUnavailableError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

__all__ = [
    "DEFAULT_LIST_LIMIT",
    "discover_namespace_backends",
    "ROLE_TAG_PREFIX",
    "create_backend",
    "create_backend_from_config",
    "make_entry",
    "resolve_backend",
]

#: Default limit for ``list_entries`` calls across all adapters.
DEFAULT_LIST_LIMIT: int = 10_000

#: Shared tag prefix for message roles (used by LangChain + LlamaIndex).
ROLE_TAG_PREFIX: str = "role:"
_SQLITE_ENCRYPTION_UNAVAILABLE = (
    "SQLCipher is required when memory_encryption_enabled=True. Install with: pip install trw-memory[encryption]"
)


def _make_id() -> str:
    """Generate a unique memory ID with ``M-`` prefix and 16 hex characters.

    Uses 64 bits of entropy from UUID4, giving collision probability
    < 0.0001% at 1 million entries (birthday paradox).
    """
    return f"M-{uuid.uuid4().hex[:16]}"


def resolve_backend(
    namespace: str,
    storage_path: str | None,
    backend: StorageBackend | None,
) -> tuple[StorageBackend, bool]:
    """Return a ``(backend, owns_backend)`` pair.

    If *backend* is provided, returns it without ownership.  Otherwise
    creates a new backend from *namespace* and *storage_path*.

    Args:
        namespace: Storage namespace.
        storage_path: Override for storage directory (or ``None``).
        backend: Pre-existing backend (for testing), or ``None``.

    Returns:
        Tuple of ``(backend_instance, owns_backend)``.
    """
    if backend is not None:
        return backend, False
    return create_backend(namespace, storage_path), True


def create_backend(
    namespace: str,
    storage_path: str | None = None,
) -> StorageBackend:
    """Create a sync :class:`StorageBackend` for the given namespace.

    Args:
        namespace: Isolation scope (e.g. ``"default"``, ``"project:my-app"``).
        storage_path: Override for the storage directory.  Falls back to
            :class:`MemoryConfig` defaults if ``None``.

    Returns:
        A ready-to-use :class:`StorageBackend` instance.
    """
    if storage_path is not None:
        config = MemoryConfig(storage_path=storage_path)
    else:
        config = MemoryConfig()

    return create_backend_from_config(config, namespace)


def create_backend_from_config(
    config: MemoryConfig,
    namespace: str,
) -> StorageBackend:
    """Create a sync :class:`StorageBackend` from an existing config object."""
    base = Path(config.storage_path)
    ns_dir = namespace.replace(":", "_")

    if config.storage_backend == "sqlite":
        if config.encryption_enabled:
            raise EncryptionUnavailableError(_SQLITE_ENCRYPTION_UNAVAILABLE)
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        db_path = base / ns_dir / config.sqlite_db_name
        return SQLiteBackend(db_path=db_path, dim=config.embedding_dim)

    from trw_memory.storage.yaml_backend import YAMLBackend

    entries_dir = base / ns_dir / "entries"
    return YAMLBackend(entries_dir=entries_dir)


@contextmanager
def discover_namespace_backends(
    config: MemoryConfig,
) -> Iterator[list[tuple[list[str], StorageBackend]]]:
    """Open every on-disk namespace store and expose its actual namespaces.

    The local backend layout uses one directory per namespace, but the directory
    name is a lossy encoding of the namespace string. To build truthful
    cross-namespace views we must open each store and read the stored namespace
    value rather than guessing it from the folder name.
    """
    from contextlib import ExitStack

    base = Path(config.storage_path)
    if not base.exists():
        yield []
        return

    with ExitStack() as stack:
        stores: list[tuple[list[str], StorageBackend]] = []

        if config.storage_backend == "sqlite":
            if config.encryption_enabled:
                raise EncryptionUnavailableError(_SQLITE_ENCRYPTION_UNAVAILABLE)
            from trw_memory.storage.sqlite_backend import SQLiteBackend

            for candidate in sorted(base.iterdir()):
                db_path = candidate / config.sqlite_db_name
                if not candidate.is_dir() or not db_path.exists():
                    continue
                store_backend: StorageBackend = stack.enter_context(
                    SQLiteBackend(db_path=db_path, dim=config.embedding_dim)
                )
                namespaces = store_backend.list_namespaces()
                if namespaces:
                    stores.append((namespaces, store_backend))
        else:
            from trw_memory.storage.yaml_backend import YAMLBackend

            for candidate in sorted(base.iterdir()):
                entries_dir = candidate / "entries"
                if not candidate.is_dir() or not entries_dir.is_dir():
                    continue
                yaml_backend: StorageBackend = stack.enter_context(YAMLBackend(entries_dir=entries_dir))
                namespaces = yaml_backend.list_namespaces()
                if namespaces:
                    stores.append((namespaces, yaml_backend))

        yield stores


def make_entry(
    content: str,
    *,
    namespace: str = "default",
    tags: list[str] | None = None,
    importance: float = 0.5,
    detail: str = "",
    metadata: dict[str, str] | None = None,
    source: str = "agent",
) -> MemoryEntry:
    """Create a new :class:`MemoryEntry` with generated ID and timestamps."""
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=_make_id(),
        content=content,
        detail=detail,
        tags=tags or [],
        importance=importance,
        namespace=namespace,
        metadata=metadata or {},
        created_at=now,
        updated_at=now,
        source=source,  # type: ignore[arg-type]  # validator coerces unknown values to "agent"
    )
