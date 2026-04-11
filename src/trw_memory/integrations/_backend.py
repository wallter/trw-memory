"""Shared sync backend bridge for integration adapters.

Adapters need sync operations but :class:`~trw_memory.client.MemoryClient` is
async.  This module provides a thin sync wrapper around
:class:`~trw_memory.storage.interface.StorageBackend` that all adapters share.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

__all__ = [
    "DEFAULT_LIST_LIMIT",
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
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        db_path = base / ns_dir / config.sqlite_db_name
        return SQLiteBackend(db_path=db_path, dim=config.embedding_dim)

    from trw_memory.storage.yaml_backend import YAMLBackend

    entries_dir = base / ns_dir / "entries"
    return YAMLBackend(entries_dir=entries_dir)


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
