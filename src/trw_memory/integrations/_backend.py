"""Shared sync backend bridge for integration adapters.

Adapters need sync operations but :class:`~trw_memory.client.MemoryClient` is
async.  This module provides a thin sync wrapper around
:class:`~trw_memory.storage.interface.StorageBackend` that all adapters share.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import refuse_encryption_at_rest
from trw_memory.models.config import MemoryConfig
from trw_memory.models.entry_factory import local_node_id_for, new_entry
from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

__all__ = [
    "NamespaceStoreLocation",
    "config_for_storage_path",
    "create_backend",
    "create_backend_from_config",
    "discover_namespace_backends",
    "make_entry",
    "namespace_store_locations",
    "open_namespace_store",
    "resolve_backend_db_path",
    "resolve_backend_location",
]

_NAMESPACE_METADATA_FILE = "namespace.txt"
logger = structlog.get_logger(__name__)


def _make_id() -> str:
    """Generate a unique memory ID with ``M-`` prefix and 16 hex characters.

    Uses 64 bits of entropy from UUID4, giving collision probability
    < 0.0001% at 1 million entries (birthday paradox).
    """
    return f"M-{uuid.uuid4().hex[:16]}"


def _write_namespace_metadata(namespace_dir: Path, namespace: str) -> None:
    namespace_dir.mkdir(parents=True, exist_ok=True)
    (namespace_dir / _NAMESPACE_METADATA_FILE).write_text(namespace, encoding="utf-8")


def _create_sqlite_backend(
    config: MemoryConfig,
    db_path: Path,
    *,
    check_integrity_once: bool = False,
) -> StorageBackend:
    """Create SQLite storage with the canonical recovery and dimension settings."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    refuse_encryption_at_rest(config)

    return SQLiteBackend(
        db_path=db_path,
        dim=config.embedding_dim,
        recovery_policy=config.memory_recovery_policy,
        corrupt_backup_keep=config.memory_corrupt_backup_keep,
        rebuild_from_cold=config.memory_recovery_rebuild_from_cold,
        recovery_inline_max_bytes=config.memory_recovery_inline_max_bytes,
        check_integrity_once=check_integrity_once,
    )


def config_for_storage_path(storage_path: str | None = None) -> MemoryConfig:
    """Build the config an adapter's backend would be created from.

    Adapters need this beyond backend creation: ``security.write_gate`` anchors
    the audit log, quarantine store and provenance key off the same config, so
    an adapter pointed at a custom ``storage_path`` must not scatter its security
    artifacts into the default location.
    """
    if storage_path is not None:
        return MemoryConfig(storage_path=storage_path)
    return MemoryConfig()


def resolve_backend_db_path(config: MemoryConfig, namespace: str) -> Path:
    """Return the SQLite file a namespace's backend resolves to.

    ``memory_single_store_path`` wins when set: every namespace then resolves to
    ONE file, which is what makes PRD-CORE-253 FR01's "one memory.db per user
    account" true rather than aspirational. It is safe because PRD-CORE-245 FR01
    keys a row on ``(namespace, id)``. Otherwise the historical
    ``base / namespace_dir / sqlite_db_name`` join applies.

    The join lived only inside :func:`create_backend_from_config`, so a caller
    that needed to know whether TWO namespaces share one file had no way to ask
    (FR05: a namespace rename is a single-file transaction when they do and a
    two-store move when they do not). One join, one source of truth.
    """
    if config.memory_single_store_path:
        return Path(config.memory_single_store_path)
    return Path(config.storage_path) / namespace.replace(":", "_") / config.sqlite_db_name


def resolve_backend_location(config: MemoryConfig, namespace: str) -> Path:
    """Return the on-disk location a namespace's rows live in, per backend.

    SQLite namespaces share a location when they resolve to the same FILE; YAML
    namespaces share one when they resolve to the same ENTRIES DIRECTORY. The
    two are different questions, and answering the YAML one with the SQLite rule
    is how a cross-namespace move can silently become a no-op against the wrong
    store.
    """
    if config.storage_backend == "sqlite":
        return resolve_backend_db_path(config, namespace)
    return Path(config.storage_path) / namespace.replace(":", "_") / "entries"


def create_backend(
    namespace: str,
    storage_path: str | None = None,
    db_path_override: Path | str | None = None,
) -> StorageBackend:
    """Create a sync :class:`StorageBackend` for the given namespace.

    Args:
        namespace: Isolation scope (e.g. ``"default"``, ``"project:my-app"``).
        storage_path: Override for the storage directory.  Falls back to
            :class:`MemoryConfig` defaults if ``None``.
        db_path_override: Explicit absolute SQLite file path that BYPASSES the
            ``base / namespace_dir / sqlite_db_name`` join. Use to land rows in
            a fixed file while keeping ``namespace`` independent of the on-disk
            directory name (e.g. trw-distill seeding the MCP-read flat store at
            ``<trw_dir>/memory/memory.db`` under ``namespace="default"``).
            SQLite backend only.

    Returns:
        A ready-to-use :class:`StorageBackend` instance.
    """
    config = config_for_storage_path(storage_path)
    return create_backend_from_config(config, namespace, db_path_override=db_path_override)


def create_backend_from_config(
    config: MemoryConfig,
    namespace: str,
    db_path_override: Path | str | None = None,
    *,
    check_integrity_once: bool = False,
) -> StorageBackend:
    """Create a sync :class:`StorageBackend` from an existing config object.

    ``check_integrity_once`` is for the daemon's recall path only; see
    ``storage._connection.open_and_configure``.

    When ``db_path_override`` is provided (SQLite only), the explicit file path
    is used directly and the ``base / namespace_dir / sqlite_db_name`` join is
    bypassed. The ``namespace`` argument still governs the row ``namespace``
    column and the sidecar ``namespace.txt`` metadata, so callers can decouple
    the on-disk directory layout from the queried namespace.
    """
    refuse_encryption_at_rest(config)  # before anything touches disk (C12)
    base = Path(config.storage_path)
    ns_dir = namespace.replace(":", "_")

    if config.storage_backend == "sqlite":
        if db_path_override is not None:
            db_path = Path(db_path_override)
        else:
            db_path = resolve_backend_db_path(config, namespace)
        if config.memory_single_store_path:
            # One file holds every namespace, so a per-directory ``namespace.txt``
            # would be N namespaces overwriting one sidecar with the last writer's
            # name. The namespace is the row's own column; discovery reads it from
            # the store (``list_namespaces``) rather than from a filename.
            db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            _write_namespace_metadata(db_path.parent, namespace)
        return _create_sqlite_backend(config, db_path, check_integrity_once=check_integrity_once)

    from trw_memory.storage.yaml_backend import YAMLBackend

    namespace_dir = base / ns_dir
    _write_namespace_metadata(namespace_dir, namespace)
    entries_dir = namespace_dir / "entries"
    return YAMLBackend(entries_dir=entries_dir)


@dataclass(frozen=True)
class NamespaceStoreLocation:
    """One on-disk SQLite namespace store."""

    db_path: Path


def namespace_store_locations(config: MemoryConfig) -> list[NamespaceStoreLocation]:
    """Every on-disk SQLite namespace store, found WITHOUT opening any of them.

    Costs a directory listing plus a stat per store, so a per-write caller
    (cross-project validation) can enumerate siblings and open only the ones it
    must read. The namespaces a store holds are only known once it is opened
    (:func:`open_namespace_store` + ``list_namespaces``); a folder name is a
    lossy encoding of them. Non-SQLite backends have no vector stores to find.
    """
    refuse_encryption_at_rest(config)  # every store here is opened keyless (C12)
    if config.memory_single_store_path:
        # One file holds every namespace; directory scanning would find nothing
        # (the store is a FILE in ``base``).
        single = Path(config.memory_single_store_path)
        return [NamespaceStoreLocation(single)] if single.exists() else []
    base = Path(config.storage_path)
    if config.storage_backend != "sqlite" or not base.exists():
        return []
    locations: list[NamespaceStoreLocation] = []
    for candidate in sorted(base.iterdir()):
        db_path = candidate / config.sqlite_db_name
        if candidate.is_dir() and db_path.exists():
            locations.append(NamespaceStoreLocation(db_path))
    return locations


def open_namespace_store(config: MemoryConfig, location: NamespaceStoreLocation) -> StorageBackend:
    """Open the store at *location* (a context manager, like every backend)."""
    return _create_sqlite_backend(config, location.db_path)


@contextmanager
def discover_namespace_backends(
    config: MemoryConfig,
    *,
    reuse: StorageBackend | None = None,
) -> Iterator[list[tuple[list[str], StorageBackend]]]:
    """Open every on-disk namespace store and expose its actual namespaces.

    The local backend layout uses one directory per namespace, but the directory
    name is a lossy encoding of the namespace string. To build truthful
    cross-namespace views we must open each store and read the stored namespace
    value rather than guessing it from the folder name.

    *reuse* is a backend the caller already holds open. A store at its file is
    served by it, not opened a second time, and the caller keeps closing it
    (PRD-CORE-298 FR05: under the single store every recall opened it twice).
    """
    from contextlib import ExitStack

    refuse_encryption_at_rest(config)
    reuse_path = getattr(reuse, "db_path", None)
    reuse_file = os.path.realpath(reuse_path) if reuse_path is not None else None
    if config.memory_single_store_path or config.storage_backend == "sqlite":
        with ExitStack() as stack:
            stores: list[tuple[list[str], StorageBackend]] = []
            for location in namespace_store_locations(config):
                if reuse is not None and os.path.realpath(location.db_path) == reuse_file:
                    store = reuse
                else:
                    store = stack.enter_context(open_namespace_store(config, location))
                namespaces = store.list_namespaces()
                if namespaces:
                    stores.append((namespaces, store))
            yield stores
        return

    base = Path(config.storage_path)
    if not base.exists():
        yield []
        return

    from trw_memory.storage.yaml_backend import YAMLBackend

    with ExitStack() as stack:
        yaml_stores: list[tuple[list[str], StorageBackend]] = []
        for candidate in sorted(base.iterdir()):
            entries_dir = candidate / "entries"
            if not candidate.is_dir() or not entries_dir.is_dir():
                continue
            yaml_backend: StorageBackend = stack.enter_context(YAMLBackend(entries_dir=entries_dir))
            namespaces = yaml_backend.list_namespaces()
            if namespaces:
                yaml_stores.append((namespaces, yaml_backend))
        yield yaml_stores


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
    """Create a new :class:`MemoryEntry` with generated ID and timestamps.

    PRD-CORE-245 FR08: through the shared factory, so this convenience
    constructor stamps the same causality fields every other writer does.
    """
    return new_entry(
        entry_id=_make_id(),
        content=content,
        namespace=namespace,
        local_node_id=local_node_id_for(namespace),
        fields={
            "detail": detail,
            "tags": tags or [],
            "importance": importance,
            "metadata": metadata or {},
            "source": source,
        },
    )
