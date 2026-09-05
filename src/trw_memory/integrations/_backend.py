"""Shared sync backend bridge for integration adapters.

Adapters need sync operations but :class:`~trw_memory.client.MemoryClient` is
async.  This module provides a thin sync wrapper around
:class:`~trw_memory.storage.interface.StorageBackend` that all adapters share.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import ConfigError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.entry_factory import local_node_id_for, new_entry
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.encryption import derive_namespace_key
from trw_memory.security.keys import get_master_key

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

__all__ = [
    "DEFAULT_LIST_LIMIT",
    "ROLE_TAG_PREFIX",
    "config_for_storage_path",
    "create_backend",
    "create_backend_from_config",
    "discover_namespace_backends",
    "make_entry",
    "resolve_backend",
    "resolve_backend_db_path",
    "resolve_backend_location",
]

#: Default limit for ``list_entries`` calls across all adapters.
DEFAULT_LIST_LIMIT: int = 10_000

#: Shared tag prefix for message roles (used by LangChain + LlamaIndex).
ROLE_TAG_PREFIX: str = "role:"
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


def _read_namespace_metadata(namespace_dir: Path) -> str | None:
    metadata_path = namespace_dir / _NAMESPACE_METADATA_FILE
    if not metadata_path.exists():
        return None
    # Fail open: a single namespace whose ``namespace.txt`` is unreadable
    # (OSError) or non-UTF-8 (UnicodeDecodeError from a torn/partial write)
    # must not abort ``discover_namespace_backends`` for every OTHER namespace.
    # The caller treats ``None`` as "skip this namespace" with a content-free
    # warning, isolating one corrupt sidecar like any other discovery miss.
    # Never log the decoded text or raw bytes — the stored namespace string can
    # carry sensitive project identifiers; only the path + error class.
    try:
        namespace = metadata_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "namespace_metadata_read_failed",
            path=str(metadata_path),
            error=type(exc).__name__,
        )
        return None
    return namespace or None


def _create_sqlite_backend(
    config: MemoryConfig,
    db_path: Path,
    *,
    sqlcipher_key_hex: str | None,
) -> StorageBackend:
    """Create SQLite storage with the canonical recovery and dimension settings."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    return SQLiteBackend(
        db_path=db_path,
        dim=config.embedding_dim,
        sqlcipher_key_hex=sqlcipher_key_hex,
        recovery_policy=config.memory_recovery_policy,
        corrupt_backup_keep=config.memory_corrupt_backup_keep,
        rebuild_from_cold=config.memory_recovery_rebuild_from_cold,
        recovery_inline_max_bytes=config.memory_recovery_inline_max_bytes,
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


def _refuse_encrypted_single_store(config: MemoryConfig) -> None:
    """Re-assert the config-level refusal at the point of use.

    :class:`~trw_memory.models.config.MemoryConfig` already rejects
    ``encryption_enabled`` together with ``memory_single_store_path``, but a
    caller can mutate a validated model afterwards (pydantic does not validate
    on assignment here), and this module is where the wrong key would actually
    be handed to SQLCipher. The cost is one boolean; the failure it prevents is
    an unopenable store.
    """
    if config.encryption_enabled and config.memory_single_store_path:
        raise ConfigError(
            "refusing to open a single shared store with per-namespace encryption: "
            "encryption_enabled and memory_single_store_path are mutually exclusive until "
            "PRD-CORE-253 FR09 ships a per-file key."
        )


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
) -> StorageBackend:
    """Create a sync :class:`StorageBackend` from an existing config object.

    When ``db_path_override`` is provided (SQLite only), the explicit file path
    is used directly and the ``base / namespace_dir / sqlite_db_name`` join is
    bypassed. The ``namespace`` argument still governs the row ``namespace``
    column and the sidecar ``namespace.txt`` metadata, so callers can decouple
    the on-disk directory layout from the queried namespace.
    """
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
        sqlcipher_key_hex: str | None = None
        if config.encryption_enabled:
            # A per-NAMESPACE key against a shared FILE is unopenable for every
            # namespace after the first, so the combination is refused rather
            # than written (PRD-CORE-253 FR09 owns the redesign).
            _refuse_encrypted_single_store(config)
            master_key = get_master_key(config)
            sqlcipher_key_hex = derive_namespace_key(master_key, namespace)
        return _create_sqlite_backend(config, db_path, sqlcipher_key_hex=sqlcipher_key_hex)

    from trw_memory.storage.yaml_backend import YAMLBackend

    namespace_dir = base / ns_dir
    _write_namespace_metadata(namespace_dir, namespace)
    entries_dir = namespace_dir / "entries"
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

    if config.memory_single_store_path:
        # One file, every namespace. Directory scanning finds nothing here (the
        # store is a FILE in ``base``, not a subdirectory), so discovery has to
        # ask the store which namespaces it holds -- which is the truthful
        # source anyway, since the namespace is a column and not a filename.
        _refuse_encrypted_single_store(config)
        single = Path(config.memory_single_store_path)
        if not single.exists():
            yield []
            return
        with ExitStack() as stack:
            # Keyless is only correct because the guard above proved the store is
            # not encrypted. Passing None to an ENCRYPTED store would not read
            # plaintext -- it would fail to open, which is a confusing way to
            # report a configuration that should never have been accepted.
            store = stack.enter_context(_create_sqlite_backend(config, single, sqlcipher_key_hex=None))
            namespaces = store.list_namespaces()
            yield [(namespaces, store)] if namespaces else []
        return

    if not base.exists():
        yield []
        return

    with ExitStack() as stack:
        stores: list[tuple[list[str], StorageBackend]] = []

        if config.storage_backend == "sqlite":
            master_key: bytes | None = get_master_key(config) if config.encryption_enabled else None

            for candidate in sorted(base.iterdir()):
                db_path = candidate / config.sqlite_db_name
                if not candidate.is_dir() or not db_path.exists():
                    continue
                sqlcipher_key_hex: str | None = None
                if master_key is not None:
                    namespace = _read_namespace_metadata(candidate)
                    if namespace is None:
                        logger.warning("encrypted_namespace_discovery_skipped", path=str(candidate))
                        continue
                    sqlcipher_key_hex = derive_namespace_key(master_key, namespace)
                store_backend = stack.enter_context(
                    _create_sqlite_backend(config, db_path, sqlcipher_key_hex=sqlcipher_key_hex)
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
