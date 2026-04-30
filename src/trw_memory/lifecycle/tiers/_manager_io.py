"""Backend-opening and warm-entry loading helpers for TierManager."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.encryption import derive_namespace_key
from trw_memory.security.keys import get_master_key
from trw_memory.storage.persistence import read_yaml

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)


def open_canonical_backend(
    base_dir: Path,
    entries_dir: Path,
    namespace: str,
    config: MemoryConfig,
) -> StorageBackend:
    """Open the canonical backend used for cold-tier promotion and sweep."""
    db_path = base_dir / config.sqlite_db_name
    if config.storage_backend == "sqlite" and db_path.exists():
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        sqlcipher_key_hex: str | None = None
        if config.encryption_enabled:
            master_key = get_master_key(config)
            sqlcipher_key_hex = derive_namespace_key(master_key, namespace)

        return SQLiteBackend(
            db_path,
            dim=config.embedding_dim,
            sqlcipher_key_hex=sqlcipher_key_hex,
            recovery_policy=config.memory_recovery_policy,
            corrupt_backup_keep=config.memory_corrupt_backup_keep,
            rebuild_from_cold=config.memory_recovery_rebuild_from_cold,
        )

    from trw_memory.storage.yaml_backend import YAMLBackend

    return YAMLBackend(entries_dir)


def load_warm_entries(
    base_dir: Path,
    entries_dir: Path,
    namespace: str,
    config: MemoryConfig,
) -> tuple[list[dict[str, object]], int]:
    """Load canonical entries for warm/cold sweep evaluation."""
    db_path = base_dir / config.sqlite_db_name
    if config.storage_backend == "sqlite" and db_path.exists():
        try:
            with open_canonical_backend(base_dir, entries_dir, namespace, config) as backend:
                backend_entries = backend.list_entries(limit=max(backend.count(), config.hot_max_entries * 8, 200))
            return [entry.model_dump(mode="json") for entry in backend_entries], 0
        except (OSError, StorageError, ValueError):
            logger.warning("tier_sweep_backend_scan_failed", namespace=namespace, exc_info=True)
            return [], 1

    if not entries_dir.exists():
        return [], 0

    entries: list[dict[str, object]] = []
    errors = 0
    for yaml_file in sorted(entries_dir.glob("*.yaml")):
        if yaml_file.name == "index.yaml":
            continue
        try:
            entries.append(read_yaml(yaml_file))
        except (OSError, StorageError, ValueError):
            logger.warning("tier_sweep_entry_scan_failed", path=str(yaml_file), exc_info=True)
            errors += 1
    return entries, errors
