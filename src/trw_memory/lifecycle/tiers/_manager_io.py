"""Backend-opening and warm-entry loading helpers for TierManager."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import refuse_encryption_at_rest
from trw_memory.models.config import MemoryConfig

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
    from trw_memory.integrations._backend import quarantine_ledger_for

    refuse_encryption_at_rest(config)
    db_path = base_dir / config.sqlite_db_name
    if config.storage_backend == "sqlite" and db_path.exists():
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        return SQLiteBackend(
            db_path,
            dim=config.embedding_dim,
            recovery_policy=config.memory_recovery_policy,
            corrupt_backup_keep=config.memory_corrupt_backup_keep,
            rebuild_from_cold=config.memory_recovery_rebuild_from_cold,
            recovery_inline_max_bytes=config.memory_recovery_inline_max_bytes,
            quarantine_ledger=quarantine_ledger_for(config),
        )

    from trw_memory.storage.yaml_backend import YAMLBackend

    return YAMLBackend(entries_dir, quarantine_ledger=quarantine_ledger_for(config))
