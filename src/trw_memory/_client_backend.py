"""Late-bound local backend adapter for client lifecycle helpers."""

from pathlib import Path

from trw_memory.models.config import MemoryConfig
from trw_memory.storage.interface import StorageBackend


def create_local_backend(
    config: MemoryConfig, namespace: str, db_path_override: Path | str | None = None
) -> StorageBackend:
    """Delegate through the public client module to preserve patch compatibility."""
    from trw_memory.client import _create_local_backend

    return _create_local_backend(config, namespace, db_path_override=db_path_override)
