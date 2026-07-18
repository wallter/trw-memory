"""Backend-derived configuration for background graph updates."""

from pathlib import Path

from trw_memory.models.config import MemoryConfig
from trw_memory.storage.interface import StorageBackend


def derive_graph_config(backend: StorageBackend, config: MemoryConfig | None) -> MemoryConfig | None:
    """Return explicit configuration or reconstruct it from a built-in backend."""
    if config is not None:
        return config

    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.storage.yaml_backend import YAMLBackend

    if isinstance(backend, SQLiteBackend):
        db_path = Path(str(backend._db_path))
        return MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(db_path.parent.parent),
            sqlite_db_name=db_path.name,
            embedding_dim=backend._dim,
        )
    if isinstance(backend, YAMLBackend):
        entries_dir = Path(str(backend._dir))
        return MemoryConfig(storage_backend="yaml", storage_path=str(entries_dir.parent.parent))
    return None
