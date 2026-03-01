"""FastAPI dependency providers."""

from __future__ import annotations

from pathlib import Path

from trw_memory.models.config import MemoryConfig
from trw_memory.storage.sqlite_backend import SQLiteBackend

_backend: SQLiteBackend | None = None


def get_config() -> MemoryConfig:
    """Return the current MemoryConfig (reads from env vars)."""
    return MemoryConfig()


def get_backend() -> SQLiteBackend:
    """Return a lazily-initialised SQLiteBackend singleton."""
    global _backend  # noqa: PLW0603
    if _backend is None:
        config = get_config()
        db_path = Path(config.storage_path) / config.sqlite_db_name
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _backend = SQLiteBackend(db_path)
    return _backend


def reset_backend() -> None:
    """Close and clear the backend singleton (for testing)."""
    global _backend  # noqa: PLW0603
    if _backend is not None:
        _backend.close()
        _backend = None
