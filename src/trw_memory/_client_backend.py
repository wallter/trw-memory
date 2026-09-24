"""Late-bound lookups through the public client module for the ``_client_*`` helpers.

The helpers cannot import :mod:`trw_memory.client` at module load (it imports
them), and tests patch names on ``trw_memory.client``; resolving at call time
keeps both working.
"""

from pathlib import Path
from typing import Any

from trw_memory.models.config import MemoryConfig
from trw_memory.storage.interface import StorageBackend


def create_local_backend(
    config: MemoryConfig, namespace: str, db_path_override: Path | str | None = None
) -> StorageBackend:
    """Delegate through the public client module to preserve patch compatibility."""
    from trw_memory.client import _create_local_backend

    return _create_local_backend(config, namespace, db_path_override=db_path_override)


def client_logger() -> Any:
    """Parent-module logger lookup so test patches on ``trw_memory.client.logger`` propagate."""
    from trw_memory import client as _c

    return _c.logger
