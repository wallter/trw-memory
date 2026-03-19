"""trw-memory — Local-first memory layer for AI coding agents."""

import logging

# Library best practice: prevent "No handler found" warnings.
# The consuming application (trw-mcp, user projects) configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

from trw_memory._version import __version__
from trw_memory.client import MemoryClient
from trw_memory.exceptions import (
    ConfigError,
    MemoryConnectionError,
    MemoryError,
    MemoryNotFoundError,
    StorageError,
    ToolAlreadyRegisteredError,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.events import MemoryEvent, MemoryEventType
from trw_memory.models.memory import MemoryEntry, MemoryIndex, MemoryStatus
from trw_memory.namespace import namespace_to_path, validate_namespace

__all__ = [
    "ConfigError",
    "MemoryClient",
    "MemoryConfig",
    "MemoryConnectionError",
    "MemoryEntry",
    "MemoryError",
    "MemoryEvent",
    "MemoryEventType",
    "MemoryIndex",
    "MemoryNotFoundError",
    "MemoryStatus",
    "StorageError",
    "ToolAlreadyRegisteredError",
    "__version__",
    "namespace_to_path",
    "validate_namespace",
]
