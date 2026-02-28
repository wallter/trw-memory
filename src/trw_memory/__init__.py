"""trw-memory — Local-first memory layer for AI coding agents."""

from trw_memory._version import __version__
from trw_memory.exceptions import ConfigError, MemoryError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.events import MemoryEvent, MemoryEventType
from trw_memory.models.memory import MemoryEntry, MemoryIndex, MemoryStatus
from trw_memory.namespace import namespace_to_path, validate_namespace

__all__ = [
    "ConfigError",
    "MemoryConfig",
    "MemoryEntry",
    "MemoryError",
    "MemoryEvent",
    "MemoryEventType",
    "MemoryIndex",
    "MemoryStatus",
    "StorageError",
    "__version__",
    "namespace_to_path",
    "validate_namespace",
]
