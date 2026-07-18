"""trw-memory — Local-first memory layer for AI coding agents."""

# MUST be first and INLINE: swap stdlib sqlite3 with pysqlite3 before any
# submodule has the chance to ``import sqlite3``. Routing this through a
# submodule (``trw_memory.storage._dbapi``) is too late, because importing
# the ``trw_memory.storage`` package triggers ``storage/__init__.py`` which
# eagerly pulls in ``sqlite_backend.py`` -> ``_init_helpers.py`` etc., each
# of which captures stdlib ``sqlite3`` in its own namespace before the
# shim has a chance to run. The inline pattern here uses only top-level
# ``sys`` + ``pysqlite3`` and does not import anything from this package,
# so no submodule loads first.
try:
    import sys as _sys

    import pysqlite3 as _pysqlite3  # type: ignore[import-untyped]

    _sys.modules["sqlite3"] = _pysqlite3
    _sys.modules["sqlite3.dbapi2"] = _pysqlite3.dbapi2
    _pysqlite3._trw_pysqlite3_active = True
except ImportError:
    # Fall through with stdlib sqlite3 — older bundled SQLite carries the
    # WAL-reset bug, but the engine still works.
    pass

# Re-import the observability shim so callers can ask ``_dbapi.backend()``
# without having to handle the optional dep themselves. By now ``sqlite3``
# is already swapped; the import is purely for the helper API.
from trw_memory.storage import _dbapi as _dbapi  # noqa: I001

from trw_memory._logging import configure_library_logging as _configure_library_logging

# Library best practice: stay silent until the consuming application or the
# trw-memory CLI explicitly configures logging.
_configure_library_logging()

from trw_memory._version import __version__
from trw_memory.client import MemoryClient
from trw_memory.exceptions import (
    AuthorizationError,
    ConfigError,
    DimensionMismatchError,
    EncryptionUnavailableError,
    KeyRotationError,
    LocalOnlyViolationError,
    MasterKeyNotFoundError,
    MemoryConnectionError,
    MemoryError,
    MemoryNotFoundError,
    PIIBlockError,
    PoisoningError,
    RateLimitError,
    SchemaValidationError,
    StorageError,
    ToolAlreadyRegisteredError,
)
from trw_memory.hype import NoOpQuestionGenerator, QuestionGenerator
from trw_memory.models.config import MemoryConfig
from trw_memory.models.events import MemoryEvent, MemoryEventType
from trw_memory.models.memory import MemoryEntry, MemoryIndex, MemoryStatus
from trw_memory.namespaces.path_mapping import namespace_to_path
from trw_memory.namespaces.validation import validate_namespace

__all__ = [
    "AuthorizationError",
    "ConfigError",
    "DimensionMismatchError",
    "EncryptionUnavailableError",
    "KeyRotationError",
    "LocalOnlyViolationError",
    "MasterKeyNotFoundError",
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
    "NoOpQuestionGenerator",
    "PIIBlockError",
    "PoisoningError",
    "QuestionGenerator",
    "RateLimitError",
    "SchemaValidationError",
    "StorageError",
    "ToolAlreadyRegisteredError",
    "__version__",
    "namespace_to_path",
    "validate_namespace",
]
