"""trw-memory — Local-first memory layer for AI coding agents.

Importing this package does three things eagerly, in this order: the
provisional ``pysqlite3`` swap, the SQLite driver selection in
``trw_memory.storage._dbapi``, and the library logging default. Every public
name in ``__all__`` resolves on first access (PEP 562). The eager form cost
1.25 s per interpreter on macOS (measured 2026-09-17), which every PreToolUse
hook paid before doing any work; the lazy form costs the driver selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# MUST be first and INLINE: swap stdlib sqlite3 with pysqlite3 before any
# submodule has the chance to ``import sqlite3``.
#
# This swap is PROVISIONAL. It has no version policy and cannot have one: it must
# run before any submodule loads, and the policy lives in a submodule. The
# ``_dbapi`` import below reaches ``trw_memory.storage``, whose ``__init__``
# imports ``storage/_dbapi.py`` FIRST -- which ranks the candidate against the
# interpreter's own SQLite and EVICTS this swap when the wheel is older (it is,
# on any current CPython: every published wheel bundles 3.51.1). That eviction is
# only effective while nothing has captured ``sqlite3`` yet, which is exactly the
# window this block opens and ``storage/__init__`` closes. Do not reorder either.
# The inline pattern here uses only top-level ``sys`` + ``pysqlite3`` and does
# not import anything from this package, so no submodule loads first.
try:
    import sys as _sys

    # No inline ignore: the pysqlite3 stub gap is handled by the mypy override
    # list in pyproject.toml, so this line type-checks on every platform rather
    # than only on one where the wheel happens to be installed (PRD-INFRA-185).
    import pysqlite3 as _pysqlite3

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
from trw_memory._version import __version__

# Library best practice: stay silent until the consuming application or the
# trw-memory CLI explicitly configures logging.
_configure_library_logging()

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient
    from trw_memory.exceptions import (
        AuthorizationError,
        ConfigError,
        DimensionMismatchError,
        EncryptionUnavailableError,
        LocalOnlyViolationError,
        MasterKeyNotFoundError,
        MemoryConnectionError,
        MemoryError,
        MemoryNotFoundError,
        MemoryQuarantinedError,
        PIIBlockError,
        PoisoningError,
        RateLimitError,
        SchemaValidationError,
        StorageError,
        ToolAlreadyRegisteredError,
    )
    from trw_memory.models.config import MemoryConfig
    from trw_memory.models.events import MemoryEvent, MemoryEventType
    from trw_memory.models.memory import MemoryEntry, MemoryIndex, MemoryStatus
    from trw_memory.namespaces.path_mapping import namespace_to_path
    from trw_memory.namespaces.validation import validate_namespace

_EXCEPTIONS = (
    "AuthorizationError",
    "ConfigError",
    "DimensionMismatchError",
    "EncryptionUnavailableError",
    "LocalOnlyViolationError",
    "MasterKeyNotFoundError",
    "MemoryConnectionError",
    "MemoryError",
    "MemoryNotFoundError",
    "MemoryQuarantinedError",
    "PIIBlockError",
    "PoisoningError",
    "RateLimitError",
    "SchemaValidationError",
    "StorageError",
    "ToolAlreadyRegisteredError",
)

_LAZY: dict[str, tuple[str, str]] = {
    **{name: ("trw_memory.exceptions", name) for name in _EXCEPTIONS},
    "MemoryClient": ("trw_memory.client", "MemoryClient"),
    "MemoryConfig": ("trw_memory.models.config", "MemoryConfig"),
    "MemoryEvent": ("trw_memory.models.events", "MemoryEvent"),
    "MemoryEventType": ("trw_memory.models.events", "MemoryEventType"),
    "MemoryEntry": ("trw_memory.models.memory", "MemoryEntry"),
    "MemoryIndex": ("trw_memory.models.memory", "MemoryIndex"),
    "MemoryStatus": ("trw_memory.models.memory", "MemoryStatus"),
    "namespace_to_path": ("trw_memory.namespaces.path_mapping", "namespace_to_path"),
    "validate_namespace": ("trw_memory.namespaces.validation", "validate_namespace"),
}

__all__ = [
    *_EXCEPTIONS,
    "MemoryClient",
    "MemoryConfig",
    "MemoryEntry",
    "MemoryEvent",
    "MemoryEventType",
    "MemoryIndex",
    "MemoryStatus",
    "__version__",
    "namespace_to_path",
    "validate_namespace",
]


def __getattr__(name: str) -> object:
    """Resolve a public package name on first access."""
    from importlib import import_module

    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        # ``import trw_memory; trw_memory.exceptions`` worked while the init was
        # eager (the submodule had been imported as a side effect); keep it working.
        if name.startswith("__"):
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
        try:
            return import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
