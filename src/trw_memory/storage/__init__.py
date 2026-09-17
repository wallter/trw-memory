"""Storage backends for trw-memory.

Only the SQLite driver policy is imported eagerly. Everything else resolves
on first attribute access (PEP 562), so ``import trw_memory.storage`` — which
``trw_memory/__init__`` and ``trw_mcp/__init__`` both do to reach ``_dbapi`` —
costs the driver selection and nothing more. Measured 2026-09-17 on macOS: the
eager form cost 0.98 s per interpreter, paid by every PreToolUse hook that
imports ``trw_mcp``; the hook's whole budget is 2.5 s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# MUST stay first. ``_dbapi`` SELECTS the SQLite engine (and undoes a provisional
# swap performed by ``trw_memory/__init__.py`` when the wheel is older than the
# interpreter). Every module below captures ``sqlite3`` in its own namespace at
# import time, starting with ``_wal_checkpoint``, so a later selection could not
# reach them. Deferring them keeps that ordering: they cannot load before this
# line has run.
from trw_memory.storage import _dbapi as _dbapi

if TYPE_CHECKING:
    from trw_memory.storage._wal_checkpoint import CheckpointMode, CheckpointResult
    from trw_memory.storage.interface import StorageBackend
    from trw_memory.storage.persistence import (
        append_jsonl,
        json_serializer,
        lock_for_rmw,
        read_yaml,
        write_yaml,
    )
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.storage.yaml_backend import YAMLBackend

_LAZY: dict[str, tuple[str, str]] = {
    "CheckpointMode": ("trw_memory.storage._wal_checkpoint", "CheckpointMode"),
    "CheckpointResult": ("trw_memory.storage._wal_checkpoint", "CheckpointResult"),
    "StorageBackend": ("trw_memory.storage.interface", "StorageBackend"),
    "append_jsonl": ("trw_memory.storage.persistence", "append_jsonl"),
    "json_serializer": ("trw_memory.storage.persistence", "json_serializer"),
    "lock_for_rmw": ("trw_memory.storage.persistence", "lock_for_rmw"),
    "read_yaml": ("trw_memory.storage.persistence", "read_yaml"),
    "write_yaml": ("trw_memory.storage.persistence", "write_yaml"),
    "SQLiteBackend": ("trw_memory.storage.sqlite_backend", "SQLiteBackend"),
    "YAMLBackend": ("trw_memory.storage.yaml_backend", "YAMLBackend"),
}

__all__ = [
    "CheckpointMode",
    "CheckpointResult",
    "SQLiteBackend",
    "StorageBackend",
    "YAMLBackend",
    "append_jsonl",
    "json_serializer",
    "lock_for_rmw",
    "read_yaml",
    "write_yaml",
]


def __getattr__(name: str) -> object:
    """Resolve a public storage name on first access."""
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
