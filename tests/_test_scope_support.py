"""Shared helpers for the PRD-CORE-245 required arguments.

``hybrid_search`` needs a ``NamespaceScope`` and ``fetch_shared_memories`` needs
a backend for the admission gate. Both are required with no default, on purpose;
these helpers give the suite one honest way to satisfy them rather than 50
hand-rolled ones.

The scope goes through the real ``authorize_namespaces`` — never a forged
instance — so a test that ranks entries is exercising the same authorization
path production does.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.namespace_scope import NamespaceScope, authorize_namespaces
from trw_memory.security.rbac import Permission
from trw_memory.storage.sqlite_backend import SQLiteBackend

__all__ = ["DEFAULT_SCOPE", "gate_backend", "scope_for", "scope_of"]


def scope_for(*namespaces: str, config: MemoryConfig | None = None) -> NamespaceScope:
    """Mint a scope over *namespaces* through the production authorizer."""
    return authorize_namespaces(config or MemoryConfig(), namespaces, Permission.READ, "test")


def scope_of(entries: Iterable[MemoryEntry], *, config: MemoryConfig | None = None) -> NamespaceScope:
    """Mint the scope that exactly covers the namespaces *entries* carry."""
    return scope_for(*{entry.namespace for entry in entries}, config=config)


#: Covers the namespace ``MemoryEntry`` defaults to, which is what most fixtures use.
DEFAULT_SCOPE = scope_for("default")


def gate_backend() -> SQLiteBackend:
    """A throwaway on-disk backend for the remote-admission gate to evaluate against.

    The gate needs somewhere to look up an existing row; these tests are about
    the HTTP boundary, not about storage, so a disposable store is the honest
    minimum rather than a mock that would stop the gate from running at all.
    On disk rather than ``:memory:`` because the backend hardens its own file
    permissions at open, which needs a real path.
    """
    directory = Path(tempfile.mkdtemp(prefix="trw-gate-backend-"))
    return SQLiteBackend(directory / "gate.db")
