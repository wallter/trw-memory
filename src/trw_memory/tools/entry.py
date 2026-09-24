"""MCP tool: memory_get -- one entry by ``(namespace, id)`` (PRD-CORE-298 FR01).

trw-mcp's store reads an entry by id over the daemon with this verb; a correction
goes through ``memory_update`` (``tools/update.py``, PRD-CORE-294 FR03). It goes
through the namespace grant like every other tool, and a cross-namespace id
answers exactly like a missing one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from trw_memory.exceptions import ConfigError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.rbac import Permission, require_namespace_permission, transport_root
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer


def refused_namespace(
    namespace: str, permission: Permission, operation: str, config: MemoryConfig
) -> dict[str, object] | None:
    """An ``invalid`` result for a malformed *namespace*; raises ``AuthorizationError`` outside the grant."""
    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    require_namespace_permission(config, namespace, permission, operation)
    return None


def checkout_path(path: str | None, operation: str, *, within: bool) -> str | None | dict[str, object]:
    """*path* as this request may use it, or a ``refused`` result.

    Over the transport a file-reading tool is bounded by the checkout its token
    was minted for: *within* admits a path inside it, otherwise only the checkout
    itself (``None`` means the checkout). A grant recording no checkout reaches
    no file. Off the transport -- the in-process SDK -- *path* is used as given.
    """
    on_transport, root = transport_root()
    if not on_transport:
        return path

    def refused(reason: str) -> dict[str, object]:
        return {"error": f"{operation} refused: {reason}", "status": "refused"}

    if root is None:
        return refused("this token's grant records no checkout; run `trw-mcp memory token` to re-mint it")
    granted = Path(root).resolve()
    if path is None:
        return None if within else str(granted)
    wanted = Path(path).resolve()
    if wanted.is_relative_to(granted) if within else wanted == granted:
        return str(wanted)
    return refused(f"{path} is not {'inside ' if within else ''}the granted checkout {root}")


_Result = TypeVar("_Result")


def in_namespace(
    namespace: str,
    permission: Permission,
    operation: str,
    run: Callable[[StorageBackend, MemoryConfig], _Result],
) -> _Result | dict[str, object]:
    """Validate and authorize *namespace*, then open its backend and *run*.

    A daemon tool serves through this, so a malformed or ungranted namespace is
    refused before any backend is created for it (PRD-CORE-298 FR02).
    """
    from trw_memory.integrations._backend import create_backend_from_config

    config = MemoryConfig()
    if refused := refused_namespace(namespace, permission, operation, config):
        return refused
    with create_backend_from_config(config, namespace) as backend:
        return run(backend, config)


def _scoped_entry(
    memory_id: str,
    namespace: str,
    permission: Permission,
    operation: str,
    backend: StorageBackend,
    config: MemoryConfig,
) -> MemoryEntry | dict[str, object]:
    if refused := refused_namespace(namespace, permission, operation, config):
        return refused
    entry = backend.get(memory_id, namespace=namespace)
    if entry is None or entry.namespace != namespace:
        return {"status": "not_found"}
    return entry


def memory_get_impl(
    memory_id: str, namespace: str, *, backend: StorageBackend, config: MemoryConfig | None = None
) -> dict[str, object]:
    """Return ``{"status": "ok", "entry": <json>}`` or ``{"status": "not_found"}``."""
    entry = _scoped_entry(memory_id, namespace, Permission.READ, "get", backend, config or MemoryConfig())
    if isinstance(entry, dict):
        return entry
    return {"status": "ok", "entry": entry.model_dump(mode="json")}


def memory_find_duplicate_impl(
    namespace: str, content: str, detail: str, *, backend: StorageBackend, config: MemoryConfig | None = None
) -> dict[str, object]:
    """``{"status": "ok", "entry_id": <id or None>}``: an ACTIVE row of *namespace* with exactly this content."""
    if refused := refused_namespace(namespace, Permission.READ, "find_duplicate", config or MemoryConfig()):
        return refused
    return {"status": "ok", "entry_id": backend.find_active_by_content(content, detail, namespace=namespace)}


def register_entry_tools(mcp: McpServer) -> None:
    """Register memory_get and memory_find_duplicate with a FastMCP server instance."""

    async def memory_get(memory_id: str, namespace: str) -> dict[str, object]:
        """Return one memory entry by id within *namespace*."""
        return in_namespace(
            namespace, Permission.READ, "get", lambda b, c: memory_get_impl(memory_id, namespace, backend=b, config=c)
        )

    async def memory_find_duplicate(namespace: str, content: str, detail: str) -> dict[str, object]:
        """Id of an ACTIVE entry in *namespace* whose content and detail match exactly, or null."""
        return in_namespace(
            namespace,
            Permission.READ,
            "find_duplicate",
            lambda b, c: memory_find_duplicate_impl(namespace, content, detail, backend=b, config=c),
        )

    mcp.tool()(memory_get)
    mcp.tool()(memory_find_duplicate)
