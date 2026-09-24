"""MCP tool: memory_list_page -- one keyset page of a namespace (PRD-CORE-298 FR01/FR02).

Every whole-namespace listing that reaches the daemon goes through this tool:
``trw-memory export`` and the trw-mcp store's ``list_entries`` (active
learnings, status listings, tag-filtered reconcile rows). Pages walk the
backend's ``(updated_at DESC, id DESC)`` keyset, so no row is skipped or
repeated when the namespace is larger than a page, and each call reads only the
one granted namespace.
"""

from __future__ import annotations

from dataclasses import asdict

from trw_memory.exceptions import ConfigError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.storage.interface import EntryCursor, StorageBackend
from trw_memory.tools._types import McpServer

#: The largest page one call returns.
LIST_PAGE_MAX = 1000


def memory_list_page_impl(
    namespace: str,
    limit: int,
    after: dict[str, str] | None,
    *,
    backend: StorageBackend,
    config: MemoryConfig | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, object]:
    """Return ``{"status": "ok", "entries": [...], "next": cursor | None}``; ``next`` resumes after the page.

    Entries are ``MemoryEntry`` JSON. *status* and *tags* (every tag must be
    present) filter inside the query, so the page limit applies after them.
    """
    try:
        validate_namespace(namespace)
        wanted = MemoryStatus(status) if status is not None else None
    except (ConfigError, ValueError) as exc:
        return {"error": str(exc), "status": "invalid"}
    require_namespace_permission(config or MemoryConfig(), namespace, Permission.READ, "list")
    if not 1 <= limit <= LIST_PAGE_MAX:
        return {"error": f"limit must be in [1, {LIST_PAGE_MAX}]", "status": "invalid"}
    cursor = EntryCursor(updated_at=after["updated_at"], entry_id=after["entry_id"]) if after else None
    entries = backend.list_entries(namespace=namespace, status=wanted, tags=tags or None, limit=limit, after=cursor)
    resume = asdict(EntryCursor.from_entry(entries[-1])) if len(entries) == limit else None
    return {"status": "ok", "entries": [entry.model_dump(mode="json") for entry in entries], "next": resume}


def register_list_page_tool(mcp: McpServer) -> None:
    """Register memory_list_page with a FastMCP server instance."""
    from trw_memory.tools.entry import in_namespace

    async def memory_list_page(
        namespace: str,
        limit: int = 500,
        after: dict[str, str] | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, object]:
        """One page of *namespace*'s rows, newest first, optionally filtered by status and tags."""
        return in_namespace(
            namespace,
            Permission.READ,
            "list",
            lambda b, c: memory_list_page_impl(namespace, limit, after, backend=b, config=c, status=status, tags=tags),
        )

    mcp.tool()(memory_list_page)
