"""MCP tool: memory_search — list/filter memory entries with pagination.

Thin wrapper around backend.list_entries() with namespace validation,
optional tag/status filtering, and offset-based pagination.
"""

from __future__ import annotations

import structlog

from trw_memory.daemon._offload import run_serialized
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.security.rbac import Permission
from trw_memory.security.runtime import append_audit_event, list_quarantined_entries
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import refused_namespace

logger = structlog.get_logger(__name__)

# Valid status values for user-supplied strings
_VALID_STATUSES = {s.value for s in MemoryStatus}
_SPECIAL_STATUSES = {"quarantined"}
_MAX_SEARCH_LIMIT = 500
#: The most rows any search reads. The filters run over what was read, on the serialized lane, so an
#: actor filter reads this many of the newest rows, never the whole namespace (rc9).
_SCAN_ROWS = 10_000


def _quarantine_rows(config: MemoryConfig, namespace: str) -> int:
    """Rows of any status in *namespace*'s quarantine store: its read takes the newest 10,000 of them."""
    from trw_memory.security._runtime_quarantine import open_quarantine_backend

    with open_quarantine_backend(config) as store:
        return store.count(namespace=namespace)


def memory_search_impl(
    namespace: str,
    *,
    backend: StorageBackend,
    config: MemoryConfig | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
    sort_by: str = "updated_at",
    offset: int = 0,
    limit: int = 50,
    actor: str | None = None,
) -> dict[str, object]:
    """Core implementation of memory_search (callable without MCP).

    Args:
        namespace: Namespace to search within.
        backend: Storage backend instance.
        status: If provided, filter to entries with this lifecycle status.
        tags: If provided, only entries containing ALL of these tags are returned.
        sort_by: Field to sort results by (informational; backends sort by updated_at).
        offset: Number of entries to skip (pagination).
        limit: Maximum entries to return per page.

    Returns:
        {"entries": list[dict], "total": int, "offset": int, "limit": int}; with
        *actor*, "truncated": true when the newest 10,000 rows were searched and older ones were not
        or {"error": str, "status": "invalid"} on validation failure.
    """
    cfg = config or MemoryConfig()
    if refused := refused_namespace(namespace, Permission.READ, "search", cfg):
        return refused
    if limit < 1 or limit > _MAX_SEARCH_LIMIT:
        return {"error": f"limit must be in [1, {_MAX_SEARCH_LIMIT}]", "status": "invalid"}
    if offset < 0:
        return {"error": "offset must be >= 0", "status": "invalid"}

    # Validate status string
    memory_status: MemoryStatus | None = None
    if status is not None:
        if status not in _VALID_STATUSES | _SPECIAL_STATUSES:
            return {
                "error": f"Invalid status {status!r}. Must be one of: {sorted(_VALID_STATUSES | _SPECIAL_STATUSES)}",
                "status": "invalid",
            }
        if status != "quarantined":
            memory_status = MemoryStatus(status)

    # Fetch all matching entries (apply status + namespace, handle pagination here)
    # With an actor, one row past the window says whether older rows went unread.
    fetch_limit = _SCAN_ROWS + 1 if actor is not None else min(max(limit + offset, 500), _SCAN_ROWS)
    if status == "quarantined":
        entries = list_quarantined_entries(
            cfg,
            namespace=namespace,
            limit=fetch_limit,
        )
    else:
        entries = backend.list_entries(
            status=memory_status,
            namespace=namespace,
            limit=fetch_limit,
        )

    unread = len(entries) > _SCAN_ROWS or (status == "quarantined" and _quarantine_rows(cfg, namespace) > _SCAN_ROWS)
    truncated, entries = actor is not None and unread, entries[:_SCAN_ROWS]
    # Apply tag filter in Python (not all backends support tag filtering in list_entries)
    if tags:
        tag_set = set(tags)
        entries = [e for e in entries if tag_set.issubset(set(e.tags))]
    if actor is not None:
        entries = [e for e in entries if e.source_identity == actor]

    total = len(entries)

    # Pagination slice
    page = entries[offset : offset + limit]

    result_dicts = [e.model_dump(mode="json") for e in page]

    logger.debug(
        "memory_search",
        namespace=namespace,
        status=status,
        total=total,
        offset=offset,
        limit=limit,
        returned=len(result_dicts),
    )
    append_audit_event(
        cfg,
        "access",
        actor=actor or "",
        namespace=namespace,
        data={
            "entries_returned": len(result_dicts),
            "status": status or "",
            "actor_filter": actor or "",
            "tag_filter": tags or [],
        },
    )

    return {
        "entries": result_dicts,
        "total": total,
        "offset": offset,
        "limit": limit,
        **({"truncated": True} if truncated else {}),
    }


def register_search_tool(mcp: McpServer) -> None:
    """Register memory_search with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from trw_memory.integrations._backend import create_backend_from_config

    async def memory_search(
        namespace: str = "project:default",
        status: str | None = None,
        tags: list[str] | None = None,
        sort_by: str = "updated_at",
        offset: int = 0,
        limit: int = 50,
        actor: str | None = None,
    ) -> dict[str, object]:
        """List and filter memory entries with pagination.

        Args:
            namespace: Namespace to search within (e.g., 'project:default').
            status: Filter by lifecycle status: 'active', 'resolved', 'obsolete', 'archived'.
            tags: Filter to entries containing ALL of these tags.
            sort_by: Sort field (default: 'updated_at').
            offset: Pagination offset.
            limit: Maximum entries per page (default 50).

        Returns:
            {"entries": [...], "total": int, "offset": int, "limit": int}
        """

        def search() -> dict[str, object]:
            with create_backend_from_config(cfg := MemoryConfig(), namespace) as backend:
                return memory_search_impl(
                    namespace,
                    backend=backend,
                    config=cfg,
                    status=status,
                    tags=tags,
                    sort_by=sort_by,
                    offset=offset,
                    limit=limit,
                    actor=actor,
                )

        return await run_serialized(search)

    mcp.tool()(memory_search)
