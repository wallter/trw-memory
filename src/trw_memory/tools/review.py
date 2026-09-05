"""MCP tools: memory_review and memory_quarantine_list — the SEC-001 review surface.

``memory_review`` resolves one quarantined row immutably. ``memory_quarantine_list``
(PRD-CORE-253 FR06) is the discovery half that was missing: without it a
maintainer could only resolve an id some other channel had already given them.
"""

from __future__ import annotations

from typing import Literal

from trw_memory.exceptions import AuthorizationError, ConfigError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.security.runtime import list_quarantined_entries, review_quarantined_entry
from trw_memory.tools._types import McpServer

#: Default page size for the quarantine list. Matches the review queue a human
#: works in one sitting; a maintainer wanting more passes ``limit`` explicitly.
QUARANTINE_LIST_DEFAULT_LIMIT = 100


def memory_review_impl(
    learning_id: str,
    *,
    decision: Literal["approve", "reject"],
    reviewer_id: str,
    namespace: str = "default",
    config: MemoryConfig | None = None,
) -> dict[str, str]:
    cfg = config or MemoryConfig()
    require_namespace_permission(cfg, namespace, Permission.ADMIN, "review")
    from trw_memory.integrations._backend import create_backend_from_config

    with create_backend_from_config(cfg, namespace) as backend:
        return review_quarantined_entry(
            cfg,
            active_backend=backend,
            learning_id=learning_id,
            decision=decision,
            reviewer_id=reviewer_id,
            namespace=namespace,
        )


def register_review_tool(mcp: McpServer) -> None:
    @mcp.tool()
    async def memory_review(
        learning_id: str,
        decision: Literal["approve", "reject"],
        reviewer_id: str,
        namespace: str = "default",
    ) -> dict[str, str]:
        """Resolve a quarantined memory row once, immutably."""

        return memory_review_impl(
            learning_id,
            decision=decision,
            reviewer_id=reviewer_id,
            namespace=namespace,
        )


def memory_quarantine_list_impl(
    namespace: str = "",
    *,
    limit: int = QUARANTINE_LIST_DEFAULT_LIMIT,
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    """List pending quarantined rows the caller may resolve (FR06).

    ``trw_memory.security._runtime_quarantine.list_quarantined_entries`` has
    existed and been exported since SEC-001, with no tool calling it -- so a
    maintainer could resolve a quarantined row only if some other channel had
    already told them its id. This is the missing discovery half. It resolves
    nothing and changes no state; resolution stays with ``memory_review``,
    which is immutable-once.

    Args:
        namespace: Restrict to one namespace. Empty lists every namespace the
            caller holds ADMIN on -- a maintainer of one team's namespace must
            never enumerate another's rejected content.
        limit: Maximum rows to return, newest first.
        config: Memory configuration.

    Returns:
        ``{"entries": [...], "count": int, "namespaces": [...], "status": "ok"}``,
        or ``{"error": str, "status": "forbidden"|"invalid"}`` when the caller
        named a namespace it may not read. An empty quarantine returns an empty
        list, not an error.
    """
    cfg = config or MemoryConfig()
    if namespace:
        # Same shape as ``namespace_admin._curate_impl``: a tool returns a typed
        # refusal, it does not raise across the MCP boundary. Naming a forbidden
        # namespace still has to be an ERROR rather than an empty list, or the
        # result reads as "nothing is quarantined there".
        try:
            require_namespace_permission(cfg, namespace, Permission.ADMIN, "quarantine list")
        except ConfigError as exc:
            return {"error": str(exc), "status": "invalid"}
        except AuthorizationError as exc:
            return {"error": str(exc), "status": "forbidden"}
    candidates = list_quarantined_entries(cfg, namespace=namespace or None, limit=limit)
    permitted = [entry for entry in candidates if _may_admin(cfg, entry.namespace)]
    return {
        "entries": [_quarantine_row(entry) for entry in permitted],
        "count": len(permitted),
        "namespaces": sorted({entry.namespace for entry in permitted}),
        "status": "ok",
    }


def _may_admin(config: MemoryConfig, namespace: str) -> bool:
    """Whether the caller holds ADMIN on *namespace*, as a predicate.

    ``require_namespace_permission`` raises, which is right for a single-target
    verb and wrong for a filter: an unpermitted row must be omitted from the
    list, not turn the whole call into an error.
    """
    try:
        require_namespace_permission(config, namespace, Permission.ADMIN, "quarantine list")
    except (AuthorizationError, ConfigError):
        return False
    return True


def _quarantine_row(entry: MemoryEntry) -> dict[str, str]:
    """Project one quarantined entry to the fields a reviewer decides on."""
    return {
        "id": entry.id,
        "namespace": entry.namespace,
        "content": entry.content,
        "source_identity": entry.source_identity,
        "quarantined_at": entry.metadata.get("quarantined_at", ""),
        "updated_at": entry.updated_at.isoformat(),
    }


def register_quarantine_list_tool(mcp: McpServer) -> None:
    @mcp.tool()
    async def memory_quarantine_list(
        namespace: str = "", limit: int = QUARANTINE_LIST_DEFAULT_LIMIT
    ) -> dict[str, object]:
        """List quarantined rows awaiting review, scoped to permitted namespaces."""

        return memory_quarantine_list_impl(namespace, limit=limit)
