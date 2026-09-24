"""MCP tool: memory_update — correct or retire one stored learning by id (PRD-CORE-294 FR03).

A thin adapter over :mod:`trw_memory.lifecycle.correction`, the same function
trw-mcp's ``trw_learn(learning_id=...)`` update mode calls, so a patch means
the same thing on both servers.
"""

from __future__ import annotations

from trw_memory.exceptions import ConfigError
from trw_memory.lifecycle.correction import LearningPatch, Store, apply_correction, not_found, parse_patch
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.security.runtime import append_audit_event
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer


def memory_update_impl(
    entry_id: str,
    patch: LearningPatch,
    namespace: str,
    *,
    backend: StorageBackend,
    config: MemoryConfig | None = None,
    actor: str | None = None,
) -> dict[str, str]:
    """Apply ``patch`` to entry ``entry_id``; returns updated / no_changes / invalid / not_found.

    The patch is typed (operator decision 2026-09-03) so the audit event names what changed;
    the MCP wrapper parses a caller's raw dict with ``parse_patch`` first.
    """
    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    cfg = config or MemoryConfig()
    require_namespace_permission(cfg, namespace, Permission.WRITE, "update")
    entry = backend.get(entry_id, namespace=namespace)
    if entry is None:
        return not_found(entry_id)
    store = Store(backend, cfg)
    prior = None
    if patch.supersedes is not None:
        prior = (store, backend.get(patch.supersedes, namespace=namespace))
    result = apply_correction(store, entry, patch, prior=prior)
    if result["status"] != "updated":
        return result
    append_audit_event(
        cfg,
        "update",
        actor=actor or "",
        namespace=namespace,
        data={"entry_id": entry_id, "changes": result["changes"]},
    )
    return result


def register_update_tool(mcp: McpServer) -> None:
    """Register memory_update with a FastMCP server instance."""
    from trw_memory.tools.entry import in_namespace

    async def memory_update(
        entry_id: str,
        patch: dict[str, object],
        namespace: str = "project:default",
    ) -> dict[str, str] | dict[str, object]:
        """Use when a stored learning is wrong, stale or superseded: correct named fields or retire it.

        patch names only the fields to change (status, summary, detail, impact, type,
        confidence, tags, tags_add, assertions, supersedes, ...); unknown keys are rejected.
        Retire with {"status": "obsolete"}; default recall then omits the entry.

        Output: status updated | no_changes | invalid | not_found, with changes or error.
        """
        parsed = parse_patch(patch)
        if isinstance(parsed, dict):
            return parsed
        # in_namespace authorizes before any backend opens.
        return in_namespace(
            namespace,
            Permission.WRITE,
            "update",
            lambda b, c: memory_update_impl(entry_id, parsed, namespace, backend=b, config=c),
        )

    mcp.tool()(memory_update)
