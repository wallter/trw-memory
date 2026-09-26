"""MCP tool: memory_forget — delete memory entries by ID or query.

Thin wrapper that validates namespace, then either deletes a single entry
by ID or performs a bulk search-and-delete.
"""

from __future__ import annotations

import contextlib

import structlog

from trw_memory.daemon._offload import run_serialized
from trw_memory.exceptions import AuthorizationError, StorageError
from trw_memory.lifecycle.tiers._runtime import remove_entry_from_tiers, supports_tier_runtime
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.rbac import Permission
from trw_memory.security.runtime import append_audit_event, delete_quarantined_entries
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import refused_namespace
from trw_memory.tools.search import _SCAN_ROWS

logger = structlog.get_logger(__name__)


def memory_forget_impl(
    memory_id: str | None,
    query: str | None,
    namespace: str,
    *,
    backend: StorageBackend,
    config: MemoryConfig | None = None,
    actor: str | None = None,
) -> dict[str, object]:
    """Core implementation of memory_forget (callable without MCP).

    At least one of *memory_id* or *query* must be provided.

    Args:
        memory_id: Specific entry ID to delete. Takes precedence over query.
        query: Free-text query — all matching entries in the namespace are deleted.
        namespace: Namespace scope for the deletion operation.
        backend: Storage backend instance.

    Returns:
        {"deleted": int, "status": "ok"}
        or {"error": str, "status": "invalid"} on validation failure.
    """
    # Validate that at least one selector is provided
    if memory_id is not None and not memory_id.strip():
        return {
            "error": "memory_id must be non-empty when provided.",
            "status": "invalid",
        }
    if not memory_id and not query and not actor:
        return {
            "error": "At least one of memory_id, query, or actor must be provided.",
            "status": "invalid",
        }

    cfg = config or MemoryConfig()
    if refused := refused_namespace(namespace, Permission.DELETE, "forget", cfg):
        return refused

    if actor:
        # Closure re-audit #5: count + scan + delete must be atomic. A concurrent
        # write landing between a separate count() and list_entries() yields a
        # wrong fetch bound / partial delete (TOCTOU). Cover them with one
        # BEGIN IMMEDIATE snapshot when the backend supports transaction().
        deleted_count = 0
        # At most _SCAN_ROWS of the actor's rows per call, selected in storage (+1 says more remain), so
        # the serialized lane never materializes the namespace (rc9); a repeat call deletes the rest.
        with backend.transaction() if hasattr(backend, "transaction") else contextlib.nullcontext():
            ids = backend.ids_by_source(namespace, actor, _SCAN_ROWS + 1)
            truncated = len(ids) > _SCAN_ROWS
            for entry_id in ids[:_SCAN_ROWS]:
                if backend.delete(entry_id, namespace=namespace):
                    deleted_count += 1
                    if supports_tier_runtime(backend):
                        remove_entry_from_tiers(cfg, namespace, entry_id)
        deleted_count += delete_quarantined_entries(cfg, namespace=namespace, actor=actor)
        append_audit_event(
            cfg,
            "forget",
            actor=actor,
            namespace=namespace,
            data={"entries_deleted": deleted_count, "selector": "actor"},
        )
        truncated_reply = {"truncated": True} if truncated else {}
        return {"deleted": deleted_count, "entries_deleted": deleted_count, "status": "ok", **truncated_reply}

    # --- Delete by ID (with namespace isolation) ---
    if memory_id:
        deleted_count = 0
        entry: MemoryEntry | None = None
        # When an entry exists but lives in a DIFFERENT namespace than the one
        # the caller is scoped to, we must NOT reveal anything about it —
        # including whether it is a system canary. Evaluating the canary refusal
        # before the namespace check (the previous ordering) let an attacker
        # probe cross-namespace canary IDs via the AuthorizationError oracle
        # (trw-memory-5). Resolve the namespace match FIRST.
        in_namespace = False
        try:
            entry = backend.get(memory_id, namespace=namespace)
            # Defence in depth: the read is already namespace-qualified under
            # PRD-CORE-245 FR03, so this second predicate only fires for a
            # backend that ignores it. Keeping it preserves the trw-memory-5
            # property that a cross-namespace id is indistinguishable from a
            # missing one, whatever the backend hands back.
            in_namespace = entry is not None and entry.namespace == namespace
            if in_namespace and entry is not None and entry.metadata.get("system_canary") == "true":
                # Canary refusal only fires for an entry in the caller's own
                # namespace, so canary IDs never leak across namespace boundaries.
                raise AuthorizationError(
                    f"Refusing to delete system canary entry '{memory_id}': deleting canaries is a security violation."
                )
            if in_namespace:
                was_deleted = backend.delete(memory_id, namespace=namespace)
                deleted_count = 1 if was_deleted else 0
                if was_deleted and supports_tier_runtime(backend):
                    remove_entry_from_tiers(cfg, namespace, memory_id)
            else:
                # Entry is missing OR lives in a different namespace. In BOTH
                # cases we only consult the caller's own quarantine partition
                # (delete_quarantined_entries is namespace-scoped) — never the
                # primary store of another namespace. Crucially, "exists in
                # another namespace" and "does not exist anywhere" return the
                # SAME shape below, so cross-namespace existence is not confirmed
                # via a distinguishable response (trw-memory-5 namespace
                # isolation / canary oracle).
                deleted_count = delete_quarantined_entries(cfg, namespace=namespace, memory_id=memory_id)
        except StorageError as exc:
            logger.warning("memory_forget_delete_error", memory_id=memory_id, error=str(exc))

        # Distinguish "actually removed something" (ok) from "nothing matched in
        # this namespace" (not_found) so callers stop treating a silent 0-delete
        # as success — without leaking whether the id exists elsewhere.
        status = "ok" if deleted_count > 0 else "not_found"
        actor_identity = entry.source_identity if (entry is not None and in_namespace) else ""
        logger.info(
            "memory_forget",
            memory_id=memory_id,
            deleted=deleted_count,
            namespace=namespace,
            outcome=status,
        )
        append_audit_event(
            cfg,
            "forget",
            entry_id=memory_id,
            actor=actor_identity,
            namespace=namespace,
            data={"entries_deleted": deleted_count, "selector": "memory_id", "outcome": status},
        )
        return {"deleted": deleted_count, "status": status}

    # --- Bulk delete via search query ---
    assert query is not None  # noqa: S101 — mypy narrowing guard; query is not None here: the memory_id branch above returns before reaching this line, and the early-return guard ensures at least one of memory_id/query is set
    try:
        matches = backend.search(
            query,
            top_k=10_000,
            namespace=namespace,
        )
    except StorageError as exc:
        logger.warning("memory_forget_search_error", query=query[:80], error=str(exc))
        return {"deleted": 0, "status": "ok"}

    deleted_count = 0
    for entry in matches:
        try:
            if backend.delete(entry.id, namespace=entry.namespace):
                deleted_count += 1
                if supports_tier_runtime(backend):
                    remove_entry_from_tiers(cfg, namespace, entry.id)
        except StorageError as exc:  # per-item error handling: log delete failure for this entry, continue bulk delete
            logger.warning("memory_forget_delete_error", memory_id=entry.id, error=str(exc))

    logger.info(
        "memory_forget_bulk",
        query=query[:80],
        matches=len(matches),
        deleted=deleted_count,
        namespace=namespace,
    )
    append_audit_event(
        cfg,
        "forget",
        actor="",
        namespace=namespace,
        data={"entries_deleted": deleted_count, "selector": "query", "query": query[:80]},
    )
    return {"deleted": deleted_count, "status": "ok"}


def register_forget_tool(mcp: McpServer) -> None:
    """Register memory_forget with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from trw_memory.integrations._backend import create_backend_from_config

    async def memory_forget(
        memory_id: str | None = None,
        query: str | None = None,
        namespace: str = "project:default",
        actor: str | None = None,
    ) -> dict[str, object]:
        """Delete memory entries by ID or bulk search query.

        Provide *memory_id* to delete a specific entry, or *query* to
        delete all entries matching the search query in the namespace.
        At least one must be provided.

        Args:
            memory_id: Specific memory entry ID to delete.
            query: Free-text query — all matching entries are deleted.
            namespace: Namespace scope for the operation.

        Returns:
            {"deleted": int, "status": "ok"}
        """

        def forget() -> dict[str, object]:
            with create_backend_from_config(cfg := MemoryConfig(), namespace) as backend:
                return memory_forget_impl(memory_id, query, namespace, backend=backend, config=cfg, actor=actor)

        return await run_serialized(forget)

    mcp.tool()(memory_forget)
