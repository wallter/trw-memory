"""MCP tools: sync over the daemon, one namespace at a time (PRD-CORE-298 FR01).

trw-mcp's push pages the dirty rows of its project namespace and marks the
pushed ones synced; its pull finds the local row a remote learning maps to and
writes the merged row through the write gate. Each tool passes the namespace
grant, so a sync cycle never holds a connection and never sees another
project's rows.
"""

from __future__ import annotations

from pydantic import ValidationError

from trw_memory.models._assertion_cap import OVERLONG, overlong
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync.delta import DeltaTracker, apply_synced_entry, find_synced_entry
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import refused_namespace, serve_namespace

#: Same ceiling as ``tools/listing.py``'s ``LIST_PAGE_MAX`` (trw-mcp's own
#: paging loop already caps a page at this size). A shared daemon serves every
#: tenant from one process; this reads and materializes *limit* rows in one
#: call, so an unbounded value is a single-caller resource exhaustion of the
#: whole daemon. trw-mcp's sync client pages at 500 (``DIRTY_PAGE_SIZE``),
#: well under this cap, so no caller needs to change.
MAX_SYNC_DIRTY_PAGE = 1000


def memory_sync_dirty_page_impl(
    namespace: str, limit: int, *, backend: StorageBackend, config: MemoryConfig
) -> dict[str, object]:
    """The oldest *limit* rows of *namespace* that still need a push."""
    if limit < 1 or limit > MAX_SYNC_DIRTY_PAGE:
        return {"error": f"limit must be in [1, {MAX_SYNC_DIRTY_PAGE}]", "status": "invalid"}
    if refused := refused_namespace(namespace, Permission.READ, "sync_dirty_page", config):
        return refused
    entries = DeltaTracker.get_dirty_entries(backend, namespace=namespace, limit=limit)
    return {"entries": [entry.model_dump(mode="json") for entry in entries]}


def memory_sync_mark_synced_impl(
    namespace: str, acks: dict[str, int], *, backend: StorageBackend, config: MemoryConfig
) -> dict[str, object]:
    """Mark pushed rows synced: *acks* maps each id to the ``sync_seq`` it was paged at.

    A row edited after it was paged has a newer ``sync_seq`` and stays dirty, so
    the edit is pushed on the next cycle instead of being marked clean unpushed.
    Ids in another namespace are not touched.
    """
    if refused := refused_namespace(namespace, Permission.WRITE, "sync_mark_synced", config):
        return refused
    return {"marked": DeltaTracker.mark_synced(list(acks), backend, namespace=namespace, expected_seq=acks)}


def memory_sync_find_impl(
    namespace: str, remote_id: str, ids: list[str], *, backend: StorageBackend, config: MemoryConfig
) -> dict[str, object]:
    """The row in *namespace* whose ``remote_id`` is *remote_id* or whose id is one of *ids*."""
    if refused := refused_namespace(namespace, Permission.READ, "sync_find", config):
        return refused
    entry = find_synced_entry(backend, namespace, remote_id, ids)
    return {"status": "ok", "entry": entry.model_dump(mode="json")} if entry else {"status": "not_found"}


def memory_sync_apply_impl(
    namespace: str,
    entry: dict[str, object],
    *,
    backend: StorageBackend,
    config: MemoryConfig,
    synced: bool = True,
) -> dict[str, object]:
    """Write a merged pulled row into *namespace*: ``stored``, ``quarantined`` or ``blocked``.

    ``synced=False`` leaves the row dirty: a merge holding local content the server lacks.
    """
    if refused := refused_namespace(namespace, Permission.WRITE, "sync_apply", config):
        return refused
    try:
        row = MemoryEntry.model_validate(entry)
    except ValidationError as exc:
        return {"error": str(exc), "status": "invalid"}
    if row.namespace != namespace:
        return {"error": f"entry namespace {row.namespace!r} is not {namespace!r}", "status": "invalid"}
    if any(map(overlong, row.assertions)):  # a pulled row is a write like any other (rc6 C12)
        return {"error": OVERLONG, "status": "invalid"}
    status, reason = apply_synced_entry(backend, config, row, synced=synced)
    return {"status": status, "reason": reason}


def register_sync_tools(mcp: McpServer) -> None:
    """Register the four sync tools; each authorizes its namespace before opening a backend."""

    async def memory_sync_dirty_page(namespace: str, limit: int = 500) -> dict[str, object]:
        """Return the oldest *limit* rows of *namespace* that still need a push."""
        return await serve_namespace(
            namespace,
            Permission.READ,
            "sync_dirty_page",
            lambda b, c: memory_sync_dirty_page_impl(namespace, limit, backend=b, config=c),
        )

    async def memory_sync_mark_synced(namespace: str, acks: dict[str, int]) -> dict[str, object]:
        """Mark pushed rows of *namespace* synced, each only while it still has the paged ``sync_seq``."""
        return await serve_namespace(
            namespace,
            Permission.WRITE,
            "sync_mark_synced",
            lambda b, c: memory_sync_mark_synced_impl(namespace, acks, backend=b, config=c),
        )

    async def memory_sync_find(namespace: str, remote_id: str, ids: list[str]) -> dict[str, object]:
        """Find the row in *namespace* a pulled learning maps to."""
        return await serve_namespace(
            namespace,
            Permission.READ,
            "sync_find",
            lambda b, c: memory_sync_find_impl(namespace, remote_id, ids, backend=b, config=c),
        )

    async def memory_sync_apply(namespace: str, entry: dict[str, object], synced: bool = True) -> dict[str, object]:
        """Write a merged pulled row into *namespace* through the write gate; ``synced=False`` leaves it dirty."""
        return await serve_namespace(
            namespace,
            Permission.WRITE,
            "sync_apply",
            lambda b, c: memory_sync_apply_impl(namespace, entry, backend=b, config=c, synced=synced),
        )

    for tool in (memory_sync_dirty_page, memory_sync_mark_synced, memory_sync_find, memory_sync_apply):
        mcp.tool()(tool)
