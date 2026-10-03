"""MCP tools: sync over the daemon, one namespace at a time (PRD-CORE-298 FR01).

trw-mcp's push pages the dirty rows of its project namespace and marks the
pushed ones synced; its pull finds the local row a remote learning maps to and
writes the merged row through the write gate. Each tool passes the namespace
grant, so a sync cycle never holds a connection and never sees another
project's rows.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from trw_memory.exceptions import SchemaValidationError
from trw_memory.lifecycle.tiers._runtime import embedding_has_consumer
from trw_memory.models._assertion_cap import OVERLONG, overlong
from trw_memory.models._type_coercion import coerce_memory_type_lenient
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync.delta import DeltaTracker, apply_synced_entry, find_synced_entries, find_synced_entry
from trw_memory.tools._embedder import resolve_embedder
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import refused_namespace, serve_namespace

#: Same ceiling as ``tools/listing.py``'s ``LIST_PAGE_MAX`` (trw-mcp's own
#: paging loop already caps a page at this size). A shared daemon serves every
#: tenant from one process; this reads and materializes *limit* rows in one
#: call, so an unbounded value is a single-caller resource exhaustion of the
#: whole daemon. trw-mcp's sync client pages at 500 (``DIRTY_PAGE_SIZE``),
#: well under this cap, so no caller needs to change.
MAX_SYNC_DIRTY_PAGE = 1000
#: Rows one ``memory_sync_apply_many`` call may carry (SYNC-APPLY-BATCH). The call holds the daemon's exclusive write lane for the
#: whole page, so it is bounded well under the 1,000-item argument ceiling; a pull page is 200 rows today and trw-mcp chunks to this.
MAX_SYNC_APPLY_MANY = 200


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


def memory_sync_find_many_impl(
    namespace: str, remote_ids: list[str], ids: list[str], *, backend: StorageBackend, config: MemoryConfig
) -> dict[str, object]:
    """Every row in *namespace* a pulled page maps to: ``remote_id`` in *remote_ids* or id in *ids*, one call."""
    if refused := refused_namespace(namespace, Permission.READ, "sync_find_many", config):
        return refused
    entries = find_synced_entries(backend, namespace, remote_ids, ids)
    return {"status": "ok", "entries": [entry.model_dump(mode="json") for entry in entries]}


def _resolved_embedder(backend: StorageBackend, config: MemoryConfig) -> object | None:
    """The embedder a pulled row is encoded with; ``None`` on a keyword-only host (SHARED-RECALL-LOCAL: the row is stored bare)."""
    resolved = (
        resolve_embedder(config, surface="memory_sync_apply") if embedding_has_consumer(config, backend) else None
    )
    return None if isinstance(resolved, dict) else resolved


def _apply_row(
    namespace: str,
    entry: dict[str, object],
    *,
    backend: StorageBackend,
    config: MemoryConfig,
    if_revision: str | None,
    synced: bool,
    embedder_of: Callable[[], object | None],
) -> dict[str, object]:
    """One pulled row through the write gate: the shared body of ``memory_sync_apply`` and ``memory_sync_apply_many``."""
    kind, type_raw = coerce_memory_type_lenient(entry.get("type"))  # PRD-CORE-334 FR05: a newer client's type
    if type_raw is not None and isinstance(meta := entry.get("metadata") or {}, dict):
        entry = {**entry, "type": kind.value, "metadata": {**meta, "type_raw": type_raw}}
    try:
        row = MemoryEntry.model_validate(entry)
    except ValidationError as exc:
        return {"error": str(exc), "status": "invalid"}
    if row.namespace != namespace:
        return {"error": f"entry namespace {row.namespace!r} is not {namespace!r}", "status": "invalid"}
    if any(map(overlong, row.assertions)):  # a pulled row is a write like any other (rc6 C12)
        return {"error": OVERLONG, "status": "invalid"}
    try:
        status, reason = apply_synced_entry(
            backend,
            config,
            row,
            if_revision=if_revision,
            synced=synced,
            embedder=embedder_of(),  # type: ignore[arg-type]
        )
    except SchemaValidationError as exc:  # the write gate's refusal (an overlong id, non-UTF-8 text) is a verdict
        return {"error": str(exc), "status": "invalid"}
    return {"status": status, "reason": reason}


def memory_sync_apply_impl(
    namespace: str,
    entry: dict[str, object],
    *,
    backend: StorageBackend,
    config: MemoryConfig,
    if_revision: str | None,
    synced: bool = True,
) -> dict[str, object]:
    """Write a merged pulled row into *namespace*: ``stored``, ``quarantined``, ``blocked`` or ``conflict``.

    *if_revision* is the ``revision_of`` the row ``memory_sync_find`` returned (``None``: none);
    a row that moved since is a ``conflict`` and nothing is written (PRD-CORE-308).
    ``synced=False`` leaves the row dirty: a merge holding local content the server lacks.
    """
    if refused := refused_namespace(namespace, Permission.WRITE, "sync_apply", config):
        return refused
    return _apply_row(
        namespace,
        entry,
        backend=backend,
        config=config,
        if_revision=if_revision,
        synced=synced,
        embedder_of=lambda: _resolved_embedder(backend, config),
    )


def memory_sync_apply_many_impl(
    namespace: str, items: list[dict[str, object]], *, backend: StorageBackend, config: MemoryConfig
) -> dict[str, object]:
    """Write a pulled page into *namespace* in one call: ``{"status": "ok", "results": [...]}``, one result per item, in order.

    Each item is ``{"entry", "if_revision", "synced"}`` and gets exactly the verdict ``memory_sync_apply`` would give it
    (``stored``, ``quarantined``, ``blocked``, ``conflict`` or ``invalid``), so one refused or conflicting row never changes the
    others. A row that raises unexpectedly is ``{"status": "error", "error": ...}`` and the rest still apply. The embedder is
    resolved once for the page. More than ``MAX_SYNC_APPLY_MANY`` items is refused whole, nothing written.
    """
    if refused := refused_namespace(namespace, Permission.WRITE, "sync_apply_many", config):
        return refused
    if len(items) > MAX_SYNC_APPLY_MANY:
        return {"error": f"too_many_items: {len(items)} > {MAX_SYNC_APPLY_MANY}", "status": "invalid"}
    cached: list[object | None] = []

    def embedder_once() -> object | None:
        if not cached:
            cached.append(_resolved_embedder(backend, config))
        return cached[0]

    results: list[dict[str, object]] = []
    for item in items:
        revision = item.get("if_revision")
        entry = item.get("entry")
        synced = item.get("synced", True)
        # Typed like the single-row tool: a coerced "false" would be truthy and leave a locally-held row marked clean.
        if (
            not isinstance(entry, dict)
            or not (revision is None or isinstance(revision, str))
            or not isinstance(synced, bool)
        ):
            results.append(
                {
                    "error": "item needs an entry object, a string-or-null if_revision and a boolean synced",
                    "status": "invalid",
                }
            )
            continue
        try:
            results.append(
                _apply_row(
                    namespace,
                    entry,
                    backend=backend,
                    config=config,
                    if_revision=revision,
                    synced=synced,
                    embedder_of=embedder_once,
                )
            )
        except Exception as exc:  # justified: per-row, an unexpected error is that row's verdict and must not lose the rest of the page
            results.append({"error": f"{type(exc).__name__}: {exc}", "status": "error"})
    return {"status": "ok", "results": results}


def register_sync_tools(mcp: McpServer) -> None:
    """Register the sync tools; each authorizes its namespace before opening a backend."""

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

    async def memory_sync_find_many(namespace: str, remote_ids: list[str], ids: list[str]) -> dict[str, object]:
        """Find every row in *namespace* a whole pulled page maps to, in one call."""
        return await serve_namespace(
            namespace,
            Permission.READ,
            "sync_find_many",
            lambda b, c: memory_sync_find_many_impl(namespace, remote_ids, ids, backend=b, config=c),
        )

    async def memory_sync_apply(
        namespace: str, entry: dict[str, object], if_revision: str | None, synced: bool = True
    ) -> dict[str, object]:
        """Write a merged pulled row into *namespace* through the write gate, only over *if_revision*
        (the ``revision_of`` the row read, ``None`` for none; else ``conflict``); ``synced=False`` leaves it dirty."""
        return await serve_namespace(
            namespace,
            Permission.WRITE,
            "sync_apply",
            lambda b, c: memory_sync_apply_impl(
                namespace, entry, backend=b, config=c, if_revision=if_revision, synced=synced
            ),
        )

    async def memory_sync_apply_many(namespace: str, items: list[dict[str, object]]) -> dict[str, object]:
        """Write a whole pulled page into *namespace* in one call: each item ``{entry, if_revision, synced}`` gets the verdict
        ``memory_sync_apply`` would give it, in order; at most ``MAX_SYNC_APPLY_MANY`` items."""
        return await serve_namespace(
            namespace,
            Permission.WRITE,
            "sync_apply_many",
            lambda b, c: memory_sync_apply_many_impl(namespace, items, backend=b, config=c),
        )

    for tool in (
        memory_sync_dirty_page,
        memory_sync_mark_synced,
        memory_sync_find,
        memory_sync_find_many,
        memory_sync_apply,
        memory_sync_apply_many,
    ):
        mcp.tool()(tool)
