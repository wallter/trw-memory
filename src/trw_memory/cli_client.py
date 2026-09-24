"""Daemon-backed trw-memory CLI verbs (PRD-CORE-298 FR02).

``store``, ``recall``, ``search``, ``forget``, ``consolidate``, ``export`` and
``status`` travel over the loopback daemon and present this checkout's grant, so
the CLI neither reads outside the grant nor becomes a second writer on the store
the daemon owns. ``--namespace`` defaults to the checkout's pinned
``project_namespace``, so a moved checkout still reaches its rows; an unpinned
directory falls back to its derived project identity. The client starts a daemon
when none is running; any refusal -- still unreachable, no grant, a namespace
outside the grant -- prints one line naming the remedy and exits 1. There is no
local fallback.

``reembed``, ``import`` and ``restore`` still write a store directly, so they
refuse while a daemon record exists rather than become that second writer.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastmcp.exceptions import ToolError

from trw_memory.cli_formatters import (
    StatusDict,
    entry_to_export_dict,
    format_export_summary,
    format_results,
    format_status,
    format_store_result,
)
from trw_memory.cli_storage import write_export
from trw_memory.client import MemoryClient
from trw_memory.daemon import (
    DaemonInfo,
    DaemonPaths,
    DiscoveryAbsent,
    read_checkout_grant,
    read_checkout_pin,
    read_live_discovery,
)
from trw_memory.daemon.client import DaemonClient
from trw_memory.exceptions import DaemonError
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.identity import resolve_project_identity

#: Result statuses that mean the verb did not happen.
_FAILURE_STATUSES = frozenset({"invalid", "forbidden", "error", "not_found"})

#: Rows per ``memory_list_page`` call.
_EXPORT_PAGE_SIZE = 500


async def _status(client: DaemonClient, namespace: str, _args: Namespace) -> dict[str, Any]:
    result: dict[str, Any] = await client.status(namespace)
    if result.get("status") in _FAILURE_STATUSES:
        return result
    return dict(
        StatusDict(
            namespace=namespace,
            entry_count=result["total_entries"],
            backend=result["config"]["storage_backend"],
            storage_path=str(client.paths.store),
        )
    )


async def _export(client: DaemonClient, namespace: str, _args: Namespace) -> dict[str, Any]:
    """Every row of *namespace*, materialized before anything is written.

    A later page failing must not leave an apparently successful prefix, so no
    output is opened until the last page has arrived.
    """
    rows: list[dict[str, Any]] = []
    after: dict[str, str] | None = None
    while True:
        page: dict[str, Any] = await client.list_page(namespace, _EXPORT_PAGE_SIZE, after)
        if page.get("status") != "ok":
            return page
        rows.extend(entry_to_export_dict(MemoryEntry.model_validate(row)) for row in page["entries"])
        resume = page["next"]
        if resume is None:
            return {"status": "ok", "rows": rows}
        if after is not None and (resume["updated_at"], resume["entry_id"]) >= (after["updated_at"], after["entry_id"]):
            return {"status": "error", "error": "the export cursor did not advance; no output written"}
        after = resume


_VERBS: dict[str, Callable[[DaemonClient, str, Namespace], Awaitable[Any]]] = {
    "store": lambda d, ns, a: d.store(a.summary, ns, tags=a.tags or None, importance=a.importance, detail=a.detail),
    "recall": lambda d, ns, a: d.recall(a.query, ns, limit=a.limit, tags=a.tags or None),
    "search": lambda d, ns, a: d.search(ns, tags=a.tags or None, status=a.status, limit=a.limit),
    "forget": lambda d, ns, a: d.forget(a.memory_id, ns),
    "consolidate": lambda d, ns, a: d.consolidate(ns, dry_run=a.dry_run),
    "export": _export,
    "status": _status,
}


def daemon_client() -> DaemonClient:
    """A client presenting the grant of the checkout enclosing the working directory."""
    return DaemonClient(read_checkout_grant(Path.cwd()))


def _render(args: Namespace, result: dict[str, Any]) -> str:
    if args.command == "status":
        return format_status(StatusDict(**result), fmt=args.fmt)  # type: ignore[typeddict-item]
    if args.command == "store":
        return format_store_result(result)
    if args.command in ("recall", "search"):
        return format_results(result["memories" if args.command == "recall" else "entries"], fmt=args.fmt)
    if args.command == "forget":
        return f"Deleted: {args.memory_id}"
    return json.dumps(result, indent=2, default=str)


async def handle_daemon_verb(args: Namespace) -> int:
    """Run ``args.command`` over the daemon; 0 on success, 1 on any refusal."""
    try:
        namespace = args.namespace or read_checkout_pin(Path.cwd()) or resolve_project_identity().namespace
        result = await _VERBS[args.command](daemon_client(), namespace, args)
    except (DaemonError, ToolError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not isinstance(result, dict) or result.get("status") in _FAILURE_STATUSES:
        detail = result.get("error", result.get("status")) if isinstance(result, dict) else repr(result)
        print(f"Error: {args.command} in {namespace}: {detail}", file=sys.stderr)
        return 1
    if args.command == "export":
        write_export(args, result["rows"], format_export_summary)
    else:
        print(_render(args, result))
    return 0


def refused_beside_daemon(verb: str) -> bool:
    """Print a refusal and return True when a daemon record exists: *verb* would be a second writer."""
    found = read_live_discovery(DaemonPaths.resolve(create=False))
    if isinstance(found, DiscoveryAbsent):
        return False
    owner = f"pid {found.pid} at {found.url}" if isinstance(found, DaemonInfo) else f"recorded in {found.path}"
    print(
        f"Error: the trw-memory daemon ({owner}) owns the store, and {verb} writes it directly. "
        f"Stop the daemon, then retry {verb}.",
        file=sys.stderr,
    )
    return True


async def handle_reembed(args: Namespace, *, client_cls: type[MemoryClient]) -> int:
    if refused_beside_daemon("reembed"):
        return 1
    client = client_cls(namespace=args.namespace, mode="local")
    try:
        result = await client.reembed(batch_size=args.batch_size)
    finally:
        await client.close()
    if args.fmt == "json":
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"{result['namespace']}: {result['reembedded']} re-embedded, {result['already_current']} already current, "
            f"{result['skipped']} skipped of {result['examined']} rows; warm tier {result['warm_reembedded']} "
            f"of {result['warm_examined']}; model {result['embedding_model']}"
        )
    return 0
