"""``trw-memory namespace`` — the FR01 repair verbs, over the daemon.

PRD-CORE-253 FR05. Three sub-verbs, all of which travel over the loopback
daemon rather than opening the store: a CLI invocation that opened SQLite would
be an extra writer on a store the daemon is meant to own alone, which is the
contention this PRD exists to remove.

``rename <old> <new>``
    Carry a moved or renamed checkout's rows forward under its new identity.
``merge <src> <dst>``
    The explicit gesture for "I want these two clones to share memory".
``doctor [namespace]``
    Report whether this checkout looks moved, and name the repair. Read-only.

A refusal -- an unreachable daemon, a permission denial, a populated rename
destination -- prints one line and exits non-zero. A person running a repair
verb should not have to read a traceback to learn they need write access.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from trw_memory.daemon.client import DaemonClient
from trw_memory.exceptions import DaemonError

__all__ = ["handle_namespace"]

#: Result statuses that mean the operation did not happen.
_FAILURE_STATUSES = frozenset({"invalid", "forbidden", "error"})


def _print_curate(result: dict[str, Any]) -> None:
    print(
        f"{result['status']}: {result['source']} -> {result['destination']} "
        f"(source_rows={result['source_rows']}, moved={result['moved']}, skipped={result['skipped']})"
    )
    if result["skipped"]:
        print(f"  {result['skipped']} row(s) stayed in the source: the destination already held that id.")


def _print_diagnosis(result: dict[str, Any]) -> None:
    print(f"namespace: {result['namespace']}")
    degraded = result.get("identity_degraded")
    if degraded:
        # Printed FIRST and unconditionally: every line below is a statement
        # about an identity this one says could not be established.
        print(
            f"WARNING: canonical identity unavailable ({degraded}); the namespace above was "
            "derived from this directory's path and may differ from the one holding this "
            "project's rows"
        )
    observation = result.get("moved_checkout")
    if observation is None:
        print("no moved-checkout signal (this identity has rows, or no same-slug sibling does)")
        return
    print("This checkout looks moved or renamed. Its identity has no rows, but these do:")
    for candidate in observation["candidates"]:
        print(f"  {candidate['namespace']}  {candidate['rows']} rows")
    print(f"Repair with:\n  {observation['repair_command']}")


async def handle_namespace(args: argparse.Namespace, *, client: DaemonClient | None = None) -> int:
    """Dispatch ``trw-memory namespace <action>`` over the daemon.

    Args:
        args: Parsed arguments carrying ``namespace_action`` and its operands.
        client: Daemon client. Defaults to one resolved from the user-space
            paths; injected by tests that point at a throwaway daemon.

    Returns:
        0 on success, 1 on any refusal or unreachable daemon.
    """
    daemon = client or DaemonClient()
    action = args.namespace_action
    try:
        if action == "doctor":
            result = await daemon.call_tool("memory_namespace_diagnose", {"namespace": args.namespace})
        else:
            tool = "memory_namespace_merge" if action == "merge" else "memory_namespace_rename"
            result = await daemon.call_tool(tool, {"source": args.source, "destination": args.destination})
    except DaemonError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not isinstance(result, dict):
        print(f"Error: unexpected result from the daemon: {type(result).__name__}", file=sys.stderr)
        return 1
    if result.get("status") in _FAILURE_STATUSES:
        print(f"Error: {result.get('error', result['status'])}", file=sys.stderr)
        return 1

    if action == "doctor":
        _print_diagnosis(result)
    else:
        _print_curate(result)
    return 0
