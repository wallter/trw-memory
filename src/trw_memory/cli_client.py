"""Client-backed trw-memory CLI handlers."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

from trw_memory.client import MemoryClient


async def handle_store(
    args: Namespace,
    *,
    client_cls: type[MemoryClient],
    format_store_result: Callable[..., str],
) -> int:
    client = client_cls(namespace=args.namespace, mode="local")
    try:
        result = await client.store(
            content=args.summary,
            tags=args.tags or None,
            importance=args.importance,
            detail=args.detail,
        )
        print(format_store_result(result))
        return 0
    finally:
        await client.close()


async def handle_recall(
    args: Namespace,
    *,
    client_cls: type[MemoryClient],
    format_results: Callable[..., str],
) -> int:
    client = client_cls(namespace=args.namespace, mode="local")
    try:
        results = await client.recall(
            query=args.query,
            limit=args.limit,
            tags=args.tags or None,
        )
        print(format_results(results, fmt=args.fmt))
        return 0
    finally:
        await client.close()


async def handle_search(
    args: Namespace,
    *,
    client_cls: type[MemoryClient],
    format_results: Callable[..., str],
) -> int:
    from datetime import datetime, timezone

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    client = client_cls(namespace=args.namespace, mode="local")
    try:
        results = await client.search(
            tags=args.tags or None,
            min_importance=args.min_importance,
            since=since,
            limit=args.limit,
        )
        print(format_results(results, fmt=args.fmt))
        return 0
    finally:
        await client.close()


async def handle_forget(
    args: Namespace,
    *,
    client_cls: type[MemoryClient],
) -> int:
    client = client_cls(namespace=args.namespace, mode="local")
    try:
        result = await client.forget(args.memory_id)
        print(f"Deleted: {result.get('memory_id', '')}")
        return 0
    finally:
        await client.close()
