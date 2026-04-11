"""trw-memory CLI — command-line interface for the trw-memory package.

Provides 8 subcommands: store, recall, search, consolidate, export,
import, status, forget.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import inspect
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import structlog

from trw_memory.cli_formatters import (
    StatusDict,
    entry_to_export_dict,
    format_export_summary,
    format_import_summary,
    format_results,
    format_status,
    format_store_result,
)
from trw_memory.cli_parser import build_parser
from trw_memory.client import MemoryClient, _create_local_backend
from trw_memory.embeddings import get_local_embedder
from trw_memory.lifecycle.consolidation import consolidate_cycle
from trw_memory.models.config import MemoryConfig

__all__ = ["main"]

logger = structlog.get_logger(__name__)


def _cli_error_boundary(fn: Callable[..., object]) -> Callable[..., object]:
    """Wrap a CLI subcommand with uniform error handling.

    Catches all exceptions, prints to stderr, logs with structlog,
    and raises ``SystemExit(1)``.  ``SystemExit`` is re-raised unchanged.

    Works with both sync and async callables.
    """

    @functools.wraps(fn)
    async def async_wrapper(*args: object, **kwargs: object) -> object:
        try:
            return await fn(*args, **kwargs)  # type: ignore[misc]
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("cli_command_failed", command=fn.__name__, error=str(exc), exc_info=True)
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    @functools.wraps(fn)
    def sync_wrapper(*args: object, **kwargs: object) -> object:
        try:
            return fn(*args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("cli_command_failed", command=fn.__name__, error=str(exc), exc_info=True)
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    import asyncio as _asyncio

    return async_wrapper if _asyncio.iscoroutinefunction(fn) else sync_wrapper


@_cli_error_boundary
async def _handle_store(args: argparse.Namespace) -> int:
    """Handle the 'store' subcommand."""
    client = MemoryClient(namespace=args.namespace, mode="local")
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


@_cli_error_boundary
async def _handle_recall(args: argparse.Namespace) -> int:
    """Handle the 'recall' subcommand."""
    client = MemoryClient(namespace=args.namespace, mode="local")
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


@_cli_error_boundary
async def _handle_search(args: argparse.Namespace) -> int:
    """Handle the 'search' subcommand."""
    since: datetime | None = None
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    client = MemoryClient(namespace=args.namespace, mode="local")
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


@_cli_error_boundary
async def _handle_consolidate(args: argparse.Namespace) -> int:
    """Handle the 'consolidate' subcommand."""
    config = MemoryConfig()
    backend = _create_local_backend(config, args.namespace)
    try:
        embedder = get_local_embedder(model_name=config.embedding_model, dim=config.embedding_dim)
        result = consolidate_cycle(
            storage=backend,
            embedder=embedder,
            dry_run=args.dry_run,
            namespace=args.namespace,
            config=config,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        backend.close()


@_cli_error_boundary
async def _handle_export(args: argparse.Namespace) -> int:
    """Handle the 'export' subcommand."""
    config = MemoryConfig()
    backend = _create_local_backend(config, args.namespace)
    try:
        entries = backend.list_entries(namespace=args.namespace, limit=10000)
        data = [entry_to_export_dict(e) for e in entries]

        if args.fmt == "yaml":
            from ruamel.yaml import YAML

            yaml = YAML()
            yaml.default_flow_style = False
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    yaml.dump(data, f)
            else:
                yaml.dump(data, sys.stdout)
        else:
            text = json.dumps(data, indent=2, default=str)
            if args.output:
                Path(args.output).write_text(text, encoding="utf-8")
            else:
                print(text)

        print(format_export_summary(len(data), args.output), file=sys.stderr)
        return 0
    finally:
        backend.close()


@_cli_error_boundary
async def _handle_import(args: argparse.Namespace) -> int:
    """Handle the 'import' subcommand."""
    file_path = Path(args.path)
    if not file_path.exists():
        print(f"Error: file not found: {args.path}", file=sys.stderr)
        return 1

    raw_text = file_path.read_text(encoding="utf-8")

    if file_path.suffix in (".yaml", ".yml"):
        from ruamel.yaml import YAML

        yaml = YAML()
        data = yaml.load(raw_text)
    else:
        data = json.loads(raw_text)

    if not isinstance(data, list):
        print("Error: expected a JSON/YAML array of entries", file=sys.stderr)
        return 1

    from trw_memory.integrations._backend import make_entry

    config = MemoryConfig()
    backend = _create_local_backend(config, args.namespace)

    imported = 0
    skipped = 0
    try:
        for entry_data in data:
            if not isinstance(entry_data, dict):
                continue

            if args.merge:
                eid = entry_data.get("id", "")
                if eid:
                    existing = backend.get(eid)
                    if existing is not None:
                        skipped += 1
                        continue

            content = str(entry_data.get("content", ""))
            if not content.strip():
                skipped += 1
                continue

            tags = entry_data.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            importance = float(entry_data.get("importance", 0.5))
            detail = str(entry_data.get("detail", ""))

            entry = make_entry(
                content=content,
                namespace=args.namespace,
                tags=tags,
                importance=importance,
                detail=detail,
            )
            backend.store(entry)
            imported += 1

        print(format_import_summary(imported, skipped))
        return 0
    finally:
        backend.close()


@_cli_error_boundary
async def _handle_status(args: argparse.Namespace) -> int:
    """Handle the 'status' subcommand."""
    config = MemoryConfig()
    backend = _create_local_backend(config, args.namespace)
    try:
        count = backend.count(namespace=args.namespace)
        status_info = StatusDict(
            namespace=args.namespace,
            entry_count=count,
            backend=config.storage_backend,
            storage_path=config.storage_path,
        )
        print(format_status(status_info, fmt=args.fmt))
        return 0
    finally:
        backend.close()


@_cli_error_boundary
async def _handle_forget(args: argparse.Namespace) -> int:
    """Handle the 'forget' subcommand."""
    client = MemoryClient(namespace=args.namespace, mode="local")
    try:
        result = await client.forget(args.memory_id)
        mid = result.get("memory_id", "")
        print(f"Deleted: {mid}")
        return 0
    finally:
        await client.close()


async def _dispatch(args: argparse.Namespace) -> int:
    """Route to the appropriate handler based on subcommand."""
    handlers: dict[str, Callable[..., object]] = {
        "store": _handle_store,
        "recall": _handle_recall,
        "search": _handle_search,
        "consolidate": _handle_consolidate,
        "export": _handle_export,
        "import": _handle_import,
        "status": _handle_status,
        "forget": _handle_forget,
    }
    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        logger.error("cli_unknown_command", command=args.command)
        return 1
    try:
        coro = handler(args)
        if inspect.isawaitable(coro):
            rc = await coro
        else:
            rc = coro
        return rc if isinstance(rc, int) else 1
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for the trw-memory CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    from trw_memory._logging import configure_logging

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Configure logging from CLI flags
    verbosity = args.verbose
    if args.quiet:
        verbosity = -1

    configure_logging(
        verbosity=verbosity,
        log_level=args.log_level,
    )

    return asyncio.run(_dispatch(args))


if __name__ == "__main__":
    sys.exit(main())
