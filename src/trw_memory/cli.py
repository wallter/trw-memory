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
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import ParamSpec, TypeVar, cast, overload

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
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.interface import StorageBackend

__all__ = ["main"]

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


@overload
def _cli_error_boundary(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]: ...


@overload
def _cli_error_boundary(fn: Callable[P, R]) -> Callable[P, R]: ...


def _cli_error_boundary(fn: Callable[P, object]) -> Callable[P, object]:
    """Wrap a CLI subcommand with uniform error handling.

    Catches all exceptions, prints to stderr, logs with structlog,
    and raises ``SystemExit(1)``.  ``SystemExit`` is re-raised unchanged.

    Works with both sync and async callables.
    """

    async_fn = cast(Callable[P, Awaitable[object]], fn)
    sync_fn = fn

    @functools.wraps(fn)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
        try:
            return await async_fn(*args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("cli_command_failed", command=fn.__name__, error=str(exc), exc_info=True)
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    @functools.wraps(fn)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
        try:
            return sync_fn(*args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("cli_command_failed", command=fn.__name__, error=str(exc), exc_info=True)
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    import asyncio as _asyncio

    if _asyncio.iscoroutinefunction(fn):
        return cast(Callable[P, object], async_wrapper)
    return cast(Callable[P, object], sync_wrapper)


def _open_validated_backend(config: MemoryConfig, namespace: str) -> tuple[str, StorageBackend]:
    """Validate namespace before backend creation to keep filesystem access scoped."""
    validated_namespace = validate_namespace(namespace)
    return validated_namespace, _create_local_backend(config, validated_namespace)


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
    namespace, backend = _open_validated_backend(config, args.namespace)
    try:
        embedder = get_local_embedder(model_name=config.embedding_model, dim=config.embedding_dim)
        result = consolidate_cycle(
            storage=backend,
            embedder=embedder,
            dry_run=args.dry_run,
            namespace=namespace,
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
    namespace, backend = _open_validated_backend(config, args.namespace)
    try:
        entries = backend.list_entries(namespace=namespace, limit=10000)
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
    namespace, backend = _open_validated_backend(config, args.namespace)

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
                namespace=namespace,
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
    namespace, backend = _open_validated_backend(config, args.namespace)
    try:
        count = backend.count(namespace=namespace)
        status_info = StatusDict(
            namespace=namespace,
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


def _resolve_base_and_db(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve ``(base_dir, db_path)`` for CLI handlers that touch storage."""
    config = MemoryConfig()
    namespace = validate_namespace(args.namespace)
    if getattr(args, "db", None):
        db_path = Path(args.db).resolve()
        base_dir = db_path.parent
    else:
        base_dir = (Path(config.storage_path) / namespace.replace(":", "_")).resolve()
        db_path = base_dir / config.sqlite_db_name
    return base_dir, db_path


@_cli_error_boundary
async def _handle_restore(args: argparse.Namespace) -> int:
    """Handle the 'restore' subcommand (PRD-CORE-140 cold + PRD-INFRA-065 snapshot).

    Dispatches to cold-tier rebuild when ``--from-cold`` is set, or snapshot
    copy when ``--from-snapshot=latest|<name>`` is set. The argparse layer
    enforces mutual exclusion.
    """
    if getattr(args, "from_snapshot", None) is not None:
        return await _restore_from_snapshot(args)
    return await _restore_from_cold(args)


@_cli_error_boundary
async def _restore_from_cold(args: argparse.Namespace) -> int:
    """Rebuild the SQLite DB from the cold YAML tier (PRD-CORE-140 FR02)."""
    import sqlite3

    from trw_memory.storage._cold_rebuild import rebuild_from_cold
    from trw_memory.storage._schema import ensure_schema

    base_dir, db_path = _resolve_base_and_db(args)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    cold_base = base_dir / "memory" / "cold"
    total_yaml = sum(1 for _ in cold_base.rglob("*.yaml")) if cold_base.exists() else 0

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        rebuilt = rebuild_from_cold(base_dir, conn)
    finally:
        conn.close()

    skipped = max(total_yaml - rebuilt, 0)
    print(f"Rebuilt {rebuilt} entries from cold tier ({skipped} skipped)")
    return 0


@_cli_error_boundary
async def _restore_from_snapshot(args: argparse.Namespace) -> int:
    """Restore the SQLite DB from a snapshot (PRD-INFRA-065)."""
    from trw_memory.storage._snapshot import (
        SnapshotError,
        list_snapshots,
        restore_from_snapshot,
        snapshots_base_dir,
    )

    base_dir, db_path = _resolve_base_and_db(args)
    target = str(args.from_snapshot).strip()

    if target == "latest":
        listing = list_snapshots(base_dir)
        # Prefer daily newest, fall back to weekly newest.
        candidates = listing["daily"] or listing["weekly"]
        if not candidates:
            print("No snapshots found to restore from.", file=sys.stderr)
            return 1
        snapshot = candidates[0]
    else:
        # Search both tiers for the requested name.
        listing = list_snapshots(base_dir)
        match = next(
            (p for tier_files in listing.values() for p in tier_files if p.name == target),
            None,
        )
        if match is None:
            # Fall back to raw path interpretation under snapshots dir.
            candidate = snapshots_base_dir(base_dir) / target
            if candidate.exists():
                match = candidate
        if match is None:
            print(f"Snapshot not found: {target}", file=sys.stderr)
            return 1
        snapshot = match

    try:
        restore_from_snapshot(base_dir, snapshot, db_path)
    except SnapshotError as exc:
        print(f"Snapshot restore failed: {exc}", file=sys.stderr)
        return 1
    print(f"Restored {db_path} from {snapshot}")
    return 0


@_cli_error_boundary
async def _handle_snapshot(args: argparse.Namespace) -> int:
    """Handle the 'snapshot' subcommand (PRD-INFRA-065)."""
    from trw_memory.storage._snapshot import (
        list_snapshots,
        rotate_snapshots,
        take_daily_snapshot,
        take_weekly_snapshot,
    )

    base_dir, db_path = _resolve_base_and_db(args)
    config = MemoryConfig()

    action = args.snapshot_action
    if action == "create":
        if not db_path.exists():
            print(f"Source DB does not exist: {db_path}", file=sys.stderr)
            return 1
        if args.tier == "weekly":
            weekly_result = take_weekly_snapshot(
                base_dir,
                db_path,
                keep_weekly=config.memory_snapshot_weekly_keep,
                force=args.force,
            )
            if weekly_result is None:
                print("Today is not Sunday UTC; use --force to override.")
                return 0
            print(f"Created weekly snapshot: {weekly_result}")
        else:
            daily_result = take_daily_snapshot(
                base_dir,
                db_path,
                keep_daily=config.memory_snapshot_daily_keep,
            )
            print(f"Created daily snapshot: {daily_result}")
        return 0

    if action == "list":
        listing = list_snapshots(base_dir)
        if args.fmt == "json":
            print(
                json.dumps(
                    {
                        "daily": [str(p) for p in listing["daily"]],
                        "weekly": [str(p) for p in listing["weekly"]],
                    },
                    indent=2,
                )
            )
        else:
            if not listing["daily"] and not listing["weekly"]:
                print("(no snapshots)")
            if listing["daily"]:
                print("daily (newest first):")
                for p in listing["daily"]:
                    print(f"  {p.name}")
            if listing["weekly"]:
                print("weekly (newest first):")
                for p in listing["weekly"]:
                    print(f"  {p.name}")
        return 0

    if action == "rotate":
        rotate_result = rotate_snapshots(
            base_dir,
            keep_daily=config.memory_snapshot_daily_keep,
            keep_weekly=config.memory_snapshot_weekly_keep,
        )
        print(
            f"Rotated: daily(kept={rotate_result.daily_kept}, pruned={rotate_result.daily_pruned}), "
            f"weekly(kept={rotate_result.weekly_kept}, pruned={rotate_result.weekly_pruned})"
        )
        return 0

    print(f"Unknown snapshot action: {action}", file=sys.stderr)
    return 1


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
        "restore": _handle_restore,
        "snapshot": _handle_snapshot,
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
