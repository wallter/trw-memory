"""trw-memory CLI — command-line interface for the trw-memory package.

Provides 8 subcommands: store, recall, search, consolidate, export,
import, status, forget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trw_memory.cli_formatters import (
    format_export_summary,
    format_import_summary,
    format_results,
    format_status,
    format_store_result,
)
from trw_memory.client import MemoryClient, _create_local_backend
from trw_memory.lifecycle.consolidation import consolidate_cycle
from trw_memory.models.config import MemoryConfig


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="trw-memory",
        description="trw-memory CLI — manage your AI agent memory",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- store ---
    p_store = subparsers.add_parser("store", help="Store a memory entry")
    p_store.add_argument("--summary", required=True, help="Core knowledge statement")
    p_store.add_argument("--detail", default="", help="Extended explanation")
    p_store.add_argument("--tags", action="append", default=[], help="Categorization tags (repeatable)")
    p_store.add_argument("--importance", type=float, default=0.5, help="Importance score 0.0-1.0")
    p_store.add_argument("--namespace", default="default", help="Namespace (default: default)")

    # --- recall ---
    p_recall = subparsers.add_parser("recall", help="Search by keyword")
    p_recall.add_argument("query", help="Free-text search query")
    p_recall.add_argument("--limit", type=int, default=10, help="Max results")
    p_recall.add_argument("--tags", action="append", default=[], help="Filter tags (repeatable)")
    p_recall.add_argument("--namespace", default="default", help="Namespace")
    p_recall.add_argument("--format", dest="fmt", choices=["table", "json", "compact"], default="table", help="Output format")

    # --- search ---
    p_search = subparsers.add_parser("search", help="Filter-based search")
    p_search.add_argument("--tags", action="append", default=[], help="Filter tags (repeatable)")
    p_search.add_argument("--min-importance", type=float, default=0.0, help="Minimum importance")
    p_search.add_argument("--since", default=None, help="ISO datetime filter")
    p_search.add_argument("--limit", type=int, default=50, help="Max results")
    p_search.add_argument("--namespace", default="default", help="Namespace")
    p_search.add_argument("--format", dest="fmt", choices=["table", "json", "compact"], default="table", help="Output format")

    # --- consolidate ---
    p_consolidate = subparsers.add_parser("consolidate", help="Trigger consolidation")
    p_consolidate.add_argument("--namespace", default="default", help="Namespace")
    p_consolidate.add_argument("--dry-run", action="store_true", help="Preview without writing")

    # --- export ---
    p_export = subparsers.add_parser("export", help="Export entries to file")
    p_export.add_argument("--format", dest="fmt", choices=["json", "yaml"], default="json", help="Export format")
    p_export.add_argument("--output", default=None, help="Output file path (default: stdout)")
    p_export.add_argument("--namespace", default="default", help="Namespace")

    # --- import ---
    p_import = subparsers.add_parser("import", help="Import entries from file")
    p_import.add_argument("path", help="Input file path")
    p_import.add_argument("--namespace", default="default", help="Namespace")
    p_import.add_argument("--merge", action="store_true", help="Skip existing IDs")

    # --- status ---
    p_status = subparsers.add_parser("status", help="Show memory system status")
    p_status.add_argument("--namespace", default="default", help="Namespace")
    p_status.add_argument("--format", dest="fmt", choices=["table", "json"], default="table", help="Output format")

    # --- forget ---
    p_forget = subparsers.add_parser("forget", help="Delete a memory entry")
    p_forget.add_argument("memory_id", help="ID of the entry to delete")
    p_forget.add_argument("--namespace", default="default", help="Namespace")

    return parser


async def _handle_store(args: argparse.Namespace) -> int:
    """Handle the 'store' subcommand."""
    client = MemoryClient(namespace=args.namespace, mode="local")
    try:
        result = await client.store(
            content=args.summary,
            tags=args.tags if args.tags else None,
            importance=args.importance,
            detail=args.detail,
        )
        print(format_store_result(result))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


async def _handle_recall(args: argparse.Namespace) -> int:
    """Handle the 'recall' subcommand."""
    client = MemoryClient(namespace=args.namespace, mode="local")
    try:
        tags = args.tags if args.tags else None
        results = await client.recall(
            query=args.query,
            limit=args.limit,
            tags=tags,
        )
        print(format_results(results, fmt=args.fmt))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


async def _handle_search(args: argparse.Namespace) -> int:
    """Handle the 'search' subcommand."""
    since: datetime | None = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Error: invalid datetime format: {args.since}", file=sys.stderr)
            return 1

    client = MemoryClient(namespace=args.namespace, mode="local")
    try:
        tags = args.tags if args.tags else None
        results = await client.search(
            tags=tags,
            min_importance=args.min_importance,
            since=since,
            limit=args.limit,
        )
        print(format_results(results, fmt=args.fmt))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


async def _handle_consolidate(args: argparse.Namespace) -> int:
    """Handle the 'consolidate' subcommand."""
    try:
        config = MemoryConfig()
        backend = _create_local_backend(config, args.namespace)
        try:
            result = consolidate_cycle(
                storage=backend,
                embedder=None,
                dry_run=args.dry_run,
                namespace=args.namespace,
                config=config,
            )
            print(json.dumps(result, indent=2, default=str))
            return 0
        finally:
            backend.close()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


async def _handle_export(args: argparse.Namespace) -> int:
    """Handle the 'export' subcommand."""
    try:
        config = MemoryConfig()
        backend = _create_local_backend(config, args.namespace)
        try:
            entries = backend.list_entries(namespace=args.namespace, limit=10000)
            data = [_entry_to_export_dict(e) for e in entries]

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
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


async def _handle_import(args: argparse.Namespace) -> int:
    """Handle the 'import' subcommand."""
    file_path = Path(args.path)
    if not file_path.exists():
        print(f"Error: file not found: {args.path}", file=sys.stderr)
        return 1

    raw_text = file_path.read_text(encoding="utf-8")

    try:
        if file_path.suffix in (".yaml", ".yml"):
            from ruamel.yaml import YAML

            yaml = YAML()
            data = yaml.load(raw_text)
        else:
            data = json.loads(raw_text)
    except Exception as exc:
        print(f"Error: failed to parse {args.path}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, list):
        print("Error: expected a JSON/YAML array of entries", file=sys.stderr)
        return 1

    client = MemoryClient(namespace=args.namespace, mode="local")
    imported = 0
    skipped = 0
    try:
        for entry_data in data:
            if not isinstance(entry_data, dict):
                continue

            if args.merge:
                # Check if entry already exists by ID
                eid = entry_data.get("id", "")
                if eid:
                    try:
                        existing = await client.recall(eid, limit=1)
                        if existing and any(
                            r.get("memory_id") == eid for r in existing
                        ):
                            skipped += 1
                            continue
                    except Exception:
                        pass

            content = str(entry_data.get("content", ""))
            if not content.strip():
                skipped += 1
                continue

            tags = entry_data.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            importance = float(entry_data.get("importance", 0.5))
            detail = str(entry_data.get("detail", ""))

            await client.store(
                content=content,
                tags=tags,
                importance=importance,
                detail=detail,
            )
            imported += 1

        print(format_import_summary(imported, skipped))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


async def _handle_status(args: argparse.Namespace) -> int:
    """Handle the 'status' subcommand."""
    try:
        config = MemoryConfig()
        backend = _create_local_backend(config, args.namespace)
        try:
            count = backend.count(namespace=args.namespace)
            status_info: dict[str, Any] = {
                "namespace": args.namespace,
                "entry_count": count,
                "backend": config.storage_backend,
                "storage_path": config.storage_path,
            }
            print(format_status(status_info, fmt=args.fmt))
            return 0
        finally:
            backend.close()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


async def _handle_forget(args: argparse.Namespace) -> int:
    """Handle the 'forget' subcommand."""
    client = MemoryClient(namespace=args.namespace, mode="local")
    try:
        result = await client.forget(args.memory_id)
        mid = result.get("memory_id", "")
        print(f"Deleted: {mid}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


def _entry_to_export_dict(entry: Any) -> dict[str, Any]:
    """Convert a MemoryEntry to a serializable dict for export."""
    return {
        "id": entry.id,
        "content": entry.content,
        "detail": entry.detail,
        "tags": list(entry.tags),
        "importance": entry.importance,
        "status": str(entry.status),
        "namespace": entry.namespace,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "metadata": dict(entry.metadata) if entry.metadata else {},
    }


async def _dispatch(args: argparse.Namespace) -> int:
    """Route to the appropriate handler based on subcommand."""
    handlers = {
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
        return 1
    return await handler(args)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the trw-memory CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return asyncio.run(_dispatch(args))


if __name__ == "__main__":
    sys.exit(main())
