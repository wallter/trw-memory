"""Argument parser for the trw-memory CLI.

Extracted from ``cli.py`` to keep the main module under 300 lines.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="trw-memory",
        description="trw-memory CLI — manage your AI agent memory",
    )

    # Global logging flags
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v=DEBUG, -vv=DEBUG+more). Stacks.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress output except warnings and errors",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Explicit log level (overrides -v/-q/env vars)",
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
    p_recall.add_argument(
        "--format", dest="fmt", choices=["table", "json", "compact"], default="table", help="Output format"
    )

    # --- search ---
    p_search = subparsers.add_parser("search", help="Filter-based search")
    p_search.add_argument("--tags", action="append", default=[], help="Filter tags (repeatable)")
    p_search.add_argument("--min-importance", type=float, default=0.0, help="Minimum importance")
    p_search.add_argument("--since", default=None, help="ISO datetime filter")
    p_search.add_argument("--limit", type=int, default=50, help="Max results")
    p_search.add_argument("--namespace", default="default", help="Namespace")
    p_search.add_argument(
        "--format", dest="fmt", choices=["table", "json", "compact"], default="table", help="Output format"
    )

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

    # --- restore (PRD-CORE-140 + PRD-INFRA-065) ---
    p_restore = subparsers.add_parser(
        "restore",
        help="Rebuild the SQLite DB from the cold YAML tier or a snapshot",
    )
    restore_source = p_restore.add_mutually_exclusive_group(required=True)
    restore_source.add_argument(
        "--from-cold",
        dest="from_cold",
        action="store_true",
        help="Rebuild from the cold YAML tier (PRD-CORE-140)",
    )
    restore_source.add_argument(
        "--from-snapshot",
        dest="from_snapshot",
        default=None,
        metavar="NAME_OR_LATEST",
        help=(
            "Restore from a snapshot — pass 'latest' for the newest daily, "
            "or a filename like 2026-04-13.db / 2026-W15.db (PRD-INFRA-065)"
        ),
    )
    p_restore.add_argument("--namespace", default="default", help="Namespace")
    p_restore.add_argument(
        "--db",
        default=None,
        help="Optional override for the SQLite DB path (default: configured path)",
    )

    # --- snapshot (PRD-INFRA-065) ---
    p_snapshot = subparsers.add_parser(
        "snapshot",
        help="Manage memory DB snapshots (VACUUM INTO rotation)",
    )
    snap_subs = p_snapshot.add_subparsers(dest="snapshot_action", required=True)

    p_snap_create = snap_subs.add_parser("create", help="Create a new snapshot")
    p_snap_create.add_argument(
        "--tier",
        choices=["daily", "weekly"],
        default="daily",
        help="Snapshot tier to write (default: daily)",
    )
    p_snap_create.add_argument("--namespace", default="default", help="Namespace")
    p_snap_create.add_argument("--db", default=None, help="Optional DB path override")
    p_snap_create.add_argument(
        "--force",
        action="store_true",
        help="For weekly: take snapshot even on non-Sunday",
    )

    p_snap_list = snap_subs.add_parser("list", help="List available snapshots")
    p_snap_list.add_argument("--namespace", default="default", help="Namespace")
    p_snap_list.add_argument("--db", default=None, help="Optional DB path override")
    p_snap_list.add_argument(
        "--format", dest="fmt", choices=["table", "json"], default="table",
        help="Output format",
    )

    p_snap_rotate = snap_subs.add_parser("rotate", help="Prune snapshots to budget")
    p_snap_rotate.add_argument("--namespace", default="default", help="Namespace")
    p_snap_rotate.add_argument("--db", default=None, help="Optional DB path override")

    return parser
