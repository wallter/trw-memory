"""Storage-backed trw-memory CLI handlers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trw_memory.cli_formatters import StatusDict
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.interface import StorageBackend


def open_validated_backend(
    config: MemoryConfig,
    namespace: str,
    *,
    backend_factory: Callable[[MemoryConfig, str], StorageBackend],
) -> tuple[str, StorageBackend]:
    validated_namespace = validate_namespace(namespace)
    return validated_namespace, backend_factory(config, validated_namespace)


def resolve_base_and_db(args: argparse.Namespace, *, config_cls: type[MemoryConfig]) -> tuple[Path, Path]:
    config = config_cls()
    namespace = validate_namespace(args.namespace)
    if getattr(args, "db", None):
        db_path = Path(args.db).resolve()
        base_dir = db_path.parent
    else:
        base_dir = (Path(config.storage_path) / namespace.replace(":", "_")).resolve()
        db_path = base_dir / config.sqlite_db_name
    return base_dir, db_path


def handle_consolidate(
    args: argparse.Namespace,
    *,
    config_cls: type[MemoryConfig],
    backend_factory: Callable[[MemoryConfig, str], StorageBackend],
    embedder_factory: Callable[..., object],
    consolidate_fn: Callable[..., dict[str, Any]],
) -> int:
    config = config_cls()
    namespace, backend = open_validated_backend(config, args.namespace, backend_factory=backend_factory)
    try:
        embedder = embedder_factory(model_name=config.embedding_model, dim=config.embedding_dim)
        result = consolidate_fn(
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


def handle_export(
    args: argparse.Namespace,
    *,
    config_cls: type[MemoryConfig],
    backend_factory: Callable[[MemoryConfig, str], StorageBackend],
    entry_to_export_dict: Callable[[Any], dict[str, Any]],
    export_summary: Callable[..., str],
) -> int:
    config = config_cls()
    namespace, backend = open_validated_backend(config, args.namespace, backend_factory=backend_factory)
    try:
        entries = backend.list_entries(namespace=namespace, limit=10000)
        data = [entry_to_export_dict(e) for e in entries]

        if args.fmt == "yaml":
            from ruamel.yaml import YAML

            yaml = YAML()
            yaml.default_flow_style = False
            if args.output:
                with open(args.output, "w", encoding="utf-8") as handle:
                    yaml.dump(data, handle)
            else:
                yaml.dump(data, sys.stdout)
        else:
            text = json.dumps(data, indent=2, default=str)
            if args.output:
                Path(args.output).write_text(text, encoding="utf-8")
            else:
                print(text)

        print(export_summary(len(data), args.output), file=sys.stderr)
        return 0
    finally:
        backend.close()


def handle_import(
    args: argparse.Namespace,
    *,
    config_cls: type[MemoryConfig],
    backend_factory: Callable[[MemoryConfig, str], StorageBackend],
    make_entry: Callable[..., Any],
    import_summary: Callable[..., str],
) -> int:
    file_path = Path(args.path)
    if not file_path.exists():
        print(f"Error: file not found: {args.path}", file=sys.stderr)
        return 1

    raw_text = file_path.read_text(encoding="utf-8")
    if file_path.suffix in (".yaml", ".yml"):
        from ruamel.yaml import YAML

        yaml = YAML(typ="safe")
        data = yaml.load(raw_text)
    else:
        data = json.loads(raw_text)

    if not isinstance(data, list):
        print("Error: expected a JSON/YAML array of entries", file=sys.stderr)
        return 1

    config = config_cls()
    namespace, backend = open_validated_backend(config, args.namespace, backend_factory=backend_factory)
    imported = 0
    skipped = 0
    try:
        for entry_data in data:
            if not isinstance(entry_data, dict):
                continue

            if args.merge:
                entry_id = entry_data.get("id", "")
                if entry_id:
                    existing = backend.get(entry_id)
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
            entry = make_entry(
                content=content,
                namespace=namespace,
                tags=tags,
                importance=float(entry_data.get("importance", 0.5)),
                detail=str(entry_data.get("detail", "")),
            )
            backend.store(entry)
            imported += 1

        print(import_summary(imported, skipped))
        return 0
    finally:
        backend.close()


def handle_status(
    args: argparse.Namespace,
    *,
    config_cls: type[MemoryConfig],
    backend_factory: Callable[[MemoryConfig, str], StorageBackend],
    status_dict_cls: Callable[..., StatusDict],
    format_status: Callable[..., str],
) -> int:
    config = config_cls()
    namespace, backend = open_validated_backend(config, args.namespace, backend_factory=backend_factory)
    try:
        status_info = status_dict_cls(
            namespace=namespace,
            entry_count=backend.count(namespace=namespace),
            backend=config.storage_backend,
            storage_path=config.storage_path,
        )
        print(format_status(status_info, fmt=args.fmt))
        return 0
    finally:
        backend.close()


def handle_restore(
    args: argparse.Namespace,
    *,
    config_cls: type[MemoryConfig],
) -> int:
    from trw_memory.storage._cold_rebuild import rebuild_from_cold
    from trw_memory.storage._schema import ensure_schema
    from trw_memory.storage._snapshot import (
        SnapshotError,
        list_snapshots,
        restore_from_snapshot,
        snapshots_base_dir,
    )

    base_dir, db_path = resolve_base_and_db(args, config_cls=config_cls)
    if getattr(args, "from_snapshot", None) is not None:
        target = str(args.from_snapshot).strip()
        listing = list_snapshots(base_dir)
        if target == "latest":
            candidates = listing["daily"] or listing["weekly"]
            if not candidates:
                print("No snapshots found to restore from.", file=sys.stderr)
                return 1
            snapshot = candidates[0]
        else:
            snapshot_match = next((p for tier in listing.values() for p in tier if p.name == target), None)
            if snapshot_match is None:
                candidate = snapshots_base_dir(base_dir) / target
                if candidate.exists():
                    snapshot_match = candidate
            if snapshot_match is None:
                print(f"Snapshot not found: {target}", file=sys.stderr)
                return 1
            snapshot = snapshot_match
        try:
            restore_from_snapshot(base_dir, snapshot, db_path)
        except SnapshotError as exc:
            print(f"Snapshot restore failed: {exc}", file=sys.stderr)
            return 1
        print(f"Restored {db_path} from {snapshot}")
        return 0

    import sqlite3

    db_path.parent.mkdir(parents=True, exist_ok=True)
    cold_base = base_dir / "memory" / "cold"
    total_yaml = sum(1 for _ in cold_base.rglob("*.yaml")) if cold_base.exists() else 0

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        rebuilt = rebuild_from_cold(base_dir, conn)
    finally:
        conn.close()

    print(f"Rebuilt {rebuilt} entries from cold tier ({max(total_yaml - rebuilt, 0)} skipped)")
    return 0


def handle_snapshot(
    args: argparse.Namespace,
    *,
    config_cls: type[MemoryConfig],
) -> int:
    from trw_memory.storage._snapshot import (
        list_snapshots,
        rotate_snapshots,
        take_daily_snapshot,
        take_weekly_snapshot,
    )

    base_dir, db_path = resolve_base_and_db(args, config_cls=config_cls)
    config = config_cls()
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
                for path in listing["daily"]:
                    print(f"  {path.name}")
            if listing["weekly"]:
                print("weekly (newest first):")
                for path in listing["weekly"]:
                    print(f"  {path.name}")
        return 0

    if action == "rotate":
        rotated = rotate_snapshots(
            base_dir,
            keep_daily=config.memory_snapshot_daily_keep,
            keep_weekly=config.memory_snapshot_weekly_keep,
        )
        print(
            f"Rotated: daily(kept={rotated.daily_kept}, pruned={rotated.daily_pruned}), "
            f"weekly(kept={rotated.weekly_kept}, pruned={rotated.weekly_pruned})"
        )
        return 0

    print(f"Unknown snapshot action: {action}", file=sys.stderr)
    return 1
