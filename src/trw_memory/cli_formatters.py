"""Output formatters for the trw-memory CLI.

Supports three output modes:
- **table**: Human-readable aligned columns (no external deps).
- **json**: Machine-readable pretty-printed JSON.
- **compact**: One-line-per-result summary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TypedDict

from trw_memory.models.memory import MemoryEntry


class StatusDict(TypedDict):
    """Shape of the status info dict passed to format_status."""

    namespace: str
    entry_count: int
    backend: str
    storage_path: str


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text to *max_len* characters, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_tags(tags: list[str]) -> str:
    """Format a tag list as ``[tag1,tag2]``."""
    if not tags:
        return "[]"
    return "[" + ",".join(tags) + "]"


def format_results(results: Sequence[Mapping[str, object]], fmt: str = "table") -> str:
    """Format recall/search results.

    Args:
        results: List of result dicts from recall/search.
        fmt: Output format — ``"table"``, ``"json"``, or ``"compact"``.

    Returns:
        Formatted string.
    """
    if fmt == "json":
        return json.dumps(results, indent=2, default=str)

    if not results:
        return "No results found."

    if fmt == "compact":
        lines: list[str] = []
        for r in results:
            mid = r.get("memory_id", "?")
            score = r.get("score", 0.0)
            content = _truncate(str(r.get("content", "")), 80)
            lines.append(f"{mid}  score={score:.2f}  {content}")
        return "\n".join(lines)

    # Table format (default)
    header = f"{'ID':<12} {'Score':>6} {'Importance':>10} {'Tags':<16} {'Content'}"
    sep = "-" * len(header)
    rows: list[str] = [header, sep]
    for r in results:
        mid = str(r.get("memory_id", "?"))[:12]
        score = float(r.get("score", 0.0))  # type: ignore[arg-type]
        importance = float(r.get("importance", 0.0))  # type: ignore[arg-type]
        raw_tags = r.get("tags", [])
        tags = _format_tags(raw_tags if isinstance(raw_tags, list) else [])[:16]
        content = _truncate(str(r.get("content", "")))
        rows.append(f"{mid:<12} {score:>6.2f} {importance:>10.2f} {tags:<16} {content}")
    return "\n".join(rows)


def format_status(status: StatusDict, fmt: str = "table") -> str:
    """Format status output.

    Args:
        status: Dict with keys like ``entry_count``, ``namespace``, etc.
        fmt: Output format — ``"table"`` or ``"json"``.

    Returns:
        Formatted string.
    """
    if fmt == "json":
        return json.dumps(status, indent=2, default=str)

    lines: list[str] = ["Memory System Status", "=" * 40]
    for key, value in status.items():
        label = key.replace("_", " ").title()
        lines.append(f"  {label:<24} {value}")
    return "\n".join(lines)


def format_store_result(result: Mapping[str, object]) -> str:
    """Format store confirmation.

    Args:
        result: Dict from ``MemoryClient.store()``.

    Returns:
        Human-readable confirmation line.
    """
    mid = result.get("memory_id", "unknown")
    ns = result.get("namespace", "default")
    ts = result.get("timestamp", "")
    return f"Stored: {mid} (namespace={ns}, timestamp={ts})"


def format_export_summary(count: int, path: str | None) -> str:
    """Format export summary.

    Args:
        count: Number of entries exported.
        path: Output file path, or ``None`` for stdout.

    Returns:
        Human-readable summary.
    """
    dest = path or "stdout"
    return f"Exported {count} entries to {dest}"


def format_import_summary(imported: int, skipped: int) -> str:
    """Format import summary.

    Args:
        imported: Number of entries imported.
        skipped: Number of entries skipped (duplicates in merge mode).

    Returns:
        Human-readable summary.
    """
    return f"Imported {imported} entries, skipped {skipped}"


def entry_to_export_dict(entry: MemoryEntry) -> dict[str, object]:
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
