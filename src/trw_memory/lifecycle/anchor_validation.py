"""Anchor validity computation for code-grounded learnings.

Scores how many of a learning's code anchors still reference valid
symbols in the current codebase. Returns 1.0 when all anchors are
valid (or there are no anchors), 0.0 when all are invalid.

PRD-CORE-111 FR03. Pure function: reads the filesystem only, never
writes (see ``test_compute_anchor_validity_is_pure``).

Note: this module previously also scanned the entire project tree for an
inline ``mcp.trw.recall(id=...)`` comment marker and added a 0.5 bonus to
the score when found (FR05). That scan was removed: it walked and
``read_text()``'d every source/doc file under the project root on every
``trw_learn`` call and every journal replay, which pegged the MCP server at
sustained CPU in repos with many nested worktrees. Markers remain a
documented code-comment convention read by ``trw_mcp.state.anchor_generation``
(``extract_marker_ids``); they no longer affect anchor validity scoring.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.models.memory import Anchor


def compute_anchor_validity(
    anchors: list[Anchor] | list[dict[str, object]],
    project_root: str | Path,
) -> float:
    """Compute what fraction of anchors still point to valid symbols.

    For each anchor, checks:
    1. File exists at anchor.file (or anchor["file"]) relative to project_root
    2. Symbol name appears in the file content (simple text search)

    Accepts both ``list[Anchor]`` (Pydantic models) and ``list[dict]``
    (raw dicts from YAML/JSON) for backward compatibility.

    Args:
        anchors: List of Anchor models or dicts with "file" and "symbol_name" keys.
        project_root: Absolute path to the project root.

    Returns:
        Float 0.0-1.0. Returns 1.0 for empty anchor lists (no anchors = no staleness).
    """
    if not anchors:
        return 1.0

    root = Path(project_root)
    valid_count = 0.0

    for anchor in anchors:
        # Support both Anchor model and raw dict
        if isinstance(anchor, Anchor):
            file_str = anchor.file
            symbol_name = anchor.symbol_name
        else:
            file_str = str(anchor.get("file", ""))
            symbol_name = str(anchor.get("symbol_name", ""))

        if not symbol_name or not file_str:
            continue

        file_path = root / file_str
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Simple text search for symbol name
        if symbol_name in content:
            valid_count += 1.0

    return round(min(1.0, valid_count / len(anchors)), 2)
