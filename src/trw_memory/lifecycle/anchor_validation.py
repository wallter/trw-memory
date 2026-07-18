"""Anchor validity computation for code-grounded learnings.

Scores how many of a learning's code anchors still reference valid
symbols in the current codebase. Returns 1.0 when all anchors are
valid (or there are no anchors), 0.0 when all are invalid.

PRD-CORE-111 FR03/FR05. Pure function: reads the filesystem only, never
writes (see ``test_compute_anchor_validity_is_pure``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from trw_memory.models.memory import Anchor

# ---------------------------------------------------------------------------
# Inline comment marker regex (PRD-CORE-111 FR05).
#
# DUPLICATION NOTE: the canonical marker regex + ``extract_marker_ids`` live in
# trw-mcp (``trw_mcp.state.anchor_generation``). trw-memory deliberately keeps a
# small local copy rather than importing from trw-mcp: trw-memory is the
# lower-level, independently-published package — trw-mcp depends on it, never the
# reverse. Importing trw-mcp here would invert the package dependency and break
# trw-memory's standalone PyPI distribution. There is no shared package below
# trw-memory to relocate the helper into, so a one-line duplicated pattern is the
# correct trade-off.
# ---------------------------------------------------------------------------
_MARKER_PATTERN = re.compile(r"mcp\.trw\.recall\(id=([A-Za-z]-[a-zA-Z0-9]{4,8}(?:,[A-Za-z]-[a-zA-Z0-9]{4,8})*)\)")

# Marker presence bonus added to the raw valid_count (FR03/FR05).
_MARKER_BONUS = 0.5

# Directories never worth scanning for inline markers (keeps the FR05 scan
# bounded — see NFR01 latency budget).
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
        "target",
        ".next",
        ".turbo",
    }
)

# Only text/source/doc files can carry a marker; scanning these bounds the walk.
_MARKER_EXTS = frozenset({".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".md", ".mdx", ".yaml", ".yml", ".txt"})

_MARKER_MAX_BYTES = 1_000_000  # skip very large files


def _iter_marker_candidate_files(project_root: Path) -> Iterator[Path]:
    """Yield source/doc files under *project_root*, skipping heavy directories."""
    if not project_root.is_dir():
        return
    stack = [project_root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in _SKIP_DIRS:
                    stack.append(entry)
                continue
            if entry.suffix.lower() not in _MARKER_EXTS:
                continue
            try:
                if entry.stat().st_size > _MARKER_MAX_BYTES:
                    continue
            except OSError:
                continue
            yield entry


def _marker_references_id(project_root: Path, learning_id: str) -> bool:
    """Return True if an ``mcp.trw.recall(id=...)`` marker names *learning_id*.

    Scoped to *project_root* (FR05 security bound). Pure read-only scan.
    """
    if not learning_id:
        return False
    for path in _iter_marker_candidate_files(project_root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # Cheap substring pre-filter before running the regex.
        if "mcp.trw.recall" not in text or learning_id not in text:
            continue
        for match in _MARKER_PATTERN.finditer(text):
            ids = [i.strip() for i in match.group(1).split(",")]
            if learning_id in ids:
                return True
    return False


def compute_anchor_validity(
    anchors: list[Anchor] | list[dict[str, object]],
    project_root: str | Path,
    *,
    learning_id: str = "",
) -> float:
    """Compute what fraction of anchors still point to valid symbols.

    For each anchor, checks:
    1. File exists at anchor.file (or anchor["file"]) relative to project_root
    2. Symbol name appears in the file content (simple text search)

    Bonus signal (FR05): when *learning_id* is supplied and an inline comment
    marker ``mcp.trw.recall(id=<learning_id>)`` is found anywhere under
    *project_root*, 0.5 is added to the valid count (still capped at 1.0).

    Accepts both ``list[Anchor]`` (Pydantic models) and ``list[dict]``
    (raw dicts from YAML/JSON) for backward compatibility.

    Args:
        anchors: List of Anchor models or dicts with "file" and "symbol_name" keys.
        project_root: Absolute path to the project root.
        learning_id: The learning's id, used to look for inline markers. When
            empty (default), the marker bonus is skipped — preserving the
            historical behaviour for callers that do not pass an id.

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

    # FR05 bonus: an inline marker referencing this learning boosts validity.
    if learning_id and _marker_references_id(root, learning_id):
        valid_count += _MARKER_BONUS

    return round(min(1.0, valid_count / len(anchors)), 2)
