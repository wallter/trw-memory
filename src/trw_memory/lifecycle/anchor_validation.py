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

PRD-SEC-016 round-2 finding 2: :class:`~trw_memory.models.memory.Anchor`
validates ``file`` (no absolute path, no ``..``) when data arrives as an
``Anchor`` instance, but this function ALSO accepts raw ``dict`` anchors --
the exact shape ``lifecycle.verification_pass._reverify_anchors`` reads
straight off a stored learning, never through ``Anchor``'s own validators.
``memory_verify``/``memory_maintain`` call into that re-verification path
over the transport, so a stored anchor with ``file="/etc/passwd"`` (which
``root / file_str`` in the old implementation would happily resolve to --
``Path.__truediv__`` DISCARDS the left operand when the right one is
absolute) turned this into a same-symbol-substring read oracle over any file
the daemon user can read, no race required. Every anchor's ``file`` is now
opened through :func:`trw_memory.tools.entry.open_checkout_file_fd`, the
PRD's one checkout-bound opener -- the same containment and no-follow
descriptor walk FR02/FR03 already give every other checkout-scoped read.
"""

from __future__ import annotations

import os
from pathlib import Path

from trw_memory._live_stores import close_reader_fd
from trw_memory.models.memory import Anchor


def compute_anchor_validity(
    anchors: list[Anchor] | list[dict[str, object]],
    project_root: str | Path,
) -> float:
    """Compute what fraction of anchors still point to valid symbols.

    For each anchor, checks:
    1. File exists at anchor.file (or anchor["file"]) relative to project_root,
       and is reachable WITHOUT following a symlink out of project_root or an
       absolute/``..`` escape (PRD-SEC-016 round-2 finding 2)
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

    root = str(project_root)
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

        content = _read_anchor_file(root, file_str)
        if content is None:
            continue

        # Simple text search for symbol name
        if symbol_name in content:
            valid_count += 1.0

    return round(min(1.0, valid_count / len(anchors)), 2)


def _read_anchor_file(root: str, file_str: str) -> str | None:
    """*file_str*'s text, opened through the checkout-bound opener; ``None`` if it could not be read that way.

    Delegates ALL containment enforcement (absolute-path escape, ``..``
    traversal, a symlinked component anywhere in the path) to
    :func:`trw_memory.tools.entry.open_checkout_file_fd` rather than
    re-implementing a second check here -- the same rule this module's
    docstring names as the reason this function exists at all.
    """
    from trw_memory.lifecycle.verification import MAX_FILE_SIZE_BYTES
    from trw_memory.tools.entry import open_checkout_file_fd

    opened = open_checkout_file_fd(root, file_str, "anchor_validation")
    if isinstance(opened, dict):
        return None
    try:
        # closefd=False: the descriptor is closed through close_reader_fd, which
        # releases its read lease (C15).
        with os.fdopen(opened, "rb", closefd=False) as handle:
            # The grep cap (C12 rc4): a multi-GB anchor file must not exhaust the shared daemon; over it, not valid.
            content = handle.read(MAX_FILE_SIZE_BYTES + 1)
        return None if len(content) > MAX_FILE_SIZE_BYTES else content.decode("utf-8", errors="replace")
    except OSError:  # trw-fail-silent-allow: the caller (compute_anchor_validity) treats a None read as "not valid," never as a pass -- unreadable is never counted toward the score
        return None
    finally:
        close_reader_fd(opened)
