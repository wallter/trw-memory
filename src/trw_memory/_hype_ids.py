"""Shared parsing for backward-compatible synthetic HyPE vector IDs."""

from __future__ import annotations

HYPE_ID_SEPARATOR = "#hype"


def hype_sibling_id(parent_id: str, index: int) -> str:
    """Build the synthetic sibling id ``{parent_id}#hype{index}`` (0-based)."""
    return f"{parent_id}{HYPE_ID_SEPARATOR}{index}"


def hype_parent_id(entry_id: str) -> str | None:
    """Return the parent for a final ASCII-numeric HyPE suffix, else ``None``."""
    parent_id, separator, suffix = entry_id.rpartition(HYPE_ID_SEPARATOR)
    if separator and parent_id and suffix.isascii() and suffix.isdigit():
        return parent_id
    return None


def is_hype_id(entry_id: str) -> bool:
    """Return whether *entry_id* has the generated HyPE sibling grammar."""
    return hype_parent_id(entry_id) is not None


def parent_of_hype_id(entry_id: str) -> str:
    """Map a generated sibling to its full parent; pass canonical ids through."""
    return hype_parent_id(entry_id) or entry_id
