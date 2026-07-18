"""SQLite statement-shaping helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

_T = TypeVar("_T")

SQLITE_SAFE_BIND_LIMIT = 900


def iter_bind_chunks(
    values: Sequence[_T], *, bindings_per_item: int = 1, reserved_bindings: int = 0
) -> Iterator[Sequence[_T]]:
    """Yield slices that stay below the conservative SQLite bind ceiling."""
    available = SQLITE_SAFE_BIND_LIMIT - reserved_bindings
    if bindings_per_item <= 0 or reserved_bindings < 0 or available < bindings_per_item:
        raise ValueError("invalid SQLite bind allocation")
    chunk_size = available // bindings_per_item
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]
