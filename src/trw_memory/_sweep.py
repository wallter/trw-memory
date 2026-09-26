"""Bounded work over a keyset-paged row source, resumable by cursor (rc9 sweep B2).

A shared daemon serves every tenant from one process, so a call that walks a whole
namespace (a re-embed, a verification pass) must stop at a daemon-owned budget and
hand back where it stopped. This is that loop, for any row source: the caller
supplies how to fetch a page after a key and how to key a row; :func:`sweep`
visits pages until the budget is spent and returns the key to resume after.

Every page advances the sweep by at least one row, whatever the clock says (see
:func:`sweep`), so a row slower than the whole budget cannot wedge it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

__all__ = ["decode_token", "encode_token", "sweep"]

T = TypeVar("T")
K = TypeVar("K")

#: A cursor token is caller-supplied on the way back in: bound what gets parsed. Entry ids carry
#: no length cap of their own, so this leaves room for any sane one.
MAX_TOKEN_CHARS = 4096


def sweep(
    fetch: Callable[[K | None, int], Sequence[T]],
    key: Callable[[T], K],
    visit: Callable[[Sequence[T], float], int],
    *,
    after: K | None,
    page: int,
    rows: int,
    seconds: float,
) -> K | None:
    """Visit rows after *after*, a page at a time, until *rows* rows or *seconds* are spent.

    *visit* gets each page and the monotonic deadline and returns how many of the page's rows
    it consumed, in order. It checks the deadline after each row, never before the first, so
    every page advances the sweep (a row slower than the whole budget would otherwise stop it
    there forever). *fetch* returns fewer rows than asked only at the end of its source. Returns
    the key of the last consumed row to resume after, or ``None`` once the source is exhausted.
    """
    if page < 1 or rows < 1:
        raise ValueError(f"a sweep needs a positive page and row budget, got page={page} rows={rows}")
    deadline = time.monotonic() + seconds
    visited = 0
    cursor = after
    while visited < rows:
        limit = min(page, rows - visited)
        batch = fetch(cursor, limit)
        if not batch:
            return None
        if len(batch) > limit:
            raise ValueError(f"fetch returned {len(batch)} rows for a limit of {limit}")
        consumed = visit(batch, deadline)
        if not 1 <= consumed <= len(batch):
            raise ValueError(f"visit consumed {consumed} of {len(batch)} rows; it must consume at least the first")
        cursor = key(batch[consumed - 1])
        visited += consumed
        if consumed == len(batch) < limit:
            return None  # a short page is the source's last
        if consumed < len(batch) or time.monotonic() >= deadline:
            break
    return cursor


def encode_token(parts: Sequence[str]) -> str:
    """An opaque cursor token for a key made of strings; ``ValueError`` if :func:`decode_token` would refuse it."""
    if not all(isinstance(part, str) for part in parts):
        raise ValueError("a cursor key is made of strings")
    token = json.dumps(list(parts), separators=(",", ":"))
    if len(token) > MAX_TOKEN_CHARS:
        raise ValueError(f"cursor longer than {MAX_TOKEN_CHARS} characters")
    return token


def decode_token(token: str, arity: int) -> list[str]:
    """The key parts of *token*; ``ValueError`` for anything :func:`encode_token` did not produce."""
    if len(token) > MAX_TOKEN_CHARS:
        raise ValueError(f"cursor longer than {MAX_TOKEN_CHARS} characters")
    parts = json.loads(token)
    if not isinstance(parts, list) or len(parts) != arity or not all(isinstance(part, str) for part in parts):
        raise ValueError("malformed cursor")
    return parts
