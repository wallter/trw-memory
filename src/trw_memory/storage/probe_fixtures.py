"""Store files ``probe_store`` must classify, one builder per case (PRD-QUAL-147's contract fixtures).

Test support, imported by no runtime module. It lives in the package, not under ``tests/``, so that
trw-mcp's tests build exactly the files trw-memory's contract test pins: trw-mcp's own ``tests``
package cannot import trw-memory's (both are named ``tests``, and the public trw-mcp repository has
no trw-memory tests). Each builder writes *path* and returns it; a trw-memory store itself is any
``SQLiteBackend`` the caller fills (``legacy_schema`` turns one into a pre-migration store).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from trw_memory.storage._connection import connect

#: The evidence repro: 20 rows whose ``metadata`` is a generated 200,000,000-character value, in 8 KB.
BOMB_WIDTH = 200_000_000
_LEGACY = (
    "ALTER TABLE memories DROP COLUMN verification_checked_at; ALTER TABLE memories DROP COLUMN protection_tier;"
    " PRAGMA user_version = 0"
)


def _sql(path: Path, script: str) -> Path:
    with closing(connect(path, dbapi=sqlite3, timeout=5.0, check_same_thread=True)) as conn:
        conn.executescript(script)
    return path


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def empty_file(path: Path) -> Path:
    return _write(path, b"")


def not_a_database(path: Path) -> Path:
    return _write(path, b"not a sqlite database\n" * 64)


def foreign_db(path: Path) -> Path:
    return _sql(path, "CREATE TABLE notes (body TEXT); CREATE TABLE tags (name TEXT);")


def generated_column_bomb(path: Path, width: int = BOMB_WIDTH) -> Path:
    """``metadata`` as in the evidence, and ``id``, which every read keyed on a row's id evaluates."""
    bomb = f"GENERATED ALWAYS AS (printf('%.*c', {int(width)}, 'x')) VIRTUAL"
    columns = f"content TEXT, namespace TEXT DEFAULT 'default', id TEXT {bomb}, metadata TEXT {bomb}"
    values = ", ".join(f"('row {i}')" for i in range(20))
    return _sql(path, f"CREATE TABLE memories ({columns}); INSERT INTO memories (content) VALUES {values};")  # noqa: S608 -- constants


def legacy_schema(path: Path) -> Path:
    """Drop ``verification_checked_at`` and ``protection_tier`` from an existing trw-memory store at *path*
    and stamp it ``user_version`` 0, the version before the bootstrap backfill that adds both: a store
    written before those migrations, which ``SQLiteBackend`` still upgrades on open."""
    return _sql(path, _LEGACY)
