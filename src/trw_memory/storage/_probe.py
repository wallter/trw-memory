"""What is this store, and how many real rows does it hold? One bounded, read-only answer (PRD-QUAL-147).

For a ``memory.db`` trw-memory did not write -- a checkout can commit one. The file is registered
with ``_connection.untrusted_store`` (every statement under ``PROBE_DEADLINE_S`` and
``UNTRUSTED_LENGTH_LIMIT``) and refused by ``_untrusted_store._REFUSED`` before any read of
``memories``, so an 8 KB file whose ``metadata`` is a generated column cannot allocate gigabytes. No
``quick_check`` (it scans the whole file), no schema migration, no write: the open is
``connect(read_only=True)``. An at-rest read is ``immutable=1``, so the file is re-stat'ed after it and
read once more if a writer changed it meanwhile (B71-07).

A real row is one :func:`classify_canary` does not call the store's canary (PRD-CORE-309): a pinned id
with its pinned content and nothing else changed, never a ``system_canary`` flag. A legacy store's
missing columns read as their migration defaults (``_schema.MIGRATE_COLS``), as the backfill would
write them.

"""

from __future__ import annotations

import enum
import os
import sqlite3
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from trw_memory.exceptions import StorageError
from trw_memory.storage._connection import connect, untrusted_store
from trw_memory.storage._untrusted_store import _REFUSED

#: Seconds every statement of one probe may run before it is interrupted.
PROBE_DEADLINE_S = 2.0
_TABLES = "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"


class StoreState(str, enum.Enum):
    ABSENT = "absent"
    UNINITIALIZED = "uninitialized"
    NOT_TRW = "not_trw"
    REFUSED = "refused"
    UNREADABLE = "unreadable"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class StoreProbe:
    state: StoreState
    real_rows: int | None = None
    tables: tuple[str, ...] = ()
    detail: str = ""


def probe_store(db_path: Path, namespace: str | None = None) -> StoreProbe:
    """Classify *db_path* and, when ``READY``, count its real rows (in *namespace*, or all). Never raises
    for file content: what cannot be read is ``UNREADABLE`` with the driver's error as ``detail``."""
    try:
        path = Path(db_path).resolve()  # the store's own name: its lock and identity check follow no link
        for _attempt in range(2):
            if (before := _stamp(path)) is None:
                return StoreProbe(StoreState.ABSENT)
            if not stat.S_ISREG(before[0]):  # a FIFO would block the open; a device is no store
                return StoreProbe(StoreState.UNREADABLE, detail="not a regular file")
            result = _read(path, namespace)
            # A read beside a -wal is plain mode=ro, a consistent snapshot; an immutable one is re-checked.
            if before[-1] or _stamp(path) == before:
                return result
    except (OSError, RuntimeError) as exc:  # RuntimeError: a symlink loop, before Python 3.13
        return StoreProbe(StoreState.UNREADABLE, detail=f"{type(exc).__name__}: {exc}")
    return StoreProbe(StoreState.UNREADABLE, detail="store changed during read")


def _stamp(path: Path) -> tuple[int, int, int, int, bool] | None:
    try:
        st = os.stat(path)
    except FileNotFoundError:  # trw-fail-silent-allow: absent is a state, and a store gone mid-read changed
        return None
    return st.st_mode, st.st_ino, st.st_size, st.st_mtime_ns, os.path.exists(f"{path}-wal")


def _read(path: Path, namespace: str | None) -> StoreProbe:
    try:
        with untrusted_store(path, time.monotonic() + PROBE_DEADLINE_S):
            conn = connect(path, dbapi=sqlite3, timeout=PROBE_DEADLINE_S, check_same_thread=True, read_only=True)
            try:
                _harden(conn)
                if refused := [row[0] for row in conn.execute(_REFUSED)]:
                    return StoreProbe(StoreState.REFUSED, detail=", ".join(refused))
                names = [row[0] for row in conn.execute(_TABLES)]
                if "memories" in names:
                    return StoreProbe(StoreState.READY, _count(conn, namespace), tuple(names[:5]))
                state = StoreState.NOT_TRW if names else StoreState.UNINITIALIZED
                return StoreProbe(state, tables=tuple(names[:5]), detail="no memories table" if names else "")
            finally:
                conn.close()
    except (sqlite3.Error, StorageError, OSError, ValueError) as exc:
        return StoreProbe(StoreState.UNREADABLE, detail=f"{type(exc).__name__}: {exc}")


def _harden(conn: sqlite3.Connection) -> None:
    """Bound this connection's schema PARSE, which runs inside prepare where no progress handler does.

    Each limit sits well past trw-memory's own schema (its longest CREATE is 2,029 characters, its widest
    table 50 columns, its expressions a few levels deep; the probe's own queries fit the same bounds).
    A stored statement past one fails the schema load: ``DatabaseError``, so ``UNREADABLE``. Named here,
    not in ``untrusted_store``, so checkout import's reads are unchanged."""
    for limit, value in (
        (sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16_384),
        (sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100),
        (sqlite3.SQLITE_LIMIT_COLUMN, 128),
    ):
        conn.setlimit(limit, value)
    # No schema function may run; malformed cells fail early; a 2 MiB page cache, not 2,000 pages of up to 64 KiB.
    for pragma in ("trusted_schema = OFF", "cell_size_check = ON", "cache_size = -2048"):
        conn.execute(f"PRAGMA {pragma}")


def _count(conn: sqlite3.Connection, namespace: str | None) -> int:
    """Rows in ``memories`` (in *namespace*), minus the pinned canaries among them."""
    from trw_memory.security._runtime_canary import classify_canary
    from trw_memory.security.canary import PINNED_HASHES
    from trw_memory.storage._row_mapper import row_to_entry
    from trw_memory.storage._schema import MIGRATE_COLS
    from trw_memory.storage._shared import ENTRY_COLUMNS

    present = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    defaults = {name: definition.partition(" DEFAULT ")[2] or "NULL" for name, definition in MIGRATE_COLS}
    columns = ", ".join(c if c in present else f"{defaults.get(c, 'NULL')} AS {c}" for c in ENTRY_COLUMNS)
    where, params = ("namespace = ?", (namespace,)) if namespace is not None else ("1", ())
    marks = ",".join("?" * len(PINNED_HASHES))
    sql = f"SELECT {columns} FROM memories WHERE id IN ({marks}) AND {where}"  # noqa: S608
    pinned = conn.execute(sql, (*PINNED_HASHES, *params))
    total = int(conn.execute(f"SELECT COUNT(*) FROM memories WHERE {where}", params).fetchone()[0])  # noqa: S608

    def is_canary(row: tuple[object, ...]) -> bool:
        try:
            return classify_canary(row_to_entry(row)) == "canary"
        # Total over whatever the file holds: a field no entry can decode (bad JSON, a float overflow, nesting
        # past the recursion limit) makes the row something other than the seeded canary, so real data.
        except Exception:  # trw-fail-silent-allow: a pinned-id row that cannot decode is counted as a real row
            return False

    return total - sum(is_canary(tuple(row)) for row in pinned)
