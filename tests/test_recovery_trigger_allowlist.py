"""Only a damaged file is ever quarantined; a failing machine never is (L-8QV8 class).

``SQLiteBackend`` moves a store to ``.corrupt.<ts>.bak`` and unlinks its WAL when
an open fails. That is right for a damaged file and wrong for everything else an
open can raise: out of descriptors, disk full, read-only, ``locking protocol``.
Each of those used to quarantine a healthy store -- in the daemon, under its own
live connections, whose next commits then went into the moved file.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.storage import _connection
from trw_memory.storage._connection import IntegrityCheckFailed, classify_open_error
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _forget_verified_stores() -> None:
    """Test seam mirroring the removed ``_connection.forget_verified_stores``."""
    with _connection._VERIFIED_LOCK:
        _connection._VERIFIED_STORES.clear()


def _healthy_store(root: Path) -> tuple[Path, bytes]:
    db = root / "memory.db"
    SQLiteBackend(db).close()
    _forget_verified_stores()
    return db, db.read_bytes()


def _sqlite_error(message: str, code: int) -> sqlite3.OperationalError:
    exc = sqlite3.OperationalError(message)
    exc.sqlite_errorcode = code  # type: ignore[attr-defined]
    return exc


def _untouched(db: Path, before: bytes) -> None:
    assert db.read_bytes() == before
    assert sorted(p.name for p in db.parent.iterdir() if ".corrupt." in p.name) == []


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("unable to open database file", 14),  # SQLITE_CANTOPEN: e.g. out of descriptors
        ("locking protocol", 15),  # SQLITE_PROTOCOL: a transient WAL race
        ("database or disk is full", 13),  # SQLITE_FULL
        ("attempt to write a readonly database", 8),  # SQLITE_READONLY
        ("database schema has changed", 17),  # SQLITE_SCHEMA
    ],
)
def test_a_machine_failure_on_open_leaves_a_healthy_store_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str, code: int
) -> None:
    db, before = _healthy_store(tmp_path)

    def _fail(conn: object, **_: object) -> None:
        raise _sqlite_error(message, code)

    monkeypatch.setattr(_connection, "apply_open_pragmas", _fail)
    with pytest.raises(sqlite3.OperationalError, match=message):
        SQLiteBackend(db)
    _untouched(db, before)


@pytest.mark.skipif(os.name != "posix", reason="descriptor exhaustion via a POSIX fd hog")
def test_running_out_of_descriptors_does_not_quarantine_the_store(tmp_path: Path) -> None:
    """End to end in a child: no injected error, the real EMFILE path."""
    db, before = _healthy_store(tmp_path)
    child = """
import os, resource, sys
from pathlib import Path
from trw_memory.storage.sqlite_backend import SQLiteBackend
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, 256), hard))  # bounded exhaustion on any host
hog = []
try:
    while True:
        hog.append(os.open(os.devnull, os.O_RDONLY))
except OSError:
    pass
os.close(hog.pop())  # one descriptor free: the store opens, its WAL cannot
try:
    SQLiteBackend(Path(sys.argv[1]))
    outcome = "opened"
except BaseException as exc:
    outcome = type(exc).__name__
for fd in hog:
    os.close(fd)
print(outcome)
"""
    out = subprocess.run(
        [sys.executable, "-c", child, str(db)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    assert out.stdout.strip() != "opened"
    _untouched(db, before)


def test_a_file_that_is_not_a_database_is_still_quarantined(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    db.write_bytes(b"this is not a sqlite database" * 200)
    with pytest.raises(CorruptDatabaseUnsalvageableError):  # strict: nothing salvaged, nothing invented
        SQLiteBackend(db)
    assert [p.name for p in tmp_path.iterdir() if ".corrupt." in p.name]


def test_only_damage_counts_as_corruption() -> None:
    corrupt = [
        IntegrityCheckFailed("database disk image is malformed (quick_check failed twice)"),
        _sqlite_error("database disk image is malformed", 11),
        _sqlite_error("vtable constructor failed", 11 | (1 << 8)),  # SQLITE_CORRUPT_VTAB
        _sqlite_error("file is not a database", 26),
        sqlite3.DatabaseError("database disk image is malformed"),  # no result code: judged by message
    ]
    other = [
        _sqlite_error("unable to open database file", 14),
        _sqlite_error("locking protocol", 15),
        _sqlite_error("database disk image is malformed", 13),  # the code decides, not the words
        sqlite3.DatabaseError("disk quota exceeded"),
        sqlite3.DatabaseError("the database disk image is malformed, maybe"),  # message fallback is a prefix
    ]
    assert [classify_open_error(exc) for exc in corrupt] == ["corrupt"] * len(corrupt)
    assert [classify_open_error(exc) for exc in other] == ["other"] * len(other)
    assert classify_open_error(_sqlite_error("disk I/O error", 10 | (3 << 8))) == "io"
    assert classify_open_error(sqlite3.OperationalError("database is locked")) == "lock"
    # The result code outranks a quoted message: damage never opens unchecked as "lock".
    assert classify_open_error(_sqlite_error("vtable constructor failed: database is locked", 11)) == "corrupt"
