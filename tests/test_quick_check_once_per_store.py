"""PRD-CORE-298 FR05 -- the recall path runs ``PRAGMA quick_check`` once per process per store.

The check reads the whole file. Run on every open it cost about 320 ms at
20,000 rows, twice per daemon recall, and was most of that recall's latency.
A ``check_once`` open (the daemon's recall path) checks a healthy store on its
first open only; a corrupt one still fails loudly; a file replaced at the same
path is checked again. Every other open, a writer's included, still checks
every time, so a store corrupted in place after verification fails closed there.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.storage import _connection
from trw_memory.storage._connection import forget_verified_stores, open_and_configure


@pytest.fixture(autouse=True)
def _fresh_record() -> Iterator[None]:
    forget_verified_stores()
    yield
    forget_verified_stores()


@pytest.fixture
def checks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every ``quick_check`` statement any connection opened here executes."""
    seen: list[str] = []
    real = _connection.connect

    def traced(*args: object, **kwargs: object) -> object:
        conn = real(*args, **kwargs)  # type: ignore[arg-type]
        conn.set_trace_callback(lambda sql: seen.append(sql) if "quick_check" in sql else None)
        return conn

    monkeypatch.setattr(_connection, "connect", traced)
    return seen


def _store(path: Path, rows: int = 200) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, body TEXT)")
    conn.execute("CREATE INDEX t_body ON t (body)")
    conn.executemany("INSERT INTO t (body) VALUES (?)", [(f"row {i} " * 20,) for i in range(rows)])
    conn.commit()
    conn.close()
    return path


def test_a_healthy_store_is_checked_on_its_first_open_only(tmp_path: Path, checks: list[str]) -> None:
    db = _store(tmp_path / "m.db")

    for _ in range(5):
        open_and_configure(db, check_once=True).close()

    assert len(checks) == 1


def test_a_corrupt_store_fails_on_first_open_and_every_open_after(tmp_path: Path, checks: list[str]) -> None:
    db = _store(tmp_path / "m.db")
    size = db.stat().st_size
    with db.open("r+b") as handle:  # overwrite interior pages, keep the header readable
        handle.seek(4096)
        handle.write(os.urandom(size - 8192))

    for _ in range(2):
        with pytest.raises(sqlite3.DatabaseError):
            open_and_configure(db, check_once=True).close()
    assert len(checks) >= 2, "a failed check is never recorded, so it runs again"


def test_a_store_replaced_at_the_same_path_is_checked_again(tmp_path: Path, checks: list[str]) -> None:
    db = _store(tmp_path / "m.db")
    open_and_configure(db, check_once=True).close()

    replacement = _store(tmp_path / "new.db")
    os.replace(replacement, db)  # a new inode behind the same path
    open_and_configure(db, check_once=True).close()
    open_and_configure(db, check_once=True).close()

    assert len(checks) == 2


def _corrupt_in_place(db: Path) -> None:
    size = db.stat().st_size
    with db.open("r+b") as handle:  # same inode: interior pages overwritten, header kept
        handle.seek(4096)
        handle.write(os.urandom(size - 8192))


def test_a_default_open_checks_every_time_and_fails_closed_on_in_place_corruption(
    tmp_path: Path, checks: list[str]
) -> None:
    """Codex P1 on 86c73275c: a writer must never skip the check."""
    db = _store(tmp_path / "m.db")
    open_and_configure(db, check_once=True).close()  # verified by the recall path
    _corrupt_in_place(db)

    open_and_configure(db, check_once=True).close()  # the recall path skips, as documented
    with pytest.raises(sqlite3.DatabaseError):
        open_and_configure(db).close()  # a writer's open checks and fails closed
    assert len(checks) >= 2
