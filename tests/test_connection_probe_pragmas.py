"""The integrity/row-count probes carry the same PRAGMA profile as every open.

``check_integrity`` and ``db_has_data`` used to call ``connect`` directly and
skip ``apply_open_pragmas``, so the 64 MiB ``journal_size_limit`` that caps WAL
growth on every other open path was simply unset on those connections. The cap
exists because an unbounded WAL widens the window for the WAL-reset corruption
class this store has already suffered once, so adding a callsite must not be a
way to opt out of it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from trw_memory.storage import _connection
from trw_memory.storage._connection import (
    WAL_JOURNAL_SIZE_LIMIT_BYTES,
    check_integrity,
    db_has_data,
    open_probe,
)

_EXPECTED_PRAGMA = f"PRAGMA journal_size_limit = {WAL_JOURNAL_SIZE_LIMIT_BYTES}"


def _seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "probe.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO memories VALUES ('a', 'x')")
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def traced_opens(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the statements issued on every connection the module opens.

    Wraps the real ``connect`` — the probes get a real connection, so what is
    observed is the production open sequence, not a stand-in for it.
    """
    per_connection: list[list[str]] = []
    real_connect = _connection.connect

    def _connect(*args: Any, **kwargs: Any) -> Any:
        conn = real_connect(*args, **kwargs)
        statements: list[str] = []
        per_connection.append(statements)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(_connection, "connect", _connect)
    return per_connection


def test_open_probe_sets_the_wal_journal_size_limit(tmp_path: Path) -> None:
    conn = open_probe(_seeded_db(tmp_path))
    try:
        limit = conn.execute("PRAGMA journal_size_limit").fetchone()[0]
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()

    assert limit == WAL_JOURNAL_SIZE_LIMIT_BYTES
    assert journal_mode == "wal"


def test_check_integrity_connection_applies_the_wal_cap(tmp_path: Path, traced_opens: list[list[str]]) -> None:
    result = check_integrity(_seeded_db(tmp_path))

    assert result["ok"] is True
    assert len(traced_opens) == 1
    assert _EXPECTED_PRAGMA in traced_opens[0]


def test_db_has_data_connection_applies_the_wal_cap(tmp_path: Path, traced_opens: list[list[str]]) -> None:
    assert db_has_data(_seeded_db(tmp_path)) is True

    assert len(traced_opens) == 1
    assert _EXPECTED_PRAGMA in traced_opens[0]


def test_probes_still_report_their_own_answers(tmp_path: Path) -> None:
    """Routing through the shared open path did not change what they return."""
    empty = tmp_path / "empty.db"
    conn = sqlite3.connect(empty)
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    assert db_has_data(empty) is False
    assert check_integrity(empty)["ok"] is True

    # A path with no database behind it: still a clean False, not an exception.
    assert db_has_data(tmp_path / "absent.db") is False
