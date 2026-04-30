"""Recovery-path tests for cold rebuild integration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_cold_rebuild_support import (
    _configure_structlog,
    _corrupt_sqlite_master,
    _make_yaml,
    _populate_real_db,
)


def test_recover_db_invokes_rebuild_when_gated(tmp_path: Path) -> None:
    """FR03 happy path: destroyed DB + strict + knob on + 3 cold YAMLs → 3 rows."""
    db_path = tmp_path / "memory.db"
    _populate_real_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)
    for i in range(3):
        _make_yaml(tmp_path, f"L-COLD{i}")

    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict", rebuild_from_cold=True)
    try:
        row_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert row_count == 3
        ids = {row[0] for row in conn.execute("SELECT id FROM memories").fetchall()}
        assert ids == {"L-COLD0", "L-COLD1", "L-COLD2"}
    finally:
        conn.close()


def test_recover_db_invokes_rebuild_for_trw_mcp_layout(tmp_path: Path) -> None:
    """Regression: ``.trw/memory/memory.db`` must rebuild from ``.trw/memory/cold``."""
    trw_dir = tmp_path / ".trw"
    db_path = trw_dir / "memory" / "memory.db"
    _populate_real_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)
    for i in range(3):
        _make_yaml(trw_dir, f"L-MCP-COLD{i}")

    conn = SQLiteBackend.recover_db(db_path, recovery_policy="strict", rebuild_from_cold=True)
    try:
        row_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert row_count == 3
        ids = {row[0] for row in conn.execute("SELECT id FROM memories").fetchall()}
        assert ids == {"L-MCP-COLD0", "L-MCP-COLD1", "L-MCP-COLD2"}
    finally:
        conn.close()


def test_recover_db_raises_when_rebuild_yields_zero(tmp_path: Path) -> None:
    """FR03: rebuild with empty cold tier still raises CorruptDatabaseUnsalvageableError."""
    db_path = tmp_path / "memory.db"
    _populate_real_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict", rebuild_from_cold=True)


def test_recover_db_knob_off_does_not_rebuild(tmp_path: Path) -> None:
    """FR03 regression: knob=False → no rebuild, strict still raises even with cold YAMLs."""
    db_path = tmp_path / "memory.db"
    _populate_real_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)
    for i in range(3):
        _make_yaml(tmp_path, f"L-IGN{i}")

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(db_path, recovery_policy="strict", rebuild_from_cold=False)


def test_recover_db_non_strict_policy_does_not_rebuild(tmp_path: Path) -> None:
    """FR03 regression: empty_ok policy → no rebuild even with cold YAMLs."""
    db_path = tmp_path / "memory.db"
    _populate_real_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)
    for i in range(3):
        _make_yaml(tmp_path, f"L-EMPTY{i}")

    conn = SQLiteBackend.recover_db(db_path, recovery_policy="empty_ok", rebuild_from_cold=True)
    try:
        row_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert row_count == 0
    finally:
        conn.close()


def test_healthy_open_does_not_invoke_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR03 regression: healthy DB open → rebuild_from_cold NEVER invoked."""
    for i in range(3):
        _make_yaml(tmp_path, f"L-COLD{i}")

    import trw_memory.storage._cold_rebuild as cold_rebuild_module

    call_counter = {"n": 0}
    original = cold_rebuild_module.rebuild_from_cold

    def spy(base_dir: Path, new_conn: sqlite3.Connection) -> int:
        call_counter["n"] += 1
        return original(base_dir, new_conn)

    monkeypatch.setattr(cold_rebuild_module, "rebuild_from_cold", spy)

    backend = SQLiteBackend(tmp_path / "memory.db", rebuild_from_cold=True)
    backend.close()

    assert call_counter["n"] == 0
