"""CLI coverage for cold-tier restore."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ._test_cold_rebuild_support import _configure_structlog, _make_yaml


def test_cli_restore_from_cold_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FR02: trw-memory restore --from-cold exits 0 and prints summary."""
    storage_path = tmp_path / "mem"
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(storage_path))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    namespace_dir = storage_path / "default"
    namespace_dir.mkdir(parents=True)
    for i in range(4):
        _make_yaml(namespace_dir, f"L-CLI{i}")

    from trw_memory.cli import main

    rc = main(["restore", "--from-cold", "--namespace=default"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Rebuilt 4 entries from cold tier (0 skipped)" in out

    db_path = namespace_dir / "memory.db"
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 4
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        assert integrity[0] == "ok"
    finally:
        conn.close()


def test_cli_restore_reports_skipped_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FR02: CLI reports skipped count when hydration fails on some YAMLs."""
    storage_path = tmp_path / "mem"
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(storage_path))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    namespace_dir = storage_path / "default"
    namespace_dir.mkdir(parents=True)
    _make_yaml(namespace_dir, "L-OK")
    bad = namespace_dir / "memory" / "cold" / "2026" / "04" / "bad.yaml"
    bad.write_text(
        "summary: bad\ncreated: 2026-04-12\nupdated: 2026-04-12\n",
        encoding="utf-8",
    )

    from trw_memory.cli import main

    rc = main(["restore", "--from-cold"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Rebuilt 1 entries from cold tier (1 skipped)" in out
