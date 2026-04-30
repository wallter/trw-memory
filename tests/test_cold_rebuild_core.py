"""Core rebuild and hydrator-mapping tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.storage._cold_rebuild import _normalize_ts, rebuild_from_cold
from trw_memory.storage.sqlite_backend import _resolve_cold_rebuild_base

from ._test_cold_rebuild_support import _configure_structlog, _make_yaml, _open_fresh_db


def test_config_knob_default_is_true() -> None:
    """FR04: memory_recovery_rebuild_from_cold defaults to True."""
    assert MemoryConfig().memory_recovery_rebuild_from_cold is True


def test_config_knob_constructor_override() -> None:
    """FR04: constructor override disables the knob."""
    cfg = MemoryConfig(memory_recovery_rebuild_from_cold=False)
    assert cfg.memory_recovery_rebuild_from_cold is False


def test_config_knob_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR04: MEMORY_RECOVERY_REBUILD_FROM_COLD env var disables the knob."""
    monkeypatch.setenv("MEMORY_RECOVERY_REBUILD_FROM_COLD", "false")
    cfg = MemoryConfig()
    assert cfg.memory_recovery_rebuild_from_cold is False


def test_rebuild_from_cold_basic_insert_count(tmp_path: Path) -> None:
    """FR01: rebuild returns count of rows inserted; SELECT count matches."""
    for i in range(5):
        _make_yaml(tmp_path, f"L-000{i}")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 5
        row_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert row_count == 5
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        assert integrity[0] == "ok"
    finally:
        conn.close()


def test_rebuild_from_cold_missing_cold_dir_returns_zero(tmp_path: Path) -> None:
    """Migration: first-run host with no cold tier returns 0 without error."""
    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        assert rebuild_from_cold(tmp_path, conn) == 0
    finally:
        conn.close()


def test_rebuild_from_cold_empty_cold_dir_returns_zero(tmp_path: Path) -> None:
    """Migration: cold dir exists but empty returns 0 without error."""
    (tmp_path / "memory" / "cold").mkdir(parents=True)
    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        assert rebuild_from_cold(tmp_path, conn) == 0
    finally:
        conn.close()


def test_hydrator_hardcodes_type_pattern(tmp_path: Path) -> None:
    """FR05: DB type='pattern' even when YAML source_type='agent'."""
    _make_yaml(tmp_path, "L-TYPE", source_type="agent")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuild_from_cold(tmp_path, conn)
        row = conn.execute("SELECT type, source FROM memories WHERE id='L-TYPE'").fetchone()
        assert row is not None
        assert row[0] == "pattern"
        assert row[1] == "agent"
    finally:
        conn.close()


def test_hydrator_source_type_consolidated(tmp_path: Path) -> None:
    """Regression for the exact ``source_type='consolidated'`` recovery bug."""
    _make_yaml(tmp_path, "L-CONS", source_type="consolidated")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 1, "consolidated entry must NOT be silently dropped"
        row = conn.execute("SELECT type, source FROM memories WHERE id='L-CONS'").fetchone()
        assert row is not None
        assert row[0] == "pattern"
        assert row[1] == "consolidated"
    finally:
        conn.close()


def test_hydrator_covers_all_entry_columns() -> None:
    """Every emitted column must still exist in the canonical shared tuple."""
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS
    from trw_memory.storage._shared import ENTRY_COLUMNS

    missing = set(_INSERT_COLUMNS) - set(ENTRY_COLUMNS)
    assert not missing, (
        f"_INSERT_COLUMNS drift — {sorted(missing)} not in ENTRY_COLUMNS. "
        "A column was removed or renamed; update _cold_rebuild.py or _shared.ENTRY_COLUMNS."
    )


@pytest.mark.slow
def test_rebuild_throughput_10k_files(tmp_path: Path) -> None:
    """NFR01: rebuild must process 10,000 cold YAML files in under 30 seconds."""
    cold_dir = tmp_path / "memory" / "cold" / "2026" / "04"
    cold_dir.mkdir(parents=True)
    for i in range(10_000):
        entry_id = f"L-{i:05x}"
        yaml_text = (
            f"id: {entry_id}\n"
            f"summary: entry {i}\n"
            "detail: ''\n"
            "impact: 0.5\n"
            "status: active\n"
            "recurrence: 1\n"
            "namespace: default\n"
            "created: '2026-04-12T00:00:00+00:00'\n"
            "updated: '2026-04-12T00:00:00+00:00'\n"
            "source_type: agent\n"
        )
        (cold_dir / f"{entry_id}.yaml").write_text(yaml_text)

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        started = time.monotonic()
        rebuilt = rebuild_from_cold(tmp_path, conn)
        elapsed = time.monotonic() - started
    finally:
        conn.close()

    assert rebuilt == 10_000, f"expected 10,000 rebuilt, got {rebuilt}"
    assert elapsed < 30.0, (
        f"NFR01 throughput SLO violated: 10k YAMLs took {elapsed:.2f}s, budget 30s. "
        "Check for quadratic work or missing transaction batching."
    )


def test_hydrator_normalizes_date_only(tmp_path: Path) -> None:
    """FR05: YAML date-only created/updated is normalised to full ISO-8601."""
    _make_yaml(tmp_path, "L-DATE", created="2026-04-12", updated="2026-04-12")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuild_from_cold(tmp_path, conn)
        row = conn.execute("SELECT created_at, updated_at FROM memories WHERE id='L-DATE'").fetchone()
        assert row is not None
        assert row[0] == "2026-04-12T00:00:00+00:00"
        assert row[1] == "2026-04-12T00:00:00+00:00"
    finally:
        conn.close()


def test_hydrator_json_dumps_list_fields(tmp_path: Path) -> None:
    """FR05: list fields are JSON-serialised in the DB."""
    _make_yaml(tmp_path, "L-LIST", tags=["a", "b", "c"])

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuild_from_cold(tmp_path, conn)
        row = conn.execute("SELECT tags FROM memories WHERE id='L-LIST'").fetchone()
        assert row is not None
        assert json.loads(row[0]) == ["a", "b", "c"]
    finally:
        conn.close()


def test_hydrator_preserves_obsolete_status(tmp_path: Path) -> None:
    """US-003 AC2: status='obsolete' is preserved verbatim, not blanket-reset."""
    _make_yaml(tmp_path, "L-OBS", status="obsolete")
    _make_yaml(tmp_path, "L-RES", status="resolved")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuild_from_cold(tmp_path, conn)
        rows = dict(conn.execute("SELECT id, status FROM memories").fetchall())
        assert rows["L-OBS"] == "obsolete"
        assert rows["L-RES"] == "resolved"
    finally:
        conn.close()


def test_normalize_ts_date_only() -> None:
    assert _normalize_ts("2026-04-12") == "2026-04-12T00:00:00+00:00"


def test_normalize_ts_full_iso_passthrough() -> None:
    assert _normalize_ts("2026-04-12T15:30:00+00:00") == "2026-04-12T15:30:00+00:00"


def test_normalize_ts_none() -> None:
    assert _normalize_ts(None) is None


def test_resolve_cold_rebuild_base_standalone_layout(tmp_path: Path) -> None:
    """FR01: standalone ``<base>/memory.db`` resolves to ``<base>``."""
    _make_yaml(tmp_path, "L-BASE-STANDALONE")
    assert _resolve_cold_rebuild_base(tmp_path / "memory.db") == tmp_path


def test_resolve_cold_rebuild_base_trw_mcp_layout(tmp_path: Path) -> None:
    """FR01: ``<trw_dir>/memory/memory.db`` resolves to ``<trw_dir>``."""
    trw_dir = tmp_path / ".trw"
    _make_yaml(trw_dir, "L-BASE-MCP")
    assert _resolve_cold_rebuild_base(trw_dir / "memory" / "memory.db") == trw_dir


def test_resolve_cold_rebuild_base_selects_largest_non_empty_candidate(tmp_path: Path) -> None:
    """FR01: when both candidates exist, choose the one with more cold YAMLs."""
    db_path = tmp_path / "memory" / "memory.db"
    _make_yaml(tmp_path, "L-BASE-TRW-ONE")
    _make_yaml(tmp_path / "memory", "L-BASE-STANDALONE-ONE")
    _make_yaml(tmp_path / "memory", "L-BASE-STANDALONE-TWO")

    assert _resolve_cold_rebuild_base(db_path) == tmp_path / "memory"
