"""Tests for PRD-CORE-140: cold-YAML-tier rebuild on recovery.

Covers all 7 FRs, 5 NFRs, and the regression matrix:
- FR01: rebuild_from_cold happy path + integrity
- FR02: restore --from-cold CLI
- FR03: gated auto-rebuild inside recover_db
- FR04: config knob default + override
- FR05: hydrator mapping (hardcoded type='pattern', source_type → source)
- FR06: per-file skip on malformed YAML with WARNING
- FR07: path-traversal guard
- Idempotency, status preservation, regression for healthy/non-strict/knob-off
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
from structlog.testing import capture_logs

from trw_memory.exceptions import CorruptDatabaseUnsalvageableError
from trw_memory.models.config import MemoryConfig
from trw_memory.storage._cold_rebuild import (
    _assert_within_cold_dir,
    _hydrate_yaml,
    _HydrationError,
    _normalize_ts,
    rebuild_from_cold,
)
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage.persistence import write_yaml
from trw_memory.storage.sqlite_backend import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _configure_structlog() -> Iterator[None]:
    """Ensure structlog routes through the testing capture."""
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def _make_yaml(base_dir: Path, entry_id: str, **overrides: Any) -> Path:
    """Write a cold-tier YAML under base_dir/memory/cold/2026/04/."""
    cold_dir = base_dir / "memory" / "cold" / "2026" / "04"
    cold_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "id": entry_id,
        "summary": f"summary for {entry_id}",
        "detail": f"detail for {entry_id}",
        "impact": 0.7,
        "status": "active",
        "recurrence": 1,
        "namespace": "default",
        "created": "2026-04-12T10:00:00+00:00",
        "updated": "2026-04-12T10:00:00+00:00",
        "tags": ["alpha", "beta"],
        "evidence": [],
        "source_type": "agent",
        "metadata": {},
        "vector_clock": {},
    }
    data.update(overrides)
    path = cold_dir / f"{entry_id}.yaml"
    write_yaml(path, data)
    return path


def _open_fresh_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    ensure_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# FR04: Config knob default and overrides
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FR01: Basic rebuild happy path
# ---------------------------------------------------------------------------


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
        # integrity_check remains ok on the rebuilt DB
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


# ---------------------------------------------------------------------------
# FR05: Hydrator mapping invariants
# ---------------------------------------------------------------------------


def test_hydrator_hardcodes_type_pattern(tmp_path: Path) -> None:
    """FR05: DB type='pattern' even when YAML source_type='agent'."""
    _make_yaml(tmp_path, "L-TYPE", source_type="agent")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuild_from_cold(tmp_path, conn)
        row = conn.execute("SELECT type, source FROM memories WHERE id='L-TYPE'").fetchone()
        assert row is not None
        assert row[0] == "pattern"  # hardcoded
        assert row[1] == "agent"  # provenance preserved on source column
    finally:
        conn.close()


def test_hydrator_source_type_consolidated(tmp_path: Path) -> None:
    """Audit finding 140-FR05-P2a — regression for the EXACT 2026-04-12 bug.

    During manual recovery, ``source_type='consolidated'`` entries were dropped
    because the broken hydrator tried to write ``consolidated`` into the DB
    ``type`` column (which is a ``MemoryType`` enum with no ``CONSOLIDATED``
    member). The corrected mapping routes ``source_type`` to the ``source``
    column and hardcodes ``type='pattern'``. This test uses the exact value
    that triggered the incident.
    """
    _make_yaml(tmp_path, "L-CONS", source_type="consolidated")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 1, "consolidated entry must NOT be silently dropped"
        row = conn.execute(
            "SELECT type, source FROM memories WHERE id='L-CONS'"
        ).fetchone()
        assert row is not None
        assert row[0] == "pattern", "type must be hardcoded, not copied from source_type"
        assert row[1] == "consolidated", "source_type value must land on the source column"
    finally:
        conn.close()


def test_hydrator_covers_all_entry_columns() -> None:
    """Audit finding 140-FR05-P2b — explicit PRD §8 test strategy entry.

    Every column the hydrator emits must be present in the canonical
    ``ENTRY_COLUMNS`` tuple. Guards against future column renames that
    would otherwise silently misalign the INSERT statement with the live
    schema. The ``_INSERT_COLUMN_SET`` runtime guard in ``_cold_rebuild.py``
    already fails at import time if this invariant breaks; this test
    upgrades that guarantee from "module-load smoke" to "CI regression".
    """
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS
    from trw_memory.storage._shared import ENTRY_COLUMNS

    emitted = set(_INSERT_COLUMNS)
    canonical = set(ENTRY_COLUMNS)
    missing = emitted - canonical
    assert not missing, (
        f"_INSERT_COLUMNS drift — {sorted(missing)} not in ENTRY_COLUMNS. "
        "A column was removed or renamed; update _cold_rebuild.py or _shared.ENTRY_COLUMNS."
    )


@pytest.mark.slow
def test_rebuild_throughput_10k_files(tmp_path: Path) -> None:
    """Audit finding 140-NFR01-P2 — automated SLO regression guard.

    PRD-CORE-140 NFR01 states rebuild must process 10,000 cold YAML files in
    under 30 seconds. The implementation uses a single transaction and
    per-file streaming iteration which meets this target empirically; this
    test is the CI contract. Generated YAMLs are minimal-valid (only the
    required fields populated) to focus on per-file overhead rather than
    serialization cost.
    """
    import time

    cold_dir = tmp_path / "memory" / "cold" / "2026" / "04"
    cold_dir.mkdir(parents=True)

    # 10_000 minimal YAMLs — raw write avoids ruamel overhead
    for i in range(10_000):
        entry_id = f"L-{i:05x}"
        yaml_text = (
            f"id: {entry_id}\n"
            f"summary: entry {i}\n"
            f"detail: ''\n"
            f"impact: 0.5\n"
            f"status: active\n"
            f"recurrence: 1\n"
            f"namespace: default\n"
            f"created: '2026-04-12T00:00:00+00:00'\n"
            f"updated: '2026-04-12T00:00:00+00:00'\n"
            f"source_type: agent\n"
        )
        (cold_dir / f"{entry_id}.yaml").write_text(yaml_text)

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        start = time.monotonic()
        rebuilt = rebuild_from_cold(tmp_path, conn)
        elapsed = time.monotonic() - start
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
        row = conn.execute(
            "SELECT created_at, updated_at FROM memories WHERE id='L-DATE'"
        ).fetchone()
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


# ---------------------------------------------------------------------------
# FR06: Per-file skip with structured WARNING
# ---------------------------------------------------------------------------


def test_malformed_yaml_skipped_with_warning(tmp_path: Path) -> None:
    """FR06: 3 good + 1 missing-required-field → 3 inserts + 1 WARNING."""
    for i in range(3):
        _make_yaml(tmp_path, f"L-GOOD{i}")
    # Emit a YAML missing ``id`` (required NOT NULL)
    bad = tmp_path / "memory" / "cold" / "2026" / "04" / "bad.yaml"
    bad.write_text(
        "summary: missing id field\ncreated: 2026-04-12\nupdated: 2026-04-12\n",
        encoding="utf-8",
    )

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        with capture_logs() as logs:
            rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 3
        warns = [
            r for r in logs if r.get("event") == "cold_rebuild_skipped" and r.get("log_level") == "warning"
        ]
        assert len(warns) == 1
        assert warns[0]["field"] == "id"
        assert "bad.yaml" in warns[0]["file"]
    finally:
        conn.close()


def test_missing_summary_skipped(tmp_path: Path) -> None:
    """FR06: YAML without summary/content is skipped with field='summary'."""
    bad = tmp_path / "memory" / "cold" / "2026" / "04" / "nosummary.yaml"
    bad.parent.mkdir(parents=True)
    bad.write_text(
        "id: L-NOSUM\ncreated: 2026-04-12\nupdated: 2026-04-12\n",
        encoding="utf-8",
    )

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        with capture_logs() as logs:
            rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 0
        warns = [r for r in logs if r.get("event") == "cold_rebuild_skipped"]
        assert len(warns) == 1
        assert warns[0]["field"] == "summary"
    finally:
        conn.close()


def test_malformed_yaml_parse_error_skipped(tmp_path: Path) -> None:
    """FR06: YAML parse failure is skipped with reason='read_failed'."""
    _make_yaml(tmp_path, "L-GOOD")
    bad = tmp_path / "memory" / "cold" / "2026" / "04" / "parse_err.yaml"
    bad.write_text("this is : not : valid : yaml :\n  [unclosed", encoding="utf-8")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        with capture_logs() as logs:
            rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 1
        reasons = {r.get("reason") for r in logs if r.get("event") == "cold_rebuild_skipped"}
        assert "read_failed" in reasons
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# FR07: Path-traversal guard
# ---------------------------------------------------------------------------


def test_symlink_traversal_skipped(tmp_path: Path) -> None:
    """FR07: a symlink inside cold dir pointing outside is rejected."""
    # Set up a valid good YAML so the traversed one is the only skip
    _make_yaml(tmp_path, "L-OK")

    # Create an out-of-tree YAML and symlink it inside the cold dir
    outside = tmp_path / "outside" / "evil.yaml"
    outside.parent.mkdir(parents=True)
    outside.write_text(
        "id: L-EVIL\nsummary: escaped\ncreated: 2026-04-12\nupdated: 2026-04-12\n",
        encoding="utf-8",
    )
    cold = tmp_path / "memory" / "cold" / "2026" / "04"
    link = cold / "escaped.yaml"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        with capture_logs() as logs:
            rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 1  # only L-OK inserted
        # Ensure the escaped entry did NOT get hydrated
        assert (
            conn.execute("SELECT count(*) FROM memories WHERE id='L-EVIL'").fetchone()[0]
            == 0
        )
        reasons = {
            r.get("reason")
            for r in logs
            if r.get("event") == "cold_rebuild_skipped"
        }
        assert "path_traversal_guard" in reasons
    finally:
        conn.close()


def test_assert_within_cold_dir_raises_on_escape(tmp_path: Path) -> None:
    """FR07 unit: guard raises ValueError on out-of-tree candidate."""
    cold_base = tmp_path / "memory" / "cold"
    cold_base.mkdir(parents=True)
    outside = tmp_path / "evil.yaml"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Path traversal guard"):
        _assert_within_cold_dir(cold_base, outside)


def test_assert_within_cold_dir_accepts_legitimate(tmp_path: Path) -> None:
    """FR07 unit: guard accepts a path strictly under cold root."""
    cold_base = tmp_path / "memory" / "cold"
    inner = cold_base / "2026" / "04" / "ok.yaml"
    inner.parent.mkdir(parents=True)
    inner.write_text("x", encoding="utf-8")
    _assert_within_cold_dir(cold_base, inner)  # does not raise


# ---------------------------------------------------------------------------
# Idempotency (NFR02)
# ---------------------------------------------------------------------------


def test_idempotent_double_run(tmp_path: Path) -> None:
    """NFR02: running rebuild twice yields the same count; second run inserts 0."""
    for i in range(3):
        _make_yaml(tmp_path, f"L-IDEM{i}")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        first = rebuild_from_cold(tmp_path, conn)
        second = rebuild_from_cold(tmp_path, conn)
        assert first == 3
        assert second == 0  # INSERT OR IGNORE suppresses duplicates
        total = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert total == 3
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Hydrator unit tests (direct)
# ---------------------------------------------------------------------------


def test_hydrate_yaml_missing_id_raises() -> None:
    """_hydrate_yaml raises _HydrationError('id') when id missing."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml({"summary": "x", "created": "2026-04-12", "updated": "2026-04-12"})
    assert exc_info.value.field == "id"


def test_hydrate_yaml_hardcodes_type() -> None:
    """_hydrate_yaml directly: the 'type' column slot is always 'pattern'."""
    row = _hydrate_yaml(
        {
            "id": "L-X",
            "summary": "x",
            "created": "2026-04-12",
            "updated": "2026-04-12",
            "source_type": "human",
        }
    )
    assert row is not None
    # type is at position 20 in _INSERT_COLUMNS
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS

    assert row[_INSERT_COLUMNS.index("type")] == "pattern"
    assert row[_INSERT_COLUMNS.index("source")] == "human"


def test_hydrate_yaml_datetime_object_created() -> None:
    """_coerce_ts handles datetime objects (ruamel loads full ISO as datetime)."""
    from datetime import datetime, timezone

    dt = datetime(2026, 4, 12, 15, 30, 0, tzinfo=timezone.utc)
    row = _hydrate_yaml(
        {"id": "L-DT", "summary": "x", "created": dt, "updated": dt}
    )
    assert row is not None
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS

    assert row[_INSERT_COLUMNS.index("created_at")] == dt.isoformat()


def test_hydrate_yaml_date_object_created() -> None:
    """_coerce_ts handles date objects (ruamel loads bare YYYY-MM-DD as date)."""
    from datetime import date

    d = date(2026, 4, 12)
    row = _hydrate_yaml({"id": "L-DO", "summary": "x", "created": d, "updated": d})
    assert row is not None
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS

    assert row[_INSERT_COLUMNS.index("created_at")] == "2026-04-12T00:00:00+00:00"


def test_hydrate_yaml_missing_updated_falls_back_to_created() -> None:
    """Permissive fallback: missing updated reuses created_at."""
    row = _hydrate_yaml(
        {"id": "L-NU", "summary": "x", "created": "2026-04-12"}
    )
    assert row is not None
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS

    created = row[_INSERT_COLUMNS.index("created_at")]
    updated = row[_INSERT_COLUMNS.index("updated_at")]
    assert created == updated


def test_hydrate_yaml_bad_impact_raises() -> None:
    """_hydrate_yaml raises on non-float impact."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml(
            {
                "id": "L-B",
                "summary": "x",
                "created": "2026-04-12",
                "updated": "2026-04-12",
                "impact": "not-a-number",
            }
        )
    assert exc_info.value.field == "impact"


def test_hydrate_yaml_bad_recurrence_raises() -> None:
    """_hydrate_yaml raises on non-int recurrence."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml(
            {
                "id": "L-B",
                "summary": "x",
                "created": "2026-04-12",
                "updated": "2026-04-12",
                "recurrence": "abc",
            }
        )
    assert exc_info.value.field == "recurrence"


def test_hydrate_yaml_bad_list_field_raises() -> None:
    """_hydrate_yaml raises _HydrationError with field name when list shape is wrong."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml(
            {
                "id": "L-L",
                "summary": "x",
                "created": "2026-04-12",
                "updated": "2026-04-12",
                "tags": "not-a-list",
            }
        )
    assert exc_info.value.field == "tags"


def test_hydrate_yaml_bad_dict_field_raises() -> None:
    """_hydrate_yaml raises _HydrationError with field name when dict shape is wrong."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml(
            {
                "id": "L-D",
                "summary": "x",
                "created": "2026-04-12",
                "updated": "2026-04-12",
                "metadata": "not-a-dict",
            }
        )
    assert exc_info.value.field == "metadata"


def test_malformed_list_field_skipped_in_rebuild(tmp_path: Path) -> None:
    """FR06: a YAML with non-list tags is skipped with field='tags' WARNING."""
    _make_yaml(tmp_path, "L-GOOD")
    bad = tmp_path / "memory" / "cold" / "2026" / "04" / "badlist.yaml"
    bad.write_text(
        "id: L-BADLIST\n"
        "summary: bad list\n"
        "created: 2026-04-12\n"
        "updated: 2026-04-12\n"
        "tags: not-a-list\n",
        encoding="utf-8",
    )

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        with capture_logs() as logs:
            rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 1
        tags_warn = [
            r
            for r in logs
            if r.get("event") == "cold_rebuild_skipped" and r.get("field") == "tags"
        ]
        assert len(tags_warn) == 1
    finally:
        conn.close()


def test_duplicate_id_does_not_double_insert(tmp_path: Path) -> None:
    """NFR02: when two YAMLs share an id, only one is inserted."""
    # Create file A under 2026/04
    _make_yaml(tmp_path, "L-DUP")
    # Manually create a second YAML with the same id under a different partition
    second_dir = tmp_path / "memory" / "cold" / "2026" / "05"
    second_dir.mkdir(parents=True)
    second = second_dir / "L-DUP.yaml"
    second.write_text(
        "id: L-DUP\n"
        "summary: duplicate\n"
        "created: 2026-04-12\n"
        "updated: 2026-04-12\n",
        encoding="utf-8",
    )

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuilt = rebuild_from_cold(tmp_path, conn)
        # One inserted, one duplicate-skipped — exact count depends on
        # rglob ordering but the DB must end with a single row.
        assert rebuilt >= 1
        total = conn.execute("SELECT count(*) FROM memories WHERE id='L-DUP'").fetchone()[0]
        assert total == 1
    finally:
        conn.close()


def test_rebuild_skips_insert_sqlite_error(tmp_path: Path) -> None:
    """FR06: sqlite3.Error during INSERT is skipped with WARNING reason='insert_failed'.

    Trigger an insert failure by pre-creating a malformed ``memories`` table
    with a CHECK constraint that rejects every row.
    """
    _make_yaml(tmp_path, "L-IE")

    # Create DB with a replacement memories table that rejects every insert.
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY CHECK (id LIKE 'IMPOSSIBLE%'))")
    conn.commit()

    try:
        with capture_logs() as logs:
            rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 0
        reasons = [
            r.get("reason")
            for r in logs
            if r.get("event") == "cold_rebuild_skipped"
        ]
        assert "insert_failed" in reasons
    finally:
        conn.close()


def test_summary_log_emitted_with_counts(tmp_path: Path) -> None:
    """NFR04: single cold_rebuild_complete log with rebuilt/skipped/cold_files."""
    for i in range(2):
        _make_yaml(tmp_path, f"L-S{i}")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        with capture_logs() as logs:
            rebuild_from_cold(tmp_path, conn)
        summaries = [r for r in logs if r.get("event") == "cold_rebuild_complete"]
        assert len(summaries) == 1
        assert summaries[0]["rebuilt"] == 2
        assert summaries[0]["skipped"] == 0
        assert summaries[0]["cold_files"] == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# FR03: Gated auto-rebuild inside recover_db
# ---------------------------------------------------------------------------


def _corrupt_sqlite_master(db_path: Path) -> None:
    """Destroy sqlite_master the same way the 2026-04-12 incident did."""
    data = db_path.read_bytes()
    corrupted = b"\x00\xff\xfe\xfd" * 512 + data[2048:]
    db_path.write_bytes(corrupted)


def _populate_real_db(db_path: Path, *, entries: int = 3) -> None:
    """Create a non-empty, structurally-valid SQLite DB."""
    from datetime import datetime, timezone

    from trw_memory.models.memory import MemoryEntry, MemoryStatus

    backend = SQLiteBackend(db_path)
    now = datetime.now(timezone.utc)
    for idx in range(entries):
        backend.store(
            MemoryEntry(
                id=f"L-HOT{idx}",
                content=f"hot {idx}",
                importance=0.5,
                status=MemoryStatus.ACTIVE,
                namespace="default",
                source="agent",
                created_at=now,
                updated_at=now,
            )
        )
    backend.close()


def test_recover_db_invokes_rebuild_when_gated(tmp_path: Path) -> None:
    """FR03 happy path: destroyed DB + strict + knob on + 3 cold YAMLs → 3 rows."""
    db_path = tmp_path / "memory.db"
    _populate_real_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)

    # Pre-stage 3 cold YAMLs
    for i in range(3):
        _make_yaml(tmp_path, f"L-COLD{i}")

    conn = SQLiteBackend.recover_db(
        db_path,
        recovery_policy="strict",
        rebuild_from_cold=True,
    )
    try:
        row_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert row_count == 3
        ids = {r[0] for r in conn.execute("SELECT id FROM memories").fetchall()}
        assert ids == {"L-COLD0", "L-COLD1", "L-COLD2"}
    finally:
        conn.close()


def test_recover_db_raises_when_rebuild_yields_zero(tmp_path: Path) -> None:
    """FR03: rebuild with empty cold tier still raises CorruptDatabaseUnsalvageableError."""
    db_path = tmp_path / "memory.db"
    _populate_real_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)

    # No cold YAMLs staged — rebuild will yield 0.
    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(
            db_path,
            recovery_policy="strict",
            rebuild_from_cold=True,
        )


def test_recover_db_knob_off_does_not_rebuild(tmp_path: Path) -> None:
    """FR03 regression: knob=False → no rebuild, strict still raises even with cold YAMLs."""
    db_path = tmp_path / "memory.db"
    _populate_real_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)
    for i in range(3):
        _make_yaml(tmp_path, f"L-IGN{i}")

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend.recover_db(
            db_path,
            recovery_policy="strict",
            rebuild_from_cold=False,
        )


def test_recover_db_non_strict_policy_does_not_rebuild(tmp_path: Path) -> None:
    """FR03 regression: empty_ok policy → no rebuild even with cold YAMLs."""
    db_path = tmp_path / "memory.db"
    _populate_real_db(db_path, entries=1)
    _corrupt_sqlite_master(db_path)
    for i in range(3):
        _make_yaml(tmp_path, f"L-EMPTY{i}")

    conn = SQLiteBackend.recover_db(
        db_path,
        recovery_policy="empty_ok",
        rebuild_from_cold=True,
    )
    try:
        # empty_ok → fresh empty DB, no rebuild
        row_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert row_count == 0
    finally:
        conn.close()


def test_healthy_open_does_not_invoke_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR03 regression: healthy DB open → rebuild_from_cold NEVER invoked."""
    # Pre-stage cold YAMLs that WOULD be rebuilt if the code called rebuild.
    for i in range(3):
        _make_yaml(tmp_path, f"L-COLD{i}")

    # Monkeypatch rebuild_from_cold to count calls.
    import trw_memory.storage._cold_rebuild as cr

    call_counter = {"n": 0}
    original = cr.rebuild_from_cold

    def spy(base_dir: Path, new_conn: sqlite3.Connection) -> int:
        call_counter["n"] += 1
        return original(base_dir, new_conn)

    monkeypatch.setattr(cr, "rebuild_from_cold", spy)

    # Healthy open + close cycle
    backend = SQLiteBackend(tmp_path / "memory.db", rebuild_from_cold=True)
    backend.close()

    assert call_counter["n"] == 0  # never invoked on healthy path


# ---------------------------------------------------------------------------
# FR02: CLI integration
# ---------------------------------------------------------------------------


def test_cli_restore_from_cold_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FR02: trw-memory restore --from-cold exits 0 and prints summary."""
    # Storage under tmp_path so the CLI does not touch the user's real store.
    storage_path = tmp_path / "mem"
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(storage_path))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    # Seed cold tier in the default namespace's base_dir.
    namespace_dir = storage_path / "default"
    namespace_dir.mkdir(parents=True)
    for i in range(4):
        _make_yaml(namespace_dir, f"L-CLI{i}")

    from trw_memory.cli import main

    rc = main(["restore", "--from-cold", "--namespace=default"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Rebuilt 4 entries from cold tier (0 skipped)" in out

    # DB file should exist + contain 4 rows
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
    # Malformed: missing id
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
