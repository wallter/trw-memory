"""Q3 (trw-memory 4.0.0 security review): ``quarantine_reviews`` gains a namespace column.

Proves ``_migrate_v8_quarantine_review_namespace`` against a **real** pre-Q3
quarantine database: ``fixtures/pre_q3_quarantine/quarantine.db`` was built by
the actual historical (pre-fix) ``store_quarantined_entry``/
``review_quarantined_entry`` code (``make_store.py`` extracts it via
``git archive`` from the last commit before this migration), so the fixture's
``quarantine_reviews`` rows are produced by the genuine pre-fix implementation
-- never a hand-written guess at the old shape.

Two scenarios are baked into the fixture:
  - ``M-solo``: quarantined exactly once, in ``project:a`` -- unambiguous,
    must backfill to ``project:a``.
  - ``M-dup``: quarantined in ``project:a``, approved (and so deleted from the
    quarantine DB), then a DIFFERENT, unrelated entry with the SAME id
    quarantined in ``project:b`` and left there. A naive "match current
    `memories`" backfill would find exactly one live row (``project:b``) and
    mislabel ``project:a``'s historical reviewer/decision as ``project:b``'s --
    reproducing the exact cross-namespace leak Q3 exists to close. This must
    stay ``''`` (unknown) for every ``M-dup`` row.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from trw_memory.storage._schema import SCHEMA_VERSION, ensure_schema

pytestmark = pytest.mark.integration

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pre_q3_quarantine"


def _build_historical_quarantine_store(db_path: Path) -> dict[str, object]:
    shutil.copyfile(_FIXTURE_DIR / "quarantine.db", db_path)
    return dict(json.loads((_FIXTURE_DIR / "receipt.json").read_text()))


def test_fixture_is_genuinely_pre_namespace_column() -> None:
    """Sanity: the committed fixture predates this migration (non-vacuity)."""
    receipt = dict(json.loads((_FIXTURE_DIR / "receipt.json").read_text()))
    assert "namespace" not in receipt["columns"]
    assert receipt["user_version"] < SCHEMA_VERSION


def test_populated_pre_v8_table_migrates_and_backfills_unambiguous_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "quarantine.db"
    receipt = _build_historical_quarantine_store(db_path)
    rows_before = receipt["rows"]
    assert len(rows_before) == 4, "fixture setup must carry the two scenarios described above"

    conn = sqlite3.connect(db_path)
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == receipt["user_version"]
    columns_before = {row[1] for row in conn.execute("PRAGMA table_info(quarantine_reviews)").fetchall()}
    assert "namespace" not in columns_before

    ensure_schema(conn)

    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    columns_after = {row[1] for row in conn.execute("PRAGMA table_info(quarantine_reviews)").fetchall()}
    assert "namespace" in columns_after

    # No data loss: every original row is preserved (learning_id, decision, reviewer_id).
    after_rows = conn.execute(
        "SELECT learning_id, decision, reviewer_id, namespace FROM quarantine_reviews ORDER BY id"
    ).fetchall()
    assert [list(row[:3]) for row in after_rows] == rows_before

    by_learning_id: dict[str, list[str]] = {}
    for learning_id, _decision, _reviewer_id, namespace in after_rows:
        by_learning_id.setdefault(learning_id, []).append(namespace)

    # Unambiguous: M-solo was only ever quarantined in project:a.
    assert by_learning_id["M-solo"] == ["project:a"]

    # Ambiguous collision: every M-dup row (the deleted project:a episode AND
    # the still-live project:b one) must stay unresolved, not guessed at
    # project:b just because that is the only live row left.
    assert by_learning_id["M-dup"] == ["", "", ""]

    conn.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "quarantine.db"
    _build_historical_quarantine_store(db_path)
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    first_pass = conn.execute(
        "SELECT learning_id, decision, reviewer_id, namespace FROM quarantine_reviews ORDER BY id"
    ).fetchall()

    ensure_schema(conn)
    second_pass = conn.execute(
        "SELECT learning_id, decision, reviewer_id, namespace FROM quarantine_reviews ORDER BY id"
    ).fetchall()

    assert first_pass == second_pass
    conn.close()


def test_a_database_with_no_quarantine_reviews_table_is_a_no_op(tmp_path: Path) -> None:
    """A store that never quarantined anything has no ``quarantine_reviews`` table at all."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    backend = SQLiteBackend(tmp_path / "fresh.db")
    try:
        tables = {
            row[0]
            for row in backend._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'quarantine_reviews'"
            ).fetchall()
        }
        assert tables == set()
        assert int(backend._conn.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
    finally:
        backend.close()


def test_a_fresh_table_created_by_current_code_already_has_namespace(tmp_path: Path) -> None:
    """The lazy ``CREATE TABLE IF NOT EXISTS`` in the fixed code already includes the column."""
    from trw_memory.models.config import MemoryConfig
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.security.runtime import store_quarantined_entry

    config = MemoryConfig(storage_path=str(tmp_path / "active"), quarantine_db_path=str(tmp_path / "quarantine.db"))
    store_quarantined_entry(config, MemoryEntry(id="M-fresh", content="suspicious", namespace="project:fresh"))
    conn = sqlite3.connect(tmp_path / "quarantine.db")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(quarantine_reviews)").fetchall()}
    assert "namespace" in columns
    row = conn.execute("SELECT namespace FROM quarantine_reviews WHERE learning_id = 'M-fresh'").fetchone()
    assert row == ("project:fresh",)
    conn.close()
