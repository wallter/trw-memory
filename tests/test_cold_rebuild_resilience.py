# ruff: noqa: F401
"""Malformed-input, logging, and guard tests for cold rebuild."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from trw_memory.storage._cold_rebuild import _assert_within_cold_dir, rebuild_from_cold

from ._test_cold_rebuild_support import _configure_structlog, _make_yaml, _open_fresh_db


def test_malformed_yaml_skipped_with_warning(tmp_path: Path) -> None:
    """FR06: 3 good + 1 missing-required-field → 3 inserts + 1 WARNING."""
    for i in range(3):
        _make_yaml(tmp_path, f"L-GOOD{i}")
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
        warns = [r for r in logs if r.get("event") == "cold_rebuild_skipped" and r.get("log_level") == "warning"]
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


@pytest.mark.parametrize("payload", ["- one\n- two\n", "plain scalar\n"])
def test_non_mapping_yaml_is_skipped_without_aborting_rebuild(tmp_path: Path, payload: str) -> None:
    """FR06: valid YAML with the wrong root shape is a per-file warning, not a rollback."""
    _make_yaml(tmp_path, "L-GOOD")
    bad = tmp_path / "memory" / "cold" / "2026" / "04" / "wrong-shape.yaml"
    bad.write_text(payload, encoding="utf-8")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        with capture_logs() as logs:
            rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 1
        warnings = [r for r in logs if r.get("event") == "cold_rebuild_skipped"]
        assert any(r.get("reason") == "read_failed" for r in warnings)
        completed = [r for r in logs if r.get("event") == "cold_rebuild_complete"]
        assert completed[-1]["files_skipped"] == 1
    finally:
        conn.close()


def test_symlink_traversal_skipped(tmp_path: Path) -> None:
    """FR07: a symlink inside cold dir pointing outside is rejected."""
    _make_yaml(tmp_path, "L-OK")
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
        assert rebuilt == 1
        assert conn.execute("SELECT count(*) FROM memories WHERE id='L-EVIL'").fetchone()[0] == 0
        reasons = {r.get("reason") for r in logs if r.get("event") == "cold_rebuild_skipped"}
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
    _assert_within_cold_dir(cold_base, inner)


def test_idempotent_double_run(tmp_path: Path) -> None:
    """NFR02: running rebuild twice yields the same count; second run inserts 0."""
    for i in range(3):
        _make_yaml(tmp_path, f"L-IDEM{i}")

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        first = rebuild_from_cold(tmp_path, conn)
        second = rebuild_from_cold(tmp_path, conn)
        assert first == 3
        assert second == 0
        total = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert total == 3
    finally:
        conn.close()


def test_malformed_list_field_skipped_in_rebuild(tmp_path: Path) -> None:
    """FR06: a YAML with non-list tags is skipped with field='tags' WARNING."""
    _make_yaml(tmp_path, "L-GOOD")
    bad = tmp_path / "memory" / "cold" / "2026" / "04" / "badlist.yaml"
    bad.write_text(
        "id: L-BADLIST\nsummary: bad list\ncreated: 2026-04-12\nupdated: 2026-04-12\ntags: not-a-list\n",
        encoding="utf-8",
    )

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        with capture_logs() as logs:
            rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 1
        tags_warn = [r for r in logs if r.get("event") == "cold_rebuild_skipped" and r.get("field") == "tags"]
        assert len(tags_warn) == 1
    finally:
        conn.close()


def test_duplicate_id_does_not_double_insert(tmp_path: Path) -> None:
    """NFR02: when two YAMLs share an id, only one is inserted."""
    _make_yaml(tmp_path, "L-DUP")
    second_dir = tmp_path / "memory" / "cold" / "2026" / "05"
    second_dir.mkdir(parents=True)
    second = second_dir / "L-DUP.yaml"
    second.write_text(
        "id: L-DUP\nsummary: duplicate\ncreated: 2026-04-12\nupdated: 2026-04-12\n",
        encoding="utf-8",
    )

    conn = _open_fresh_db(tmp_path / "memory.db")
    try:
        rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt >= 1
        total = conn.execute("SELECT count(*) FROM memories WHERE id='L-DUP'").fetchone()[0]
        assert total == 1
    finally:
        conn.close()


def test_rebuild_skips_insert_sqlite_error(tmp_path: Path) -> None:
    """FR06: sqlite3.Error during INSERT is skipped with WARNING reason='insert_failed'."""
    _make_yaml(tmp_path, "L-IE")
    conn = sqlite3.connect(str(tmp_path / "memory.db"))
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY CHECK (id LIKE 'IMPOSSIBLE%'))")
    conn.commit()

    try:
        with capture_logs() as logs:
            rebuilt = rebuild_from_cold(tmp_path, conn)
        assert rebuilt == 0
        reasons = [r.get("reason") for r in logs if r.get("event") == "cold_rebuild_skipped"]
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
