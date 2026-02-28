"""Tests for trw_memory.migration.from_trw.

Covers:
- from_learning_entry: minimal dict, field mapping, date conversion,
  all fields including merged_from/consolidated_from, missing fields,
  unknown fields are ignored, lenient numeric coercion
- migrate_entries_dir: reads YAML files from tmp directory, skips
  non-existent directory, skips malformed files gracefully
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from trw_memory.migration.from_trw import from_learning_entry, migrate_entries_dir
from trw_memory.models.memory import MemoryEntry, MemoryStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: object) -> None:
    yml = YAML()
    yml.default_flow_style = False
    with open(path, "w", encoding="utf-8") as fh:
        yml.dump(data, fh)


# ---------------------------------------------------------------------------
# from_learning_entry — happy path: minimal dict
# ---------------------------------------------------------------------------


def test_from_learning_entry_minimal() -> None:
    """Minimal required fields produce a valid MemoryEntry with defaults."""
    data: dict[str, object] = {
        "id": "L-001",
        "summary": "Use absolute paths",
        "impact": 0.7,
        "status": "active",
        "created": date(2026, 1, 15),
    }
    entry = from_learning_entry(data)
    assert isinstance(entry, MemoryEntry)
    assert entry.id == "L-001"
    assert entry.content == "Use absolute paths"
    assert entry.importance == 0.7
    assert entry.status == MemoryStatus.ACTIVE


# ---------------------------------------------------------------------------
# from_learning_entry — field mapping
# ---------------------------------------------------------------------------


def test_from_learning_entry_summary_maps_to_content() -> None:
    data: dict[str, object] = {
        "id": "L-002",
        "summary": "summary text here",
        "created": date(2026, 1, 1),
    }
    entry = from_learning_entry(data)
    assert entry.content == "summary text here"


def test_from_learning_entry_impact_maps_to_importance() -> None:
    data: dict[str, object] = {
        "id": "L-003",
        "summary": "x",
        "impact": 0.85,
        "created": date(2026, 1, 1),
    }
    entry = from_learning_entry(data)
    assert entry.importance == 0.85


# ---------------------------------------------------------------------------
# from_learning_entry — date → datetime conversion
# ---------------------------------------------------------------------------


def test_from_learning_entry_date_to_datetime_midnight_utc() -> None:
    data: dict[str, object] = {
        "id": "L-004",
        "summary": "date conversion test",
        "created": date(2026, 3, 10),
        "updated": date(2026, 3, 15),
    }
    entry = from_learning_entry(data)
    assert entry.created_at == datetime(2026, 3, 10, 0, 0, 0, tzinfo=timezone.utc)
    assert entry.updated_at == datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
    assert entry.created_at.tzinfo is not None
    assert entry.updated_at.tzinfo is not None


def test_from_learning_entry_iso_string_date_conversion() -> None:
    """ISO date strings (as loaded from YAML) also convert correctly."""
    data: dict[str, object] = {
        "id": "L-005",
        "summary": "iso string date",
        "created": "2026-02-01",
    }
    entry = from_learning_entry(data)
    assert entry.created_at == datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# from_learning_entry — all fields including provenance
# ---------------------------------------------------------------------------


def test_from_learning_entry_all_fields() -> None:
    """All LearningEntry fields are correctly mapped."""
    data: dict[str, object] = {
        "id": "L-006",
        "summary": "full entry",
        "detail": "extended explanation",
        "tags": ["pydantic", "yaml"],
        "evidence": ["test_x.py::test_y"],
        "impact": 0.9,
        "status": "resolved",
        "created": date(2026, 1, 1),
        "updated": date(2026, 2, 1),
        "last_accessed_at": date(2026, 2, 20),
        "recurrence": 4,
        "access_count": 7,
        "q_value": 0.75,
        "q_observations": 12,
        "namespace": "project",
        "source": "human",
        "source_identity": "tyler",
        "merged_from": ["L-003", "L-004"],
        "consolidated_from": ["L-010"],
        "consolidated_into": "L-099",
        "metadata": {"sprint": "31"},
    }
    entry = from_learning_entry(data)
    assert entry.id == "L-006"
    assert entry.content == "full entry"
    assert entry.detail == "extended explanation"
    assert entry.tags == ["pydantic", "yaml"]
    assert entry.evidence == ["test_x.py::test_y"]
    assert entry.importance == 0.9
    assert entry.status == MemoryStatus.RESOLVED
    assert entry.recurrence == 4
    assert entry.access_count == 7
    assert entry.q_value == 0.75
    assert entry.q_observations == 12
    assert entry.namespace == "project"
    assert entry.source == "human"
    assert entry.source_identity == "tyler"
    assert entry.merged_from == ["L-003", "L-004"]
    assert entry.consolidated_from == ["L-010"]
    assert entry.consolidated_into == "L-099"
    assert entry.metadata == {"sprint": "31"}
    assert entry.last_accessed_at == datetime(2026, 2, 20, 0, 0, 0, tzinfo=timezone.utc)


def test_from_learning_entry_obsolete_status() -> None:
    data: dict[str, object] = {
        "id": "L-007",
        "summary": "obsolete entry",
        "status": "obsolete",
        "created": date(2026, 1, 1),
    }
    entry = from_learning_entry(data)
    assert entry.status == MemoryStatus.OBSOLETE


# ---------------------------------------------------------------------------
# from_learning_entry — missing fields use MemoryEntry defaults
# ---------------------------------------------------------------------------


def test_from_learning_entry_missing_fields_use_defaults() -> None:
    """An entry with only id and summary gets all other fields from defaults."""
    data: dict[str, object] = {
        "id": "L-008",
        "summary": "bare minimum",
    }
    entry = from_learning_entry(data)
    assert entry.importance == 0.5
    assert entry.status == MemoryStatus.ACTIVE
    assert entry.tags == []
    assert entry.merged_from == []
    assert entry.consolidated_from == []
    assert entry.consolidated_into is None
    assert entry.namespace == "default"
    assert entry.source == "agent"
    assert entry.recurrence == 1
    assert entry.access_count == 0


def test_from_learning_entry_unknown_fields_ignored() -> None:
    """Unknown keys in the source dict are silently ignored."""
    data: dict[str, object] = {
        "id": "L-009",
        "summary": "unknown field test",
        "created": date(2026, 1, 1),
        "some_future_field": "ignored",
        "another_unknown": 999,
    }
    entry = from_learning_entry(data)
    assert entry.id == "L-009"


def test_from_learning_entry_missing_id_generates_uuid() -> None:
    """Missing id generates a UUID4 string."""
    data: dict[str, object] = {
        "summary": "no id provided",
        "created": date(2026, 1, 1),
    }
    entry = from_learning_entry(data)
    assert len(entry.id) == 36  # UUID4 format: 8-4-4-4-12


def test_from_learning_entry_impact_clamped_to_range() -> None:
    """Impact values outside [0, 1] are clamped rather than rejected."""
    high_data: dict[str, object] = {"id": "L-clamp-hi", "summary": "x", "impact": 2.5}
    low_data: dict[str, object] = {"id": "L-clamp-lo", "summary": "x", "impact": -0.5}
    assert from_learning_entry(high_data).importance == 1.0
    assert from_learning_entry(low_data).importance == 0.0


# ---------------------------------------------------------------------------
# migrate_entries_dir — reads YAML files from tmp directory
# ---------------------------------------------------------------------------


def test_migrate_entries_dir_reads_yaml_files(tmp_path: Path) -> None:
    """migrate_entries_dir converts all YAML files in the given directory."""
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()

    _write_yaml(
        entries_dir / "entry-001.yaml",
        {
            "id": "L-test001",
            "summary": "Test learning one",
            "detail": "Detailed explanation for one",
            "tags": ["test"],
            "impact": 0.7,
            "status": "active",
            "created": "2026-01-01",
            "updated": "2026-02-01",
        },
    )
    _write_yaml(
        entries_dir / "entry-002.yaml",
        {
            "id": "L-test002",
            "summary": "Test learning two",
            "impact": 0.5,
            "status": "resolved",
            "created": "2026-01-15",
            "updated": "2026-02-15",
        },
    )

    results = migrate_entries_dir(entries_dir)
    assert len(results) == 2
    ids = {e.id for e in results}
    assert "L-test001" in ids
    assert "L-test002" in ids


def test_migrate_entries_dir_field_values_correct(tmp_path: Path) -> None:
    """Field values from YAML are faithfully mapped into MemoryEntry."""
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()

    _write_yaml(
        entries_dir / "entry-full.yaml",
        {
            "id": "L-full",
            "summary": "Check all fields migrate",
            "detail": "Extended detail text",
            "tags": ["migration", "yaml"],
            "impact": 0.8,
            "status": "active",
            "created": "2026-01-01",
            "updated": "2026-02-01",
        },
    )

    results = migrate_entries_dir(entries_dir)
    assert len(results) == 1
    entry = results[0]
    assert entry.id == "L-full"
    assert entry.content == "Check all fields migrate"
    assert entry.detail == "Extended detail text"
    assert entry.tags == ["migration", "yaml"]
    assert entry.importance == 0.8
    assert entry.status == MemoryStatus.ACTIVE
    assert entry.created_at == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert entry.updated_at == datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_migrate_entries_dir_nonexistent_returns_empty(tmp_path: Path) -> None:
    """A nonexistent directory returns an empty list without raising."""
    results = migrate_entries_dir(tmp_path / "does_not_exist")
    assert results == []


def test_migrate_entries_dir_skips_malformed_yaml(tmp_path: Path) -> None:
    """A malformed YAML file is skipped; valid files are still processed."""
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()

    # Valid file
    _write_yaml(
        entries_dir / "valid.yaml",
        {"id": "L-valid", "summary": "good", "created": "2026-01-01"},
    )
    # Malformed (not a dict) — ruamel.yaml will parse as a scalar string
    (entries_dir / "bad.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")

    results = migrate_entries_dir(entries_dir)
    assert len(results) == 1
    assert results[0].id == "L-valid"


def test_migrate_entries_dir_three_files_all_converted(tmp_path: Path) -> None:
    """Three sample files all convert successfully."""
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()

    for i in range(1, 4):
        _write_yaml(
            entries_dir / f"entry-{i:03d}.yaml",
            {
                "id": f"L-{i:03d}",
                "summary": f"Learning number {i}",
                "impact": round(0.3 + i * 0.2, 1),
                "status": "active",
                "created": "2026-01-01",
            },
        )

    results = migrate_entries_dir(entries_dir)
    assert len(results) == 3
    importances = [e.importance for e in results]
    assert 0.5 in importances
    assert 0.7 in importances
    assert 0.9 in importances
