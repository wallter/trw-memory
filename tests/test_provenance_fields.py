"""Tests for PRD-CORE-099 provenance fields (client_profile, model_id).

Validates schema migration, model fields, row mapper round-trip,
YAML backend round-trip, and backward compatibility.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._row_mapper import entry_to_row, row_to_entry
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage._shared import ENTRY_COLUMNS

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestProvenanceSchema:
    """DDL and migration tests for client_profile and model_id columns."""

    def test_columns_in_entry_columns(self) -> None:
        assert "client_profile" in ENTRY_COLUMNS
        assert "model_id" in ENTRY_COLUMNS

    def test_client_profile_after_source_identity(self) -> None:
        si_idx = ENTRY_COLUMNS.index("source_identity")
        cp_idx = ENTRY_COLUMNS.index("client_profile")
        mi_idx = ENTRY_COLUMNS.index("model_id")
        assert cp_idx == si_idx + 1
        assert mi_idx == si_idx + 2

    def test_schema_creates_columns(self) -> None:
        conn = sqlite3.connect(":memory:")
        ensure_schema(conn)
        cursor = conn.execute("PRAGMA table_info(memories)")
        col_names = [row[1] for row in cursor.fetchall()]
        assert "client_profile" in col_names
        assert "model_id" in col_names
        conn.close()

    def test_migration_adds_columns_to_existing_db(self) -> None:
        """Simulate an old DB without the new columns, then migrate."""
        conn = sqlite3.connect(":memory:")
        # Create table WITHOUT the new columns (simulating old schema)
        conn.execute("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                detail TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                evidence TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                status TEXT DEFAULT 'active',
                recurrence INTEGER DEFAULT 1,
                namespace TEXT DEFAULT 'default',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                access_count INTEGER DEFAULT 0,
                q_value REAL DEFAULT 0.5,
                q_observations INTEGER DEFAULT 0,
                source TEXT DEFAULT 'agent',
                source_identity TEXT DEFAULT '',
                merged_from TEXT DEFAULT '[]',
                consolidated_from TEXT DEFAULT '[]',
                consolidated_into TEXT,
                metadata TEXT DEFAULT '{}',
                vector_clock TEXT DEFAULT '{}',
                remote_id TEXT,
                published_to_platform INTEGER DEFAULT 0,
                pending_delete INTEGER DEFAULT 0,
                cross_validated INTEGER DEFAULT 0,
                outcome_history TEXT DEFAULT '[]',
                assertions TEXT DEFAULT '[]'
            )
        """)
        conn.commit()
        # Now run migration
        ensure_schema(conn)
        cursor = conn.execute("PRAGMA table_info(memories)")
        col_names = [row[1] for row in cursor.fetchall()]
        assert "client_profile" in col_names
        assert "model_id" in col_names
        conn.close()


class TestProvenanceModel:
    """MemoryEntry model field defaults and validation."""

    def test_default_empty_strings(self) -> None:
        entry = MemoryEntry(
            id="test-1",
            content="test",
            created_at=_NOW,
            updated_at=_NOW,
        )
        assert entry.client_profile == ""
        assert entry.model_id == ""

    def test_explicit_values(self) -> None:
        entry = MemoryEntry(
            id="test-2",
            content="test",
            created_at=_NOW,
            updated_at=_NOW,
            client_profile="claude-code",
            model_id="claude-opus-4-6",
        )
        assert entry.client_profile == "claude-code"
        assert entry.model_id == "claude-opus-4-6"

    def test_to_dict_includes_fields(self) -> None:
        entry = MemoryEntry(
            id="test-3",
            content="test",
            created_at=_NOW,
            updated_at=_NOW,
            client_profile="opencode",
            model_id="claude-sonnet-4-6",
        )
        d = entry.to_dict()
        assert d["client_profile"] == "opencode"
        assert d["model_id"] == "claude-sonnet-4-6"

    def test_to_dict_field_filter(self) -> None:
        entry = MemoryEntry(
            id="test-4",
            content="test",
            created_at=_NOW,
            updated_at=_NOW,
            client_profile="cursor",
        )
        d = entry.to_dict(fields={"id", "client_profile"})
        assert "client_profile" in d
        assert "model_id" not in d


class TestProvenanceRowMapper:
    """Row mapper round-trip with provenance fields."""

    def test_entry_to_row_includes_fields(self) -> None:
        entry = MemoryEntry(
            id="test-rt",
            content="round trip",
            created_at=_NOW,
            updated_at=_NOW,
            client_profile="codex",
            model_id="gpt-4o",
        )
        row = entry_to_row(entry)
        # client_profile and model_id should be in the row tuple
        assert "codex" in row
        assert "gpt-4o" in row

    def test_round_trip(self) -> None:
        original = MemoryEntry(
            id="test-round",
            content="round trip test",
            created_at=_NOW,
            updated_at=_NOW,
            client_profile="claude-code",
            model_id="claude-opus-4-6",
        )
        row = entry_to_row(original)
        restored = row_to_entry(row)
        assert restored.client_profile == "claude-code"
        assert restored.model_id == "claude-opus-4-6"
        assert restored.id == "test-round"

    def test_empty_provenance_round_trip(self) -> None:
        original = MemoryEntry(
            id="test-empty",
            content="empty provenance",
            created_at=_NOW,
            updated_at=_NOW,
        )
        row = entry_to_row(original)
        restored = row_to_entry(row)
        assert restored.client_profile == ""
        assert restored.model_id == ""


class TestProvenanceYamlBackend:
    """YAML backend round-trip for provenance fields."""

    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        """YAMLBackend write → read preserves client_profile and model_id."""
        from trw_memory.storage.yaml_backend import YAMLBackend

        backend = YAMLBackend(tmp_path)
        entry = MemoryEntry(
            id="yaml-prov-001",
            content="yaml provenance test",
            created_at=_NOW,
            updated_at=_NOW,
            client_profile="claude-code",
            model_id="claude-opus-4-6",
        )
        backend.store(entry)
        restored = backend.get("yaml-prov-001")
        assert restored is not None
        assert restored.client_profile == "claude-code"
        assert restored.model_id == "claude-opus-4-6"

    def test_yaml_empty_provenance_round_trip(self, tmp_path: Path) -> None:
        """YAMLBackend correctly round-trips entries with empty provenance."""
        from trw_memory.storage.yaml_backend import YAMLBackend

        backend = YAMLBackend(tmp_path)
        entry = MemoryEntry(
            id="yaml-prov-002",
            content="empty provenance yaml",
            created_at=_NOW,
            updated_at=_NOW,
        )
        backend.store(entry)
        restored = backend.get("yaml-prov-002")
        assert restored is not None
        assert restored.client_profile == ""
        assert restored.model_id == ""
