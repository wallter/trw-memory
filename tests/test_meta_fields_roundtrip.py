"""Integration tests for meta-learning field store/retrieve round-trips.

Exercises the full SQLiteBackend store -> retrieve cycle for all 15+
meta-learning columns added by PRD-CORE-108/110/111 implementations.
Uses real SQLiteBackend instances with tmp_path -- no mocks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import (
    Anchor,
    MemoryEntry,
)
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_entry(entry_id: str = "M-FULL") -> MemoryEntry:
    """Create a MemoryEntry with ALL meta-learning fields populated."""
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content="Full meta-learning entry for round-trip test",
        detail="Detailed explanation of the finding",
        tags=["auth", "security"],
        evidence=["error log at line 42", "stack trace from prod"],
        importance=0.85,
        recurrence=3,
        namespace="test-ns",
        created_at=now,
        updated_at=now,
        last_accessed_at=now,
        access_count=7,
        q_value=0.72,
        q_observations=15,
        source="agent",
        source_identity="claude-opus",
        client_profile="claude-code",
        model_id="claude-opus-4-6",
        merged_from=["M-OLD-1", "M-OLD-2"],
        consolidated_from=["M-PREV-1"],
        consolidated_into=None,
        metadata={"sprint": "42", "prd": "CORE-110"},
        vector_clock={"node-a": 3, "node-b": 1},
        remote_id="remote-abc",
        published_to_platform=True,
        pending_delete=False,
        cross_validated=True,
        outcome_history=["boost:+0.05:reason=cross_validated"],
        # PRD-CORE-110: Typed entry fields
        type="incident",
        confidence="verified",
        protection_tier="critical",
        domain=["auth", "api"],
        phase_origin="IMPLEMENT",
        phase_affinity=["IMPLEMENT", "VALIDATE"],
        team_origin="backend",
        nudge_line="Test nudge line",
        task_type="bug-fix",
        expires="2027-01-01",
        # PRD-CORE-111: Code anchors
        anchors=[
            Anchor(
                file="src/foo.py",
                symbol_name="bar",
                symbol_type="function",
                signature="def bar():",
            ),
        ],
        anchor_validity=1.0,
        # PRD-CORE-108: Outcome attribution
        sessions_surfaced=5,
        avg_rework_delta=0.3,
        outcome_correlation="positive",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFullColumnRoundTrip:
    """Store a fully-populated entry, retrieve it, and verify every field."""

    def test_45_column_roundtrip(self, tmp_path: Path) -> None:
        """All 45 columns survive a store -> get round-trip."""
        backend = SQLiteBackend(tmp_path / "roundtrip.db")
        entry = _make_full_entry("M-RT-FULL")
        backend.store(entry)

        retrieved = backend.get("M-RT-FULL")
        assert retrieved is not None

        # Core fields
        assert retrieved.id == "M-RT-FULL"
        assert retrieved.content == entry.content
        assert retrieved.detail == entry.detail
        assert retrieved.tags == entry.tags
        assert retrieved.evidence == entry.evidence
        assert retrieved.importance == pytest.approx(entry.importance)
        assert retrieved.status == entry.status
        assert retrieved.recurrence == entry.recurrence
        assert retrieved.namespace == entry.namespace
        assert retrieved.access_count == entry.access_count
        assert retrieved.q_value == pytest.approx(entry.q_value)
        assert retrieved.q_observations == entry.q_observations

        # Provenance
        assert retrieved.source == entry.source
        assert retrieved.source_identity == entry.source_identity
        assert retrieved.client_profile == entry.client_profile
        assert retrieved.model_id == entry.model_id

        # Merge/consolidation
        assert retrieved.merged_from == entry.merged_from
        assert retrieved.consolidated_from == entry.consolidated_from
        assert retrieved.consolidated_into == entry.consolidated_into

        # Metadata
        assert retrieved.metadata == entry.metadata
        assert retrieved.vector_clock == entry.vector_clock
        assert retrieved.remote_id == entry.remote_id
        assert retrieved.published_to_platform == entry.published_to_platform
        assert retrieved.pending_delete == entry.pending_delete
        assert retrieved.cross_validated == entry.cross_validated
        assert retrieved.outcome_history == entry.outcome_history

        # PRD-CORE-110: Typed entry fields
        assert retrieved.type == "incident"
        assert retrieved.confidence == "verified"
        assert retrieved.protection_tier == "critical"
        assert retrieved.domain == ["auth", "api"]
        assert retrieved.phase_origin == "IMPLEMENT"
        assert retrieved.phase_affinity == ["IMPLEMENT", "VALIDATE"]
        assert retrieved.team_origin == "backend"
        assert retrieved.nudge_line == "Test nudge line"
        assert retrieved.task_type == "bug-fix"
        assert retrieved.expires == "2027-01-01"

        # PRD-CORE-111: Anchors
        assert len(retrieved.anchors) == 1
        assert retrieved.anchors[0].file == "src/foo.py"
        assert retrieved.anchors[0].symbol_name == "bar"
        assert retrieved.anchors[0].symbol_type == "function"
        assert retrieved.anchors[0].signature == "def bar():"
        assert retrieved.anchor_validity == pytest.approx(1.0)

        # PRD-CORE-108: Outcome attribution
        assert retrieved.sessions_surfaced == 5
        assert retrieved.avg_rework_delta == pytest.approx(0.3)
        assert retrieved.outcome_correlation == "positive"

        backend.close()


@pytest.mark.unit
class TestEnumRoundTrip:
    """All enum values for MemoryType, Confidence, ProtectionTier survive round-trip."""

    @pytest.mark.parametrize(
        "type_val",
        ["incident", "pattern", "convention", "hypothesis", "workaround"],
    )
    def test_memory_type_roundtrip(self, tmp_path: Path, type_val: str) -> None:
        """Each MemoryType value survives store -> get."""
        backend = SQLiteBackend(tmp_path / f"type_{type_val}.db")
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id=f"M-TYPE-{type_val}",
            content=f"test {type_val}",
            type=type_val,
            created_at=now,
            updated_at=now,
        )
        backend.store(entry)
        retrieved = backend.get(f"M-TYPE-{type_val}")
        assert retrieved is not None
        assert retrieved.type == type_val
        backend.close()

    @pytest.mark.parametrize(
        "conf_val",
        ["unverified", "low", "medium", "high", "verified"],
    )
    def test_confidence_roundtrip(self, tmp_path: Path, conf_val: str) -> None:
        """Each Confidence value survives store -> get."""
        backend = SQLiteBackend(tmp_path / f"conf_{conf_val}.db")
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id=f"M-CONF-{conf_val}",
            content=f"test {conf_val}",
            confidence=conf_val,
            created_at=now,
            updated_at=now,
        )
        backend.store(entry)
        retrieved = backend.get(f"M-CONF-{conf_val}")
        assert retrieved is not None
        assert retrieved.confidence == conf_val
        backend.close()

    @pytest.mark.parametrize(
        "tier_val",
        ["critical", "high", "normal", "low", "protected", "permanent"],
    )
    def test_protection_tier_roundtrip(self, tmp_path: Path, tier_val: str) -> None:
        """Each ProtectionTier value survives store -> get."""
        backend = SQLiteBackend(tmp_path / f"tier_{tier_val}.db")
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id=f"M-TIER-{tier_val}",
            content=f"test {tier_val}",
            protection_tier=tier_val,
            created_at=now,
            updated_at=now,
        )
        backend.store(entry)
        retrieved = backend.get(f"M-TIER-{tier_val}")
        assert retrieved is not None
        assert retrieved.protection_tier == tier_val
        backend.close()


@pytest.mark.unit
class TestAnchorJsonRoundTrip:
    """Anchor objects with different symbol_types survive store -> retrieve."""

    def test_three_anchors_roundtrip(self, tmp_path: Path) -> None:
        """3 Anchor objects with function/class/method types survive round-trip."""
        backend = SQLiteBackend(tmp_path / "anchors.db")
        now = datetime.now(timezone.utc)
        anchors = [
            Anchor(
                file="src/auth.py",
                symbol_name="validate_token",
                symbol_type="function",
                signature="def validate_token(token: str) -> bool:",
            ),
            Anchor(
                file="src/models.py",
                symbol_name="UserProfile",
                symbol_type="class",
                signature="class UserProfile(BaseModel):",
            ),
            Anchor(
                file="src/api.py",
                symbol_name="get_user",
                symbol_type="method",
                signature="def get_user(self, user_id: int) -> User:",
            ),
        ]
        entry = MemoryEntry(
            id="M-ANCHORS",
            content="multi-anchor test",
            anchors=anchors,
            anchor_validity=0.95,
            created_at=now,
            updated_at=now,
        )
        backend.store(entry)

        retrieved = backend.get("M-ANCHORS")
        assert retrieved is not None
        assert len(retrieved.anchors) == 3

        # Verify each anchor individually
        a0 = retrieved.anchors[0]
        assert a0.file == "src/auth.py"
        assert a0.symbol_name == "validate_token"
        assert a0.symbol_type == "function"
        assert a0.signature == "def validate_token(token: str) -> bool:"

        a1 = retrieved.anchors[1]
        assert a1.file == "src/models.py"
        assert a1.symbol_name == "UserProfile"
        assert a1.symbol_type == "class"
        assert a1.signature == "class UserProfile(BaseModel):"

        a2 = retrieved.anchors[2]
        assert a2.file == "src/api.py"
        assert a2.symbol_name == "get_user"
        assert a2.symbol_type == "method"
        assert a2.signature == "def get_user(self, user_id: int) -> User:"

        assert retrieved.anchor_validity == pytest.approx(0.95)
        backend.close()


@pytest.mark.unit
class TestSchemaMigrationIdempotency:
    """Opening a second SQLiteBackend on the same DB path is safe."""

    def test_reopen_preserves_data(self, tmp_path: Path) -> None:
        """Creating a new SQLiteBackend on existing DB preserves data."""
        db_path = tmp_path / "migrate.db"

        # First backend -- create and store
        backend1 = SQLiteBackend(db_path)
        entry = _make_full_entry("M-MIGRATE")
        backend1.store(entry)
        backend1.close()

        # Second backend on same path -- triggers schema/migration again
        backend2 = SQLiteBackend(db_path)
        retrieved = backend2.get("M-MIGRATE")
        assert retrieved is not None
        assert retrieved.id == "M-MIGRATE"
        assert retrieved.type == "incident"
        assert retrieved.confidence == "verified"
        assert retrieved.protection_tier == "critical"
        assert len(retrieved.anchors) == 1
        assert retrieved.sessions_surfaced == 5
        assert retrieved.avg_rework_delta == pytest.approx(0.3)
        backend2.close()

    def test_ensure_schema_twice_no_error(self, tmp_path: Path) -> None:
        """Calling ensure_schema twice on the same connection does not error."""
        backend = SQLiteBackend(tmp_path / "idempotent.db")
        # Second call should be a no-op
        ensure_schema(backend._conn)
        # Verify schema still intact
        cursor = backend._conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "type" in columns
        assert "confidence" in columns
        assert "anchors" in columns
        assert "sessions_surfaced" in columns
        backend.close()


@pytest.mark.unit
class TestEmptyFieldsBackwardCompat:
    """Entries created without meta-learning fields get correct defaults."""

    def test_minimal_entry_defaults(self, tmp_path: Path) -> None:
        """A MemoryEntry with no meta-learning fields gets safe defaults after round-trip."""
        backend = SQLiteBackend(tmp_path / "compat.db")
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="M-MINIMAL",
            content="pre-meta-learning entry",
            created_at=now,
            updated_at=now,
        )
        backend.store(entry)

        retrieved = backend.get("M-MINIMAL")
        assert retrieved is not None

        # PRD-CORE-110 defaults
        assert retrieved.type == "pattern"
        assert retrieved.confidence == "unverified"
        assert retrieved.protection_tier == "normal"
        assert retrieved.domain == []
        assert retrieved.phase_origin == ""
        assert retrieved.phase_affinity == []
        assert retrieved.team_origin == ""
        assert retrieved.nudge_line == ""
        assert retrieved.task_type == ""
        assert retrieved.expires == ""

        # PRD-CORE-111 defaults
        assert retrieved.anchors == []
        assert retrieved.anchor_validity == pytest.approx(1.0)

        # PRD-CORE-108 defaults
        assert retrieved.sessions_surfaced == 0
        assert retrieved.avg_rework_delta is None
        assert retrieved.outcome_correlation == ""

        backend.close()

    def test_search_returns_entry_with_defaults(self, tmp_path: Path) -> None:
        """search() returns entries that have default meta-learning fields."""
        backend = SQLiteBackend(tmp_path / "search_compat.db")
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="M-SEARCH",
            content="searchable content for backward compat test",
            created_at=now,
            updated_at=now,
        )
        backend.store(entry)

        results = backend.search("searchable content", top_k=10)
        assert len(results) == 1
        result = results[0]
        assert result.id == "M-SEARCH"
        assert result.type == "pattern"
        assert result.confidence == "unverified"
        assert result.anchors == []
        backend.close()
