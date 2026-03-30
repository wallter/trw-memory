"""Tests for PRD-QUAL-053: Type Safety and Validation Hardening.

Covers:
  FR-01: Config Literal types and weight bounds
  FR-02: Source field validation with backward-compatible coercion
  FR-03: Fixture typing (compile-time, tested via mypy)
  FR-04: Narrowed auto_recall exception handling
  FR-05: Enum redundancy cleanup in assertion pattern validator
  FR-06: Dimension mismatch catch in tier scoring
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# FR-01: Config Literal Types and Weight Bounds
# ---------------------------------------------------------------------------


class TestConfigLiteralTypes:
    """Verify that MemoryConfig rejects invalid Literal values."""

    def test_config_invalid_pii_action_rejected(self) -> None:
        """Invalid pii_action string must raise ValidationError."""
        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError, match="pii_action"):
            MemoryConfig(pii_action="invalid")  # type: ignore[arg-type]

    def test_config_invalid_default_role_rejected(self) -> None:
        """Invalid default_role string must raise ValidationError."""
        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError, match="default_role"):
            MemoryConfig(default_role="superuser")  # type: ignore[arg-type]

    def test_config_valid_pii_actions(self) -> None:
        """All valid pii_action values must be accepted."""
        from trw_memory.models.config import MemoryConfig

        for action in ("block", "redact", "warn"):
            cfg = MemoryConfig(pii_action=action)  # type: ignore[arg-type]
            assert cfg.pii_action == action

    def test_config_valid_default_roles(self) -> None:
        """All valid default_role values must be accepted."""
        from trw_memory.models.config import MemoryConfig

        for role in ("admin", "editor", "viewer", "writer"):
            cfg = MemoryConfig(default_role=role)  # type: ignore[arg-type]
            assert cfg.default_role == role

    def test_config_negative_weight_rejected(self) -> None:
        """Negative score weight must raise ValidationError (ge=0.0)."""
        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError, match="score_relevance_weight"):
            MemoryConfig(
                score_relevance_weight=-0.1,
                score_recency_weight=0.55,
                score_importance_weight=0.55,
            )

    def test_config_weight_above_one_rejected(self) -> None:
        """Score weight > 1.0 must raise ValidationError (le=1.0)."""
        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError, match="score_recency_weight"):
            MemoryConfig(
                score_relevance_weight=0.0,
                score_recency_weight=1.5,
                score_importance_weight=0.0,
            )

    def test_config_valid_weights_accepted(self) -> None:
        """Boundary weights (0.0, 1.0) must be accepted when they sum to 1.0."""
        from trw_memory.models.config import MemoryConfig

        cfg = MemoryConfig(
            score_relevance_weight=1.0,
            score_recency_weight=0.0,
            score_importance_weight=0.0,
        )
        assert cfg.score_relevance_weight == 1.0
        assert cfg.score_recency_weight == 0.0
        assert cfg.score_importance_weight == 0.0


# ---------------------------------------------------------------------------
# FR-02: Source Field Validation
# ---------------------------------------------------------------------------


class TestSourceValidation:
    """Verify source field Literal type and backward-compatible coercion."""

    def test_source_valid_values(self) -> None:
        """All 4 Literal source values must be accepted."""
        from trw_memory.models.memory import MemoryEntry

        for src in ("human", "agent", "tool", "consolidated"):
            entry = MemoryEntry(id="M-001", content="test", source=src)
            assert entry.source == src

    def test_source_unknown_coerced(self) -> None:
        """Unknown source value must be silently coerced to 'agent' for backward compat."""
        from trw_memory.models.memory import MemoryEntry

        entry = MemoryEntry(id="M-001", content="test", source="unknown_origin")
        assert entry.source == "agent"

    def test_source_empty_coerced(self) -> None:
        """Empty source string must be coerced to 'agent'."""
        from trw_memory.models.memory import MemoryEntry

        entry = MemoryEntry(id="M-001", content="test", source="")
        assert entry.source == "agent"

    def test_source_default_is_agent(self) -> None:
        """Default source must be 'agent'."""
        from trw_memory.models.memory import MemoryEntry

        entry = MemoryEntry(id="M-001", content="test")
        assert entry.source == "agent"


# ---------------------------------------------------------------------------
# FR-04: Narrowed auto_recall Exception
# ---------------------------------------------------------------------------


class TestAutoRecallNarrowed:
    """Verify auto_recall catches specific exceptions and fails open."""

    async def test_auto_recall_fails_open_on_storage_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """auto_recall must return empty list when recall raises OSError."""
        from trw_memory.client import MemoryClient

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="default", mode="local")

        @client.auto_recall(query_from="query")
        async def handler(query: str, recalled_memories: list[object] | None = None) -> list[object]:
            return recalled_memories or []

        with patch.object(client, "recall", side_effect=OSError("disk full")):
            result = await handler(query="test")

        assert result == []

    async def test_auto_recall_fails_open_on_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """auto_recall must return empty list when recall raises ValueError."""
        from trw_memory.client import MemoryClient

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="default", mode="local")

        @client.auto_recall(query_from="query")
        async def handler(query: str, recalled_memories: list[object] | None = None) -> list[object]:
            return recalled_memories or []

        with patch.object(client, "recall", side_effect=ValueError("bad query")):
            result = await handler(query="test")

        assert result == []

    async def test_auto_recall_propagates_unexpected_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """auto_recall must NOT catch unexpected exceptions like RuntimeError."""
        from trw_memory.client import MemoryClient

        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="default", mode="local")

        @client.auto_recall(query_from="query")
        async def handler(query: str, recalled_memories: list[object] | None = None) -> list[object]:
            return recalled_memories or []

        with patch.object(client, "recall", side_effect=RuntimeError("unexpected")):
            with pytest.raises(RuntimeError, match="unexpected"):
                await handler(query="test")


# ---------------------------------------------------------------------------
# FR-05: Enum Redundancy Cleanup
# ---------------------------------------------------------------------------


class TestAssertionPatternValidator:
    """Verify pattern validator works with string-normalized enum types."""

    def test_assertion_pattern_validator_with_string_type(self) -> None:
        """Validator must accept string assertion type after use_enum_values normalization."""
        from trw_memory.models.memory import Assertion

        # grep_present requires a non-empty pattern
        a = Assertion(type="grep_present", pattern="import os", target="src/**/*.py")
        assert a.pattern == "import os"

    def test_assertion_pattern_validator_rejects_empty_grep_pattern(self) -> None:
        """grep type with empty pattern must raise ValidationError."""
        from trw_memory.models.memory import Assertion

        with pytest.raises(ValidationError, match="non-empty pattern"):
            Assertion(type="grep_present", pattern="", target="src/**/*.py")

    def test_assertion_pattern_validator_allows_empty_glob_pattern(self) -> None:
        """glob type does not require a pattern — empty string is fine."""
        from trw_memory.models.memory import Assertion

        a = Assertion(type="glob_exists", pattern="", target="src/**/*.py")
        assert a.pattern == ""


# ---------------------------------------------------------------------------
# FR-06: Dimension Mismatch Catch in Tier Scoring
# ---------------------------------------------------------------------------


class TestDimensionMismatchInScoring:
    """Verify scoring gracefully handles dimension mismatches."""

    def test_dimension_mismatch_in_scoring_handled(self) -> None:
        """Mismatched embedding dimensions must not crash — relevance falls to 0.0."""
        from trw_memory.lifecycle.tiers._scoring import compute_importance_score

        entry: dict[str, object] = {
            "id": "M-001",
            "content": "test content",
            "detail": "",
            "importance": 0.5,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Mismatched dimensions: 3 vs 2
        query_embedding = [0.1, 0.2, 0.3]
        entry_embedding = [0.4, 0.5]

        # Should not raise — dimension mismatch is caught
        score = compute_importance_score(
            entry=entry,
            query_tokens=["test"],
            query_embedding=query_embedding,
            entry_embedding=entry_embedding,
        )

        # Score should still be non-negative (recency + importance contribute)
        assert score >= 0.0
        assert score <= 1.0

    def test_zero_division_in_scoring_handled(self) -> None:
        """Zero-magnitude embedding must not crash scoring."""
        from trw_memory.lifecycle.tiers._scoring import compute_importance_score

        entry: dict[str, object] = {
            "id": "M-002",
            "content": "test content",
            "detail": "",
            "importance": 0.5,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Zero vectors cause ZeroDivisionError in cosine similarity
        query_embedding = [0.0, 0.0, 0.0]
        entry_embedding = [0.0, 0.0, 0.0]

        score = compute_importance_score(
            entry=entry,
            query_tokens=["test"],
            query_embedding=query_embedding,
            entry_embedding=entry_embedding,
        )

        assert score >= 0.0
        assert score <= 1.0
