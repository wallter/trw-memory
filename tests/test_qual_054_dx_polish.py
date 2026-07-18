"""Tests for PRD-QUAL-054: DX polish and quality fixes.

Covers: CLI error boundary, config weight validation, RRF k-guard,
MemoryEntry __repr__, MemoryConfig fsync_on_append, path traversal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

# --- FR02: Config weight validation ---


class TestConfigWeightValidation:
    def test_negative_weight_rejected(self) -> None:
        """MemoryConfig rejects negative score weights."""
        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError):
            MemoryConfig(score_relevance_weight=-0.5)

    def test_weight_above_one_rejected(self) -> None:
        """MemoryConfig rejects weights > 1.0."""
        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError):
            MemoryConfig(score_relevance_weight=1.5)

    def test_valid_weights_accepted(self) -> None:
        """MemoryConfig accepts valid weights that sum to 1.0."""
        from trw_memory.models.config import MemoryConfig

        config = MemoryConfig(
            score_relevance_weight=0.5,
            score_recency_weight=0.3,
            score_importance_weight=0.2,
        )
        assert config.score_relevance_weight == 0.5


# --- FR03: RRF k-parameter guard ---


class TestRrfKGuard:
    def test_k_zero_no_error(self) -> None:
        """rrf_fuse(k=0) does not raise ZeroDivisionError."""
        from trw_memory.retrieval.fusion import rrf_fuse

        rankings = [[("a", 1.0), ("b", 0.5)]]
        result = rrf_fuse(rankings, k=0)
        assert len(result) > 0

    def test_k_negative_resets_to_default(self) -> None:
        """rrf_fuse(k=-1) resets to 5 (tuned default) and produces valid results."""
        from trw_memory.retrieval.fusion import rrf_fuse

        rankings = [[("a", 1.0), ("b", 0.5)]]
        result_neg = rrf_fuse(rankings, k=-1)
        result_default = rrf_fuse(rankings, k=5)
        assert result_neg == result_default

    def test_k_default_unchanged(self) -> None:
        """rrf_fuse() with default k=5 (tuned) produces expected results."""
        from trw_memory.retrieval.fusion import rrf_fuse

        rankings = [[("a", 1.0)]]
        result = rrf_fuse(rankings)
        assert len(result) == 1
        assert result[0][0] == "a"


# --- FR05: fsync_on_append config ---


class TestFsyncOnAppend:
    def test_config_field_default_false(self) -> None:
        """fsync_on_append defaults to False."""
        from trw_memory.models.config import MemoryConfig

        config = MemoryConfig()
        assert config.fsync_on_append is False

    def test_config_field_set_true(self) -> None:
        """fsync_on_append can be set to True."""
        from trw_memory.models.config import MemoryConfig

        config = MemoryConfig(fsync_on_append=True)
        assert config.fsync_on_append is True

    def test_audit_log_fsync_wired(self, tmp_path: Path) -> None:
        """AuditLog with fsync=True calls os.fsync."""
        from unittest.mock import patch

        from trw_memory.security.audit import AuditLog

        log = AuditLog(tmp_path / "audit.jsonl", fsync=True)
        with patch("os.fsync") as mock_fsync:
            log.append(action="test", target_id="M-test")
            mock_fsync.assert_called_once()
        assert '"id":"M-test"' in (tmp_path / "audit.jsonl").read_text()


# --- FR07: __repr__ ---


class TestRepr:
    def test_memory_config_repr(self) -> None:
        """MemoryConfig repr includes key settings."""
        from trw_memory.models.config import MemoryConfig

        config = MemoryConfig()
        r = repr(config)
        assert "backend=" in r
        assert "path=" in r

    def test_memory_entry_repr(self) -> None:
        """MemoryEntry repr includes id, content preview, tags, importance."""
        from trw_memory.models.memory import MemoryEntry

        entry = MemoryEntry(
            id="M-test123",
            content="This is a test learning about caching strategies",
            tags=["caching", "perf"],
            importance=0.85,
        )
        r = repr(entry)
        assert "M-test123" in r
        assert "caching" in r
        assert "0.85" in r

    def test_memory_entry_repr_long_content_truncated(self) -> None:
        """MemoryEntry repr truncates content longer than 40 chars."""
        from trw_memory.models.memory import MemoryEntry

        long_content = "A" * 80
        entry = MemoryEntry(id="M-long", content=long_content)
        r = repr(entry)
        assert "..." in r
        assert len(r) < len(long_content) + 100


# --- FR04: Lossless export ---


class TestLosslessExport:
    def test_to_dict_includes_all_fields(self) -> None:
        """MemoryEntry.to_dict() includes all model fields."""
        from trw_memory.models.memory import MemoryEntry

        entry = MemoryEntry(id="M-export", content="test content")
        exported = entry.to_dict()
        model_fields = set(MemoryEntry.model_fields.keys())
        exported_keys = set(exported.keys())
        assert model_fields == exported_keys, f"Missing: {model_fields - exported_keys}"
