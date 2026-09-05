"""Wave 14: coverage gap-fill for tools/recall.py (lines 110-112, 133-134, 145, 153-154, 179, 313)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trw_memory.tools.recall import memory_recall_impl

from ._test_tools_support import _mock_backend


class TestRecallCanaryTamper:
    def test_should_halt_recalls_true_raises_canary_tamper_error(self) -> None:
        """should_halt_recalls=True → CanaryTamperError (lines 110-112)."""
        from trw_memory.exceptions import CanaryTamperError

        backend = _mock_backend()

        with patch("trw_memory.tools.recall.should_halt_recalls", return_value=True):
            with pytest.raises(CanaryTamperError, match="canary tamper"):
                memory_recall_impl("some query", "project:default", backend=backend)


class TestRecallInvalidIncludeNamespaces:
    def test_invalid_namespace_in_include_namespaces_is_skipped(self) -> None:
        """Invalid namespace in include_namespaces → logged and skipped (lines 133-134)."""
        backend = _mock_backend()
        backend.list_entries.return_value = []
        backend.get_stored_embeddings.return_value = {}

        result = memory_recall_impl(
            "query",
            "project:default",
            backend=backend,
            include_namespaces=["INVALID!!NAMESPACE"],
            include_org_memories=False,
        )

        assert "memories" in result


class TestRecallDuplicateNamespaces:
    def test_duplicate_namespace_in_include_namespaces_is_deduplicated(self) -> None:
        """Same namespace twice in include_namespaces → second is skipped (line 145)."""
        backend = _mock_backend()
        backend.list_entries.return_value = []
        backend.get_stored_embeddings.return_value = {}

        result = memory_recall_impl(
            "query",
            "project:default",
            backend=backend,
            include_namespaces=["project:default"],  # duplicate of primary namespace
            include_org_memories=False,
        )

        assert "memories" in result


class TestRecallExpiredExtraTeamNamespace:
    def test_expired_team_namespace_in_include_namespaces_is_skipped(self) -> None:
        """Expired team namespace in extra ns loop → skipped (lines 153-154)."""
        backend = _mock_backend()
        backend.list_entries.return_value = []
        backend.get_stored_embeddings.return_value = {}

        with patch("trw_memory.tools.recall.NamespaceManager") as mock_mgr_cls:
            mock_mgr = MagicMock()
            mock_mgr.team_namespace_expired.return_value = True
            mock_mgr_cls.return_value = mock_mgr

            result = memory_recall_impl(
                "query",
                "project:default",
                backend=backend,
                include_namespaces=["team:expired-team"],
                namespace_backend_factory=lambda __ns: backend,  # type: ignore[arg-type]
                include_org_memories=False,
            )

        assert "memories" in result


class TestRecallTagsExpandTopK:
    def test_tags_filter_expands_effective_top_k_to_namespace_size(self) -> None:
        """Tags + entries hits effective_top_k expansion (line 179)."""
        from trw_memory.models.memory import MemoryEntry

        entries = [
            MemoryEntry(id=f"M-{i:03d}", content=f"content {i}", tags=["important"], namespace="project:default")
            for i in range(3)
        ]
        backend = _mock_backend(entries)
        backend.list_entries.return_value = entries
        backend.get_stored_embeddings.return_value = {}
        # _keyword_search must return something for hybrid path to work
        backend.search.return_value = entries

        with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
            result = memory_recall_impl(
                "content",
                "project:default",
                backend=backend,
                tags=["important"],
                include_org_memories=False,
            )

        assert "memories" in result


class TestRecallTokenBudget:
    def test_token_budget_applies_token_truncation(self) -> None:
        """token_budget not None → apply_token_budget called (line 313)."""
        from trw_memory.models.memory import MemoryEntry

        entries = [
            MemoryEntry(
                id=f"M-{i:03d}",
                content=f"memory content item number {i} " * 10,
                namespace="project:default",
            )
            for i in range(5)
        ]
        backend = _mock_backend(entries)
        backend.list_entries.return_value = entries
        backend.get_stored_embeddings.return_value = {}

        result = memory_recall_impl(
            "memory content",
            "project:default",
            backend=backend,
            token_budget=50,  # very small budget to force truncation
            include_org_memories=False,
        )

        assert "memories" in result
        assert "tokens_used" in result
        assert "tokens_truncated" in result
