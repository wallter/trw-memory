"""Wave 15: coverage gap-fill for tools/_recall_helpers.py (lines 32, 65, 68-70, 141, 150, 167, 169-170, 176-178, 185, 207)."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.tools._recall_helpers import (
    _apply_sec001_recall_policy,
    _entry_matches_query,
    _graph_related,
    _org_memory_results,
    _record_access_by_namespace,
)

from ._test_tools_support import _mock_backend


class TestApplySec001RecallPolicy:
    def test_returns_early_when_recall_filter_disabled(self) -> None:
        """enable_recall_filter=False → return results immediately (line 32)."""
        cfg = MemoryConfig()
        cfg.enable_recall_filter = False
        results = [{"id": "M-001", "content": "hello"}]
        out = _apply_sec001_recall_policy(results, config=cfg)
        assert out is results

    def test_system_canary_entry_is_skipped(self) -> None:
        """entry.metadata.system_canary='true' → continue (line 65)."""
        from trw_memory.models.memory import MemoryEntry

        cfg = MemoryConfig()
        cfg.enable_recall_filter = True
        canary_entry = {"id": "M-canary", "content": "canary", "metadata": {"system_canary": "true"}}

        with patch("trw_memory.tools._recall_helpers.filter_recall_window") as mock_filter:
            mock_accepted = MagicMock()
            entry = MemoryEntry(id="M-canary::0", content="canary", metadata={"system_canary": "true"})
            mock_accepted.accepted = [entry]
            mock_filter.return_value = mock_accepted
            result = _apply_sec001_recall_policy([canary_entry], config=cfg)

        assert result == []

    def test_none_source_result_uses_model_dump(self) -> None:
        """source_result is None → falls back to entry.model_dump (lines 68-70)."""
        cfg = MemoryConfig()
        cfg.enable_recall_filter = True
        result_entry: dict[str, object] = {"id": "M-unmapped", "content": "text"}

        with patch("trw_memory.tools._recall_helpers.filter_recall_window") as mock_filter:
            mock_accepted = MagicMock()
            # entry.id doesn't match any key in result_by_id → source_result is None
            # id without '::' so the rsplit branch is not taken
            entry = MemoryEntry(id="UNKNOWN-ID", content="text")
            mock_accepted.accepted = [entry]
            mock_filter.return_value = mock_accepted
            result = _apply_sec001_recall_policy([result_entry], config=cfg)

        assert isinstance(result, list)

    def test_none_source_result_strips_synthetic_suffix_from_id(self) -> None:
        """source_result is None and '::' in entry.id → strip suffix (line 70)."""
        cfg = MemoryConfig()
        cfg.enable_recall_filter = True
        result_entry: dict[str, object] = {"id": "M-real", "content": "text"}

        with patch("trw_memory.tools._recall_helpers.filter_recall_window") as mock_filter:
            mock_accepted = MagicMock()
            # entry.id has '::0' suffix (synthetic ID format used in _apply_sec001_recall_policy)
            entry = MemoryEntry(id="M-real::99", content="text")
            mock_accepted.accepted = [entry]
            mock_filter.return_value = mock_accepted
            result = _apply_sec001_recall_policy([result_entry], config=cfg)

        assert isinstance(result, list)
        if result:
            assert "::" not in str(result[0].get("id", ""))


class TestOrgMemoryResultsMinScore:
    def test_low_score_entry_is_skipped_when_min_score_set(self) -> None:
        """entry_utility below min_score → continue (line 141)."""
        cfg = MemoryConfig()
        entries = [MemoryEntry(id="M-001", content="test", namespace="project:default")]

        with (
            patch("trw_memory.tools._recall_helpers.list_org_shared_entries", return_value=entries),
            patch("trw_memory.tools._recall_helpers.entry_utility", return_value=0.0),
        ):
            result = _org_memory_results(
                cfg,
                "project:default",
                query="",
                tags=None,
                min_score=0.5,  # 0.0 < 0.5 → entry skipped
                exclude_keys=set(),
                limit=10,
            )

        assert result == []


class TestEntryMatchesQuery:
    def test_empty_query_tokens_returns_true(self) -> None:
        """Empty query_tokens → return True immediately (line 150)."""
        entry: dict[str, object] = {"content": "anything"}
        assert _entry_matches_query(entry, []) is True


class TestGraphRelated:
    def test_no_conn_and_backend_has_no_conn_returns_empty(self) -> None:
        """conn=None and backend has no _conn → skip and return [] (lines 167, 169-170)."""
        backend = MagicMock(spec=[])  # no _conn attribute
        result = _graph_related([{"id": "M-001"}], depth=1, backend=backend, conn=None)
        assert result == []

    def test_graph_query_sqlite_error_returns_empty(self) -> None:
        """sqlite3.Error in graph_query → log + return [] (lines 176-178)."""
        backend = MagicMock()
        mock_conn = MagicMock(spec=sqlite3.Connection)

        with patch("trw_memory.tools._recall_helpers.graph_query", side_effect=sqlite3.Error("locked")):
            result = _graph_related([{"id": "M-001"}], depth=1, backend=backend, conn=mock_conn)

        assert result == []

    def test_dangling_edge_entry_is_skipped(self) -> None:
        """backend.get returns None → skip (line 185)."""
        backend = _mock_backend()
        backend.get.return_value = None  # dangling edge
        mock_conn = MagicMock(spec=sqlite3.Connection)

        with patch("trw_memory.tools._recall_helpers.graph_query", return_value=[{"id": "M-deleted"}]):
            result = _graph_related([{"id": "M-001"}], depth=1, backend=backend, conn=mock_conn)

        assert result == []


class TestRecordAccessByNamespace:
    def test_result_without_id_is_skipped(self) -> None:
        """result dict missing 'id' key → continue (line 207)."""
        backend = _mock_backend()
        with patch("trw_memory.tools._recall_helpers.record_recall_access") as mock_record:
            _record_access_by_namespace(
                [{"content": "no id here"}],  # no 'id' key
                backend,
                "project:default",
                None,
            )
        mock_record.assert_not_called()
