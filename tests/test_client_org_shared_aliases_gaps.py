"""Wave 12: coverage for _client_org_shared_aliases.py static method delegates."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


class TestOrgSharedStaticMethods:
    """Test the static method aliases on OrgSharedAliasMixin via the MemoryClient class."""

    def _get_client_cls(self):
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        return OrgSharedAliasMixin

    def test_shared_result_to_result_converts_dict(self) -> None:
        """_shared_result_to_result delegates to _client_org_shared.shared_result_to_result."""
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        raw = {
            "id": "shared-001",
            "content": "some content",
            "score": 0.85,
            "source": "org",
            "namespace": "org:shared",
        }
        result = OrgSharedAliasMixin._shared_result_to_result(raw)
        # Should return a MemoryResultDict-compatible dict with memory_id
        assert "memory_id" in result
        assert result["content"] == "some content"

    def test_is_retired_shared_result_true_for_obsolete(self) -> None:
        """_is_retired_shared_result returns True for obsolete/deleted entries."""
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        assert OrgSharedAliasMixin._is_retired_shared_result({"status": "obsolete"}) is True
        assert OrgSharedAliasMixin._is_retired_shared_result({"status": "deleted"}) is True

    def test_is_retired_shared_result_false_for_active(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        assert OrgSharedAliasMixin._is_retired_shared_result({"status": "active"}) is False
        assert OrgSharedAliasMixin._is_retired_shared_result({}) is False

    def _make_result(self, content: str = "", detail: str = "", tags: list[str] | None = None) -> dict:
        return {
            "memory_id": "MQ-001",
            "content": content,
            "detail": detail,
            "tags": tags or [],
            "score": 0.9,
            "source": "local",
            "importance": 0.5,
            "created_at": "",
            "updated_at": "",
            "namespace": "default",
        }

    def test_matches_query_true_when_content_contains_query(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        result = self._make_result(content="information about caching strategies")
        assert OrgSharedAliasMixin._matches_query(result, "caching") is True  # type: ignore[arg-type]

    def test_matches_query_false_when_no_match(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        result = self._make_result(content="nothing relevant here")
        assert OrgSharedAliasMixin._matches_query(result, "database migration") is False  # type: ignore[arg-type]

    def test_strip_shared_prefix_removes_lowercase_prefix(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        content = "[shared] important shared knowledge"
        result = OrgSharedAliasMixin._strip_shared_prefix(content)
        assert result == "important shared knowledge"

    def test_strip_shared_prefix_no_prefix_unchanged(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        content = "regular content without prefix"
        result = OrgSharedAliasMixin._strip_shared_prefix(content)
        assert result == content

    async def test_load_entries_for_results_calls_impl(self) -> None:
        """_load_entries_for_results delegates to _client_org_shared."""
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        mock_client = AsyncMock(spec=OrgSharedAliasMixin)
        results = [{"id": "M-load-001", "content": "entry", "score": 0.9, "source": "local"}]

        with patch(
            "trw_memory._client_org_shared.load_entries_for_results",
            new=AsyncMock(return_value=[]),
        ) as mock_impl:
            await OrgSharedAliasMixin._load_entries_for_results(mock_client, results)

        mock_impl.assert_called_once()

    async def test_mark_fetch_retirements_calls_impl(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        mock_client = AsyncMock(spec=OrgSharedAliasMixin)

        with patch(
            "trw_memory._client_org_shared.mark_fetch_retirements",
            new=AsyncMock(return_value=None),
        ) as mock_impl:
            await OrgSharedAliasMixin._mark_fetch_retirements(mock_client, [])

        mock_impl.assert_called_once()

    def test_snapshot_cached_shared_results_calls_impl(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        mock_client = MagicMock(spec=OrgSharedAliasMixin)

        with patch(
            "trw_memory._client_org_shared.snapshot_cached_shared_results",
            return_value=[],
        ) as mock_impl:
            result = OrgSharedAliasMixin._snapshot_cached_shared_results(mock_client, "test query")

        mock_impl.assert_called_once()
        assert result == []

    async def test_dedupe_cached_shared_results_calls_impl(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        mock_client = AsyncMock(spec=OrgSharedAliasMixin)

        with patch(
            "trw_memory._client_org_shared.dedupe_cached_shared_results",
            new=AsyncMock(return_value=[]),
        ) as mock_impl:
            result = await OrgSharedAliasMixin._dedupe_cached_shared_results(
                mock_client,
                [],
                local_entries=[],
                embedder=None,
            )

        mock_impl.assert_called_once()
        assert result == []

    def test_coerce_float_converts_numeric(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        assert OrgSharedAliasMixin._coerce_float(0.5) == 0.5
        assert OrgSharedAliasMixin._coerce_float(1) == 1.0
        assert OrgSharedAliasMixin._coerce_float("0.75") == 0.75
        assert OrgSharedAliasMixin._coerce_float("invalid") == 0.0

    def test_merge_shared_candidates_combines_results(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        def _make_result(mid: str, content: str) -> dict:
            return {
                "memory_id": mid,
                "content": content,
                "detail": "",
                "tags": [],
                "score": 0.9,
                "source": "local",
                "importance": 0.5,
                "created_at": "",
                "updated_at": "",
                "namespace": "default",
            }

        local = [_make_result("L-001", "local content")]
        shared = [_make_result("S-001", "shared content")]
        mock_mixin = MagicMock(spec=OrgSharedAliasMixin)
        result = OrgSharedAliasMixin._merge_shared_candidates(mock_mixin, local, shared)  # type: ignore[arg-type]
        assert len(result) == 2

    async def test_merge_shared_results_calls_impl(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        mock_client = AsyncMock(spec=OrgSharedAliasMixin)

        with patch(
            "trw_memory._client_org_shared.merge_shared_results",
            new=AsyncMock(return_value=[]),
        ) as mock_impl:
            result = await OrgSharedAliasMixin._merge_shared_results(mock_client, "query", [], 5)

        mock_impl.assert_called_once()
        assert result == []
