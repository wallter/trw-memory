"""Wave 12: coverage for _client_org_shared_aliases.py static method delegates."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


class TestOrgSharedStaticMethods:
    """Test the static method aliases on OrgSharedAliasMixin via the MemoryClient class."""

    def _get_client_cls(self):
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        return OrgSharedAliasMixin

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

    def test_coerce_float_converts_numeric(self) -> None:
        from trw_memory._client_org_shared_aliases import OrgSharedAliasMixin

        assert OrgSharedAliasMixin._coerce_float(0.5) == 0.5
        assert OrgSharedAliasMixin._coerce_float(1) == 1.0
        assert OrgSharedAliasMixin._coerce_float("0.75") == 0.75
        assert OrgSharedAliasMixin._coerce_float("invalid") == 0.0

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
