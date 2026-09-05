"""Wave 14: coverage gap-fill for tools/consolidate.py (lines 118, 243-245, 307-310)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import ANY, MagicMock, patch

import pytest

from trw_memory.exceptions import AuthorizationError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.tools.consolidate import TEAM_NAMESPACE_WILDCARD, memory_consolidate_impl, register_consolidate_tool

from ._test_tools_support import _mock_backend


class TestConsolidateTeamNamespaceContinue:
    def test_non_team_namespace_in_discover_backends_is_skipped(self) -> None:
        """Non-team namespace in discover_namespace_backends loop → continue (line 118)."""
        cfg = MemoryConfig()
        backend = _mock_backend()

        # Patch discover_namespace_backends to yield a non-team namespace
        @contextmanager
        def _mock_discover(_cfg):
            yield [
                (["project:default"], backend),  # non-team → hits 'continue'
                (["team:my-team"], backend),  # team → processes normally
            ]

        with (
            patch("trw_memory.tools.consolidate.discover_namespace_backends", side_effect=_mock_discover),
            patch("trw_memory.tools.consolidate.NamespaceManager") as mock_mgr_cls,
            patch(
                "trw_memory.tools.consolidate._promote_team_memories",
                return_value={
                    "promoted_count": 2,
                    "discarded_count": 1,
                    "namespace_id": "team:my-team",
                    "completed_at": "now",
                },
            ) as promote,
        ):
            mock_mgr = MagicMock()
            mock_mgr.team_namespace_completed.return_value = False
            mock_mgr_cls.return_value = mock_mgr

            result = memory_consolidate_impl(
                TEAM_NAMESPACE_WILDCARD,
                backend=backend,
                dry_run=True,
                config=cfg,
            )

        promote.assert_called_once_with("team:my-team", backend, target_backend=backend)
        assert result["promoted_count"] == 2
        assert [item["namespace_id"] for item in result["namespaces"]] == ["team:my-team"]

    def test_backend_factory_failure_isolated_per_team_namespace(self) -> None:
        cfg = MemoryConfig()
        source_backend = _mock_backend()
        target_backend = _mock_backend()
        target_context = MagicMock()
        target_context.__enter__.return_value = target_backend
        target_context.__exit__.return_value = False

        @contextmanager
        def _mock_discover(_cfg):
            yield [(["team:a", "team:b"], source_backend)]

        factory = MagicMock(side_effect=[StorageError("open failed"), target_context])

        def _summary(namespace: str, *_args, **_kwargs) -> dict[str, object]:
            return {
                "promoted_count": 1,
                "discarded_count": 0,
                "namespace_id": namespace,
                "completed_at": "now",
            }

        with (
            patch("trw_memory.tools.consolidate.discover_namespace_backends", side_effect=_mock_discover),
            patch("trw_memory.tools.consolidate.NamespaceManager") as manager_cls,
            patch("trw_memory.tools.consolidate._promote_team_memories", side_effect=_summary),
        ):
            manager_cls.return_value.team_namespace_completed.return_value = False
            result = memory_consolidate_impl(
                TEAM_NAMESPACE_WILDCARD,
                backend=source_backend,
                config=cfg,
                namespace_backend_factory=factory,
            )

        assert result["status"] == "partial"
        assert [item["namespace_id"] for item in result["namespaces"]] == ["team:b"]
        assert result["errors"] == [{"namespace": "team:a", "error": "open failed"}]

    def test_backend_close_failure_isolated_per_team_namespace(self) -> None:
        cfg = MemoryConfig()
        source_backend = _mock_backend()
        bad_context = MagicMock()
        bad_context.__enter__.return_value = _mock_backend()
        bad_context.__exit__.side_effect = StorageError("close failed")
        good_context = MagicMock()
        good_context.__enter__.return_value = _mock_backend()
        good_context.__exit__.return_value = False

        @contextmanager
        def _mock_discover(_cfg):
            yield [(["team:a", "team:b"], source_backend)]

        factory = MagicMock(side_effect=[bad_context, good_context])

        def _summary(namespace: str, *_args, **_kwargs) -> dict[str, object]:
            return {
                "promoted_count": 1,
                "discarded_count": 0,
                "namespace_id": namespace,
                "completed_at": "now",
            }

        with (
            patch("trw_memory.tools.consolidate.discover_namespace_backends", side_effect=_mock_discover),
            patch("trw_memory.tools.consolidate.NamespaceManager") as manager_cls,
            patch("trw_memory.tools.consolidate._promote_team_memories", side_effect=_summary),
        ):
            manager_cls.return_value.team_namespace_completed.return_value = False
            result = memory_consolidate_impl(
                TEAM_NAMESPACE_WILDCARD,
                backend=source_backend,
                config=cfg,
                namespace_backend_factory=factory,
            )

        assert result["status"] == "partial"
        assert result["promoted_count"] == 2
        assert [item["namespace_id"] for item in result["namespaces"]] == ["team:a", "team:b"]
        assert result["errors"] == [{"namespace": "team:a", "error": "close failed"}]


class TestConsolidatePermissions:
    def test_team_promotion_requires_project_write_permission(self) -> None:
        cfg = MemoryConfig(
            rbac_enabled=True,
            namespace_roles={"team:a": "writer", "project:default": "reader"},
            default_role="none",
        )
        factory = MagicMock()

        with pytest.raises(AuthorizationError, match="promote permission"):
            memory_consolidate_impl(
                "team:a",
                backend=_mock_backend(),
                config=cfg,
                namespace_backend_factory=factory,
            )

        factory.assert_not_called()

    @pytest.mark.parametrize(
        ("namespace_roles", "operation"),
        [
            ({"team:a": "reader", "project:default": "writer"}, "consolidate"),
            ({"team:a": "writer", "project:default": "reader"}, "promote"),
        ],
    )
    def test_wildcard_reports_per_team_permission_denial(
        self,
        namespace_roles: dict[str, str],
        operation: str,
    ) -> None:
        cfg = MemoryConfig(
            rbac_enabled=True,
            namespace_roles=namespace_roles,
            default_role="none",
        )
        backend = _mock_backend()

        @contextmanager
        def _mock_discover(_cfg):
            yield [(["team:a"], backend)]

        with patch("trw_memory.tools.consolidate.discover_namespace_backends", side_effect=_mock_discover):
            result = memory_consolidate_impl(TEAM_NAMESPACE_WILDCARD, backend=backend, config=cfg)

        assert result["status"] == "error"
        assert result["promoted_count"] == 0
        assert len(result["errors"]) == 1
        assert f"{operation} permission" in result["errors"][0]["error"]


class TestConsolidateStorageError:
    def test_storage_error_in_consolidate_cycle_returns_error_status(self) -> None:
        """StorageError from consolidate_cycle → returns error status (lines 243-245)."""
        backend = _mock_backend()
        cfg = MemoryConfig()

        with patch(
            "trw_memory.tools.consolidate.consolidate_cycle",
            side_effect=StorageError("disk full", path="/tmp/test.db"),
        ):
            result = memory_consolidate_impl(
                "project:default",
                backend=backend,
                dry_run=False,
                config=cfg,
            )

        assert result["status"] == "error"
        assert "consolidation error" in str(result.get("error", ""))


class TestRegisterConsolidateTool:
    async def test_registered_function_delegates_to_impl(self) -> None:
        """register_consolidate_tool wires memory_consolidate to impl (lines 307-310)."""
        registered: dict[str, object] = {}
        mock_mcp = MagicMock()

        def _capture(f, **_kwargs):
            registered["fn"] = f
            return f

        mock_backend = _mock_backend()
        backend_context = MagicMock()
        backend_context.__enter__.return_value = mock_backend
        backend_context.__exit__.return_value = False
        expected = {"clusters_found": 0, "entries_consolidated": 0, "dry_run": True}
        mock_mcp.tool.return_value = _capture

        with (
            patch("trw_memory.integrations._backend.create_backend_from_config", return_value=backend_context),
            patch("trw_memory.tools.consolidate.memory_consolidate_impl", return_value=expected) as impl,
        ):
            register_consolidate_tool(mock_mcp)
            assert "fn" in registered
            result = await registered["fn"]("project:default", True)  # type: ignore[operator]

        assert result == expected
        impl.assert_called_once_with(
            "project:default",
            backend=mock_backend,
            dry_run=True,
            config=ANY,
            namespace_backend_factory=ANY,
        )

    async def test_registered_wildcard_does_not_open_unused_default_backend(self) -> None:
        registered: dict[str, object] = {}
        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda fn: registered.setdefault("fn", fn)
        expected = {"namespace_id": TEAM_NAMESPACE_WILDCARD, "status": "skipped"}

        with (
            patch("trw_memory.integrations._backend.create_backend_from_config") as create_backend,
            patch("trw_memory.tools.consolidate._promote_all_team_namespaces", return_value=expected) as promote_all,
        ):
            register_consolidate_tool(mock_mcp)
            result = await registered["fn"](TEAM_NAMESPACE_WILDCARD, False)  # type: ignore[operator]

        assert result == expected
        create_backend.assert_not_called()
        promote_all.assert_called_once_with(ANY, namespace_backend_factory=ANY)
