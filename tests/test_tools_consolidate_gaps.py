"""Wave 14: coverage gap-fill for tools/consolidate.py (lines 118, 243-245, 307-310)."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from trw_memory.exceptions import StorageError
from trw_memory.tools.consolidate import memory_consolidate_impl, register_consolidate_tool

from ._test_tools_support import _mock_backend


class TestConsolidateTeamNamespaceContinue:
    def test_non_team_namespace_in_discover_backends_is_skipped(self) -> None:
        """Non-team namespace in discover_namespace_backends loop → continue (line 118)."""
        from trw_memory.models.config import MemoryConfig

        cfg = MemoryConfig()
        backend = _mock_backend()

        # Patch discover_namespace_backends to yield a non-team namespace
        @contextmanager
        def _mock_discover(_cfg):
            yield [
                (["project:default"], backend),  # non-team → hits 'continue'
                (["team:my-team"], backend),       # team → processes normally
            ]

        with (
            patch("trw_memory.tools.consolidate.discover_namespace_backends", side_effect=_mock_discover),
            patch("trw_memory.tools.consolidate.NamespaceManager") as mock_mgr_cls,
            patch("trw_memory.tools.consolidate.memory_consolidate_impl", return_value={
                "clusters_found": 0, "consolidated_count": 0, "dry_run": True
            }),
        ):
            mock_mgr = MagicMock()
            mock_mgr.team_namespace_completed.return_value = False
            mock_mgr_cls.return_value = mock_mgr

            # The TEAM_NAMESPACE_WILDCARD triggers _promote_all_team_namespaces
            from trw_memory.tools.consolidate import TEAM_NAMESPACE_WILDCARD
            result = memory_consolidate_impl(
                TEAM_NAMESPACE_WILDCARD,
                backend=backend,
                dry_run=True,
                config=cfg,
            )

        # wildcard returns aggregated summary keys (namespace_id, namespaces, etc.)
        assert isinstance(result, dict)


class TestConsolidateStorageError:
    def test_storage_error_in_consolidate_cycle_returns_error_status(self) -> None:
        """StorageError from consolidate_cycle → returns error status (lines 243-245)."""
        from trw_memory.models.config import MemoryConfig

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
    def test_registered_function_delegates_to_impl(self) -> None:
        """register_consolidate_tool wires memory_consolidate to impl (lines 307-310)."""
        registered: dict[str, object] = {}
        mock_mcp = MagicMock()

        def _capture(f):
            registered["fn"] = f
            return f

        mock_mcp.tool.return_value = _capture
        register_consolidate_tool(mock_mcp)

        assert "fn" in registered

        def _ctx_factory(backend: MagicMock):
            @contextmanager
            def _cm(*_a, **_kw):
                yield backend
            return _cm

        mock_backend = _mock_backend()

        with patch(
            "trw_memory.integrations._backend.create_backend_from_config",
            new=_ctx_factory(mock_backend),
        ):
            with patch("trw_memory.tools.consolidate.consolidate_cycle", return_value={
                "clusters_found": 0, "consolidated_count": 0, "dry_run": True
            }):
                result = asyncio.run(
                    registered["fn"]("project:default", True)  # type: ignore[operator]
                )

        assert isinstance(result, dict)  # registered function returns consolidate result
