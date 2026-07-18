"""Wave 12: direct unit tests for memory_audit_impl and memory_review_impl."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from trw_memory.tools.audit import memory_audit_impl
from trw_memory.tools.review import memory_review_impl


def _ctx_factory(backend):
    @contextmanager
    def _cm(*args, **kwargs):
        yield backend

    return _cm


class TestMemoryAuditImpl:
    def test_calls_audit_entry_with_backend(self) -> None:
        mock_backend = MagicMock()
        expected = {"learning_id": "L-001", "events": []}

        with (
            patch(
                "trw_memory.integrations._backend.create_backend_from_config",
                new=_ctx_factory(mock_backend),
            ),
            patch(
                "trw_memory.tools.audit.audit_entry",
                return_value=expected,
            ) as mock_audit,
        ):
            result = memory_audit_impl("L-001")

        mock_audit.assert_called_once()
        assert result == expected

    def test_accepts_explicit_config(self) -> None:
        from trw_memory.models.config import MemoryConfig

        mock_backend = MagicMock()
        cfg = MemoryConfig()

        with (
            patch(
                "trw_memory.integrations._backend.create_backend_from_config",
                new=_ctx_factory(mock_backend),
            ),
            patch("trw_memory.tools.audit.audit_entry", return_value={"ok": True}),
        ):
            result = memory_audit_impl("L-002", config=cfg)

        assert result == {"ok": True}


class TestMemoryReviewImpl:
    def test_calls_review_quarantined_with_backend(self) -> None:
        mock_backend = MagicMock()
        expected = {"status": "approved"}

        with (
            patch(
                "trw_memory.integrations._backend.create_backend_from_config",
                new=_ctx_factory(mock_backend),
            ),
            patch(
                "trw_memory.tools.review.review_quarantined_entry",
                return_value=expected,
            ) as mock_review,
        ):
            result = memory_review_impl("L-003", decision="approve", reviewer_id="rev-1")

        mock_review.assert_called_once()
        assert result == expected

    def test_reject_decision_passed_through(self) -> None:
        mock_backend = MagicMock()

        with (
            patch(
                "trw_memory.integrations._backend.create_backend_from_config",
                new=_ctx_factory(mock_backend),
            ),
            patch(
                "trw_memory.tools.review.review_quarantined_entry",
                return_value={"status": "rejected"},
            ) as mock_review,
        ):
            result = memory_review_impl("L-004", decision="reject", reviewer_id="rev-2")

        assert result == {"status": "rejected"}


class TestRegisterAuditTool:
    def test_register_calls_mcp_tool_decorator(self) -> None:
        from trw_memory.tools.audit import register_audit_tool

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda f: f

        register_audit_tool(mock_mcp)

        mock_mcp.tool.assert_called_once()
        assert mock_mcp.tool.call_args.args == ()

    async def test_registered_function_delegates_to_impl(self) -> None:
        from trw_memory.tools.audit import register_audit_tool

        registered = {}
        mock_mcp = MagicMock()

        def capture_decorator(f):
            registered["fn"] = f
            return f

        mock_mcp.tool.return_value = capture_decorator
        register_audit_tool(mock_mcp)

        mock_backend = MagicMock()
        expected = {"learning_id": "L-005", "events": []}

        with (
            patch(
                "trw_memory.integrations._backend.create_backend_from_config",
                new=_ctx_factory(mock_backend),
            ),
            patch("trw_memory.tools.audit.audit_entry", return_value=expected),
        ):
            result = await registered["fn"]("L-005")

        assert result == expected


class TestRegisterReviewTool:
    def test_register_calls_mcp_tool_decorator(self) -> None:
        from trw_memory.tools.review import register_review_tool

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda f: f

        register_review_tool(mock_mcp)

        mock_mcp.tool.assert_called_once()
        assert mock_mcp.tool.call_args.args == ()

    async def test_registered_function_delegates_to_impl(self) -> None:
        from trw_memory.tools.review import register_review_tool

        registered = {}
        mock_mcp = MagicMock()

        def capture_decorator(f):
            registered["fn"] = f
            return f

        mock_mcp.tool.return_value = capture_decorator
        register_review_tool(mock_mcp)

        mock_backend = MagicMock()
        expected = {"status": "approved"}

        with (
            patch(
                "trw_memory.integrations._backend.create_backend_from_config",
                new=_ctx_factory(mock_backend),
            ),
            patch("trw_memory.tools.review.review_quarantined_entry", return_value=expected),
        ):
            result = await registered["fn"]("L-006", "approve", "rev-3")

        assert result == expected
