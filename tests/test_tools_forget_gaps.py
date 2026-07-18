"""Wave 13: coverage gap-fill for tools/forget.py (lines 99-100, 126-128, 136-138, 185-187)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from trw_memory.exceptions import StorageError
from trw_memory.tools.forget import memory_forget_impl, register_forget_tool

from ._test_tools_support import _make_entry, _mock_backend


class TestForgetStorageErrorPaths:
    def test_storage_error_on_get_logs_warning_and_returns_not_found(self) -> None:
        """StorageError during backend.get() → warning logged, nothing deleted.

        trw-memory-5: a 0-delete now surfaces as not_found rather than a silent
        ok, so the caller can tell nothing was removed.
        """
        backend = _mock_backend()
        backend.get.side_effect = StorageError("disk error", path="/tmp/test.db")

        result = memory_forget_impl("M-001", None, "project:default", backend=backend)

        assert result["status"] == "not_found"
        assert result["deleted"] == 0

    def test_storage_error_on_search_returns_ok_with_zero_deleted(self) -> None:
        """StorageError during backend.search() → warning logged, returns ok (lines 126-128)."""
        backend = _mock_backend()
        backend.search.side_effect = StorageError("index corrupted", path="/tmp/test.db")

        result = memory_forget_impl(None, "some query", "project:default", backend=backend)

        assert result["status"] == "ok"
        assert result["deleted"] == 0

    def test_storage_error_on_bulk_delete_logs_and_continues(self) -> None:
        """StorageError on individual entry.delete() during bulk → log+continue (lines 136-138)."""
        entries = [_make_entry("M-001"), _make_entry("M-002")]
        backend = _mock_backend(entries)
        backend.search.return_value = entries
        backend.delete.side_effect = StorageError("locked", path="/tmp/test.db")

        result = memory_forget_impl(None, "some query", "project:default", backend=backend)

        assert result["status"] == "ok"
        assert result["deleted"] == 0


class TestBulkDeleteWithTierRuntime:
    def test_bulk_delete_calls_remove_entry_from_tiers_when_supported(self) -> None:
        """Bulk delete triggers remove_entry_from_tiers when tier runtime is supported (line 136)."""
        from unittest.mock import patch as _patch

        entries = [_make_entry("M-001"), _make_entry("M-002")]
        backend = _mock_backend(entries)
        backend.search.return_value = entries
        backend.delete.return_value = True

        with (
            _patch("trw_memory.tools.forget.supports_tier_runtime", return_value=True),
            _patch("trw_memory.tools.forget.remove_entry_from_tiers") as mock_remove,
        ):
            result = memory_forget_impl(None, "some query", "project:default", backend=backend)

        assert result["status"] == "ok"
        assert result["deleted"] == 2
        assert mock_remove.call_count == 2


class TestRegisterForgetTool:
    async def test_registered_function_delegates_to_impl(self) -> None:
        """register_forget_tool wires memory_forget to memory_forget_impl (lines 185-187)."""
        mock_backend = _mock_backend()
        mock_backend.get.return_value = None
        mock_backend.delete.return_value = False

        def _ctx_factory(backend: MagicMock):
            @contextmanager
            def _cm(*_a, **_kw):
                yield backend

            return _cm

        registered: dict[str, object] = {}
        mock_mcp = MagicMock()

        def _capture(f):
            registered["fn"] = f
            return f

        mock_mcp.tool.return_value = _capture
        register_forget_tool(mock_mcp)

        assert "fn" in registered

        with patch(
            "trw_memory.integrations._backend.create_backend_from_config",
            new=_ctx_factory(mock_backend),
        ):
            result = await registered["fn"]("M-missing", None, "project:default")  # type: ignore[operator]

        assert result["status"] == "not_found"
