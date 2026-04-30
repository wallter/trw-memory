from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from trw_memory.exceptions import AuthorizationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.tools.forget import memory_forget_impl
from trw_memory.tools.store import memory_store_impl

from ._test_tools_support import _make_entry, _mock_backend


class TestMemoryForgetImpl:
    def test_no_args_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_forget_impl(None, None, "project:default", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_delete_by_id(self) -> None:
        entry = _make_entry("M-001")
        backend = _mock_backend([entry])
        backend.get.return_value = entry
        backend.delete.return_value = True
        result = memory_forget_impl("M-001", None, "project:default", backend=backend)
        assert result["status"] == "ok"
        assert result["deleted"] == 1

    def test_delete_missing_id(self) -> None:
        backend = _mock_backend()
        backend.get.return_value = None
        backend.delete.return_value = False
        result = memory_forget_impl("M-999", None, "project:default", backend=backend)
        assert result["status"] == "ok"
        assert result["deleted"] == 0

    def test_invalid_namespace_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_forget_impl("M-001", None, "INVALID!!", backend=backend)
        assert result["status"] == "invalid"

    def test_forget_denied_for_reader_namespace_role(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(rbac_enabled=True, namespace_roles={"project:default": "reader"})

        with pytest.raises(
            AuthorizationError,
            match=r"Role 'reader' does not have forget permission on namespace 'project:default'\.",
        ):
            memory_forget_impl("M-001", None, "project:default", backend=backend, config=cfg)

    def test_bulk_delete_via_query(self) -> None:
        entries = [_make_entry("M-001"), _make_entry("M-002")]
        backend = _mock_backend(entries)
        backend.search.return_value = entries
        backend.delete.return_value = True
        result = memory_forget_impl(None, "some query", "project:default", backend=backend)
        assert result["status"] == "ok"
        assert int(str(result["deleted"])) >= 0

    def test_bulk_delete_no_matches(self) -> None:
        backend = _mock_backend([])
        backend.search.return_value = []
        result = memory_forget_impl(None, "nothing matches", "project:default", backend=backend)
        assert result["status"] == "ok"
        assert result["deleted"] == 0

    def test_namespace_isolation_blocks_cross_namespace_delete(self) -> None:
        entry = _make_entry("M-001", namespace="project:repo-a")
        backend = _mock_backend([entry])
        backend.get.return_value = entry
        result = memory_forget_impl("M-001", None, "project:repo-b", backend=backend)
        assert result["status"] == "ok"
        assert result["deleted"] == 0
        backend.delete.assert_not_called()

    def test_empty_memory_id_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_forget_impl("  ", None, "project:default", backend=backend)
        assert result["status"] == "invalid"
        assert "non-empty" in str(result.get("error", "")).lower()

    def test_forget_removes_entry_from_tier_runtime(self, tmp_path: Path) -> None:
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))
        with create_backend_from_config(cfg, "project:default") as backend:
            stored = memory_store_impl("tool tracked entry", "project:default", backend=backend, config=cfg)
            memory_id = cast("str", stored["memory_id"])
            manager = get_tier_manager(cfg, "project:default")

            warm_ids = [str(item["id"]) for item in manager.warm_search(["tracked"], None)]
            assert memory_id in warm_ids

            result = memory_forget_impl(memory_id, None, "project:default", backend=backend, config=cfg)

            assert result["deleted"] == 1
            warm_ids_after = [str(item["id"]) for item in manager.warm_search(["tracked"], None)]
            assert memory_id not in warm_ids_after
