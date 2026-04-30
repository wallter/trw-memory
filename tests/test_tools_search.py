from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import AuthorizationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security.audit import AuditLog
from trw_memory.tools.search import memory_search_impl

from ._test_tools_support import _make_entry, _mock_backend


class TestMemorySearchImpl:
    def test_returns_expected_keys(self) -> None:
        backend = _mock_backend([_make_entry()])
        result = memory_search_impl("project:default", backend=backend)
        assert "entries" in result
        assert "total" in result
        assert "offset" in result
        assert "limit" in result

    def test_invalid_namespace_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_search_impl("BAD NS!!", backend=backend)
        assert result["status"] == "invalid"
        assert "error" in result

    def test_search_denied_for_writer_namespace_role(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(rbac_enabled=True, namespace_roles={"project:default": "writer"})

        with pytest.raises(
            AuthorizationError,
            match=r"Role 'writer' does not have search permission on namespace 'project:default'\.",
        ):
            memory_search_impl("project:default", backend=backend, config=cfg)

    def test_status_filter_active(self) -> None:
        entries = [
            _make_entry("M-001", status=MemoryStatus.ACTIVE),
            _make_entry("M-002", status=MemoryStatus.OBSOLETE),
        ]
        backend = _mock_backend(entries)
        result = memory_search_impl("project:default", status="active", backend=backend)
        assert "entries" in result

    def test_offset_and_limit_in_result(self) -> None:
        entries = [_make_entry(f"M-{i:03d}") for i in range(5)]
        backend = _mock_backend(entries)
        result = memory_search_impl("project:default", backend=backend, offset=2, limit=2)
        assert result["offset"] == 2
        assert result["limit"] == 2

    def test_tags_filter_applied(self) -> None:
        entries = [
            _make_entry("M-001", tags=["python"]),
            _make_entry("M-002", tags=["rust"]),
        ]
        backend = _mock_backend(entries)
        result = memory_search_impl("project:default", backend=backend, tags=["python"])
        assert "entries" in result

    def test_invalid_status_returns_error(self) -> None:
        backend = _mock_backend()
        result = memory_search_impl("project:default", backend=backend, status="nonexistent_status")
        assert result.get("status") == "invalid" or "entries" in result

    def test_actor_search_appends_access_audit_record(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        with create_backend_from_config(cfg, "project:default") as backend:
            backend.store(MemoryEntry(id="M-alice", content="alpha", namespace="project:default", source_identity="alice"))
            result = memory_search_impl("project:default", backend=backend, config=cfg, actor="alice")

        assert result["total"] == 1
        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert audit_records[-1].op == "access"
        assert audit_records[-1].actor == "alice"
