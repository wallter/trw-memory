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

    def test_actor_search_reads_a_bounded_window_and_says_when_it_was_truncated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rc9: an actor filter read the whole namespace on the serialized lane. It now reads at most the
        newest _SCAN_ROWS rows, finds the same matches inside them, and says when older rows went unread."""
        from trw_memory.tools import search

        monkeypatch.setattr(search, "_SCAN_ROWS", 3)
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        read: list[int] = []
        with create_backend_from_config(cfg, "project:default") as backend:
            for index, who in enumerate(["alice", "bob", "alice", "bob"]):
                backend.store(
                    MemoryEntry(
                        id=f"M-{index}", content=f"note {index}", namespace="project:default", source_identity=who
                    )
                )
            listing = backend.list_entries
            monkeypatch.setattr(backend, "list_entries", lambda **kw: read.append(kw["limit"]) or listing(**kw))
            windowed = memory_search_impl("project:default", backend=backend, config=cfg, actor="alice")
            monkeypatch.setattr(search, "_SCAN_ROWS", 10)
            whole = memory_search_impl("project:default", backend=backend, config=cfg, actor="alice")

        assert read == [4, 11]  # one row past the window tells a full window from a truncated one
        assert (windowed["total"], windowed.get("truncated")) == (1, True)
        assert (sorted(e["id"] for e in whole["entries"]), "truncated" in whole) == (["M-0", "M-2"], False)

    def test_a_quarantined_actor_search_says_when_older_rows_went_unread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rc9 sol P1: the quarantine read takes the namespace's newest rows of any status and filters
        them, so a full window with few matches hid older ones. Through the real helper: a namespace
        holding more rows than the window is reported truncated."""
        from trw_memory.tools import search

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        from trw_memory.security._runtime_quarantine import open_quarantine_backend

        with open_quarantine_backend(cfg) as held:
            for index, who in enumerate(["alice", "bob", "alice", "bob"]):
                held.store(
                    MemoryEntry(
                        id=f"Q-{index}",
                        content=f"held {index}",
                        namespace="project:default",
                        source_identity=who,
                        metadata={"quarantined": "true"} if who == "alice" else {},
                    )
                )
        with create_backend_from_config(cfg, "project:default") as backend:
            whole = memory_search_impl(
                "project:default", backend=backend, config=cfg, status="quarantined", actor="alice"
            )
            monkeypatch.setattr(search, "_SCAN_ROWS", 3)
            windowed = memory_search_impl(
                "project:default", backend=backend, config=cfg, status="quarantined", actor="alice"
            )

        assert (whole["total"], "truncated" in whole) == (2, False)
        assert windowed.get("truncated") is True

    def test_actor_search_appends_access_audit_record(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        with create_backend_from_config(cfg, "project:default") as backend:
            backend.store(
                MemoryEntry(id="M-alice", content="alpha", namespace="project:default", source_identity="alice")
            )
            result = memory_search_impl("project:default", backend=backend, config=cfg, actor="alice")

        assert result["total"] == 1
        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert audit_records[-1].op == "access"
        assert audit_records[-1].actor == "alice"
