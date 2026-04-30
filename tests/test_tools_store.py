from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from trw_memory.exceptions import AuthorizationError, PIIBlockError, SchemaValidationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.audit import AuditLog
from trw_memory.tools.consolidate import memory_consolidate_impl
from trw_memory.tools.forget import memory_forget_impl
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.search import memory_search_impl
from trw_memory.tools.store import memory_store_impl

from ._test_tools_support import _make_entry, _mock_backend


class TestMemoryStoreImpl:
    def test_store_denied_for_reader_namespace_role(self) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(rbac_enabled=True, namespace_roles={"project:default": "reader"})

        with pytest.raises(
            AuthorizationError,
            match=r"Role 'reader' does not have store permission on namespace 'project:default'\.",
        ):
            memory_store_impl("blocked write", "project:default", backend=backend, config=cfg)

    def test_store_blocks_api_keys_before_backend_write(self, tmp_path: Path) -> None:
        backend = _mock_backend()
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        result = memory_store_impl(
            "api token sk-abcdefghijklmnopqrstuvwxyz",
            "project:default",
            backend=backend,
            config=cfg,
        )

        assert result["status"] == "blocked"
        assert backend.store.called is False
        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert audit_records[-1].op == "store_rejected"
        assert audit_records[-1].data["reason"] == "pii_detected"

    def test_store_quarantines_anomalous_entries(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), poisoning_z_threshold=1.0)

        with create_backend_from_config(cfg, "project:default") as backend:
            for index in range(20):
                backend.store(
                    MemoryEntry(
                        id=f"M-seed-{index}",
                        content="normal content",
                        namespace="project:default",
                        source_identity="seed",
                    )
                )

            result = memory_store_impl(
                "A" * 5000,
                "project:default",
                backend=backend,
                config=cfg,
                source_identity="alice",
            )
            quarantined = memory_search_impl(
                "project:default",
                backend=backend,
                config=cfg,
                status="quarantined",
                actor="alice",
            )

        assert result["status"] == "quarantined"
        assert quarantined["total"] == 1
        entries = cast("list[dict[str, object]]", quarantined["entries"])
        assert entries[0]["metadata"]["quarantined"] == "true"

    def test_forget_actor_deletes_matching_entries_and_audits(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        with create_backend_from_config(cfg, "project:default") as backend:
            backend.store(MemoryEntry(id="M-alice", content="alice entry", namespace="project:default", source_identity="alice"))
            backend.store(MemoryEntry(id="M-bob", content="bob entry", namespace="project:default", source_identity="bob"))

            result = memory_forget_impl(None, None, "project:default", backend=backend, config=cfg, actor="alice")
            remaining = memory_search_impl("project:default", backend=backend, config=cfg, actor="alice")

        assert result["entries_deleted"] == 1
        assert remaining["total"] == 0
        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert audit_records[-2].op == "forget"
        assert audit_records[-2].data["entries_deleted"] == 1
        assert audit_records[-1].op == "access"

    def test_forget_actor_with_zero_entries_still_audits(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        with create_backend_from_config(cfg, "project:default") as backend:
            result = memory_forget_impl(None, None, "project:default", backend=backend, config=cfg, actor="ghost")

        assert result["entries_deleted"] == 0
        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert audit_records[-1].op == "forget"
        assert audit_records[-1].data["entries_deleted"] == 0

    def test_search_actor_scans_full_namespace_before_filtering(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entries = [_make_entry(f"M-seed-{index}", namespace="project:default") for index in range(600)] + [
            _make_entry(f"M-alice-{index}", namespace="project:default") for index in range(100)
        ]
        for entry in entries[600:]:
            entry.source_identity = "alice"

        backend = _mock_backend(entries)
        backend.list_entries.side_effect = lambda **kwargs: entries[: int(kwargs["limit"])]

        result = memory_search_impl("project:default", backend=backend, config=cfg, actor="alice", limit=50)

        returned = cast("list[dict[str, object]]", result["entries"])
        assert result["total"] == 100
        assert len(returned) == 50
        assert all(entry["source_identity"] == "alice" for entry in returned)

    def test_forget_actor_scans_full_namespace_before_bulk_delete(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entries = [_make_entry(f"M-seed-{index}", namespace="project:default") for index in range(10_050)] + [
            _make_entry(f"M-alice-{index}", namespace="project:default") for index in range(20)
        ]
        for entry in entries[10_050:]:
            entry.source_identity = "alice"

        backend = _mock_backend(entries)
        deleted_ids: set[str] = set()

        backend.list_entries.side_effect = lambda **kwargs: [entry for entry in entries if entry.id not in deleted_ids][
            : int(kwargs["limit"])
        ]
        backend.delete.side_effect = lambda entry_id: deleted_ids.add(entry_id) is None

        result = memory_forget_impl(None, None, "project:default", backend=backend, config=cfg, actor="alice")

        assert result["entries_deleted"] == 20
        assert all(entry.id in deleted_ids for entry in entries[10_050:])

    def test_recall_audits_recall_and_access_operations(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        with create_backend_from_config(cfg, "project:default") as backend:
            backend.store(MemoryEntry(id="M-001", content="alpha memory", namespace="project:default"))
            result = memory_recall_impl("alpha", "project:default", backend=backend, config=cfg)

        assert result["total_matches"] == 1
        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert [record.op for record in audit_records[-2:]] == ["access", "recall"]

    def test_consolidate_audits_operation(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        with create_backend_from_config(cfg, "project:default") as backend:
            backend.store(MemoryEntry(id="M-001", content="alpha", namespace="project:default"))
            backend.store(MemoryEntry(id="M-002", content="alpha", namespace="project:default"))
            result = memory_consolidate_impl("project:default", backend=backend, config=cfg, dry_run=True)

        assert result["dry_run"] is True
        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert audit_records[-1].op == "consolidate"

    def test_consolidate_denied_for_reader_namespace_role(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(
            storage_path=str(tmp_path / "mem"), rbac_enabled=True, namespace_roles={"project:default": "reader"}
        )

        with create_backend_from_config(cfg, "project:default") as backend:
            with pytest.raises(
                AuthorizationError,
                match=r"Role 'reader' does not have consolidate permission on namespace 'project:default'\.",
            ):
                memory_consolidate_impl("project:default", backend=backend, config=cfg)

    def test_store_impl_can_raise_typed_security_errors_for_public_tool_contract(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        backend = _mock_backend()

        with pytest.raises(PIIBlockError):
            memory_store_impl(
                "sk-abcdefghijklmnopqrstuvwxyz",
                "project:default",
                backend=backend,
                config=cfg,
                raise_security_errors=True,
            )

        with pytest.raises(SchemaValidationError):
            memory_store_impl(  # type: ignore[arg-type]
                "valid content",
                "project:default",
                backend=backend,
                config=cfg,
                metadata={"owner": 1},
                raise_security_errors=True,
            )

        audit_records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert audit_records[-1].op == "store_rejected"
        assert audit_records[-1].data["reason"] == "schema_invalid"
