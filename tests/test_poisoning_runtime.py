"""Tests for trw_memory.security.poisoning — runtime enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import RateLimitError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.poisoning import quarantine_entry
from trw_memory.security.runtime import (
    append_audit_event,
    delete_quarantined_entries,
    list_quarantined_entries,
    prepare_entry_for_store,
    store_quarantined_entry,
)
from trw_memory.storage.persistence import read_yaml


class TestRuntimePoisoningPolicy:
    def test_runtime_persists_anomaly_stats_for_recent_non_quarantined_entries(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), poisoning_z_threshold=1.0)

        with create_backend_from_config(cfg, "project:default") as backend:
            for index in range(120):
                backend.store(
                    MemoryEntry(
                        id=f"M-seed-{index}",
                        content="seed content",
                        namespace="project:default",
                        importance=0.5,
                    )
                )

            prepared = prepare_entry_for_store(
                MemoryEntry(id="M-new", content="normal", namespace="project:default"),
                backend=backend,
                config=cfg,
            )

        stats = read_yaml(Path(cfg.quarantine_path).parent / "anomaly_stats.yaml")
        assert prepared.quarantined is False
        assert stats["sample_count"] == 100
        assert set(stats["dimensions"]) == {"entry_length", "tag_count", "importance"}

    def test_runtime_rate_limit_raises_retry_after(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), max_memory_writes_per_minute=1)

        with create_backend_from_config(cfg, "project:default") as backend:
            first = MemoryEntry(id="M-1", content="first", namespace="project:default")
            second = MemoryEntry(id="M-2", content="second", namespace="project:default")
            prepare_entry_for_store(first, backend=backend, config=cfg, session_id="s1")
            with pytest.raises(RateLimitError) as excinfo:
                prepare_entry_for_store(second, backend=backend, config=cfg, session_id="s1")

        assert excinfo.value.retry_after > 0.0
        audit_records = list(Path(cfg.audit_log_path).read_text(encoding="utf-8").splitlines())
        assert any('"op":"store_rejected"' in line for line in audit_records)
        assert any('"reason":"rate_limited"' in line for line in audit_records)
        assert any('"session_id":"s1"' in line for line in audit_records)

    def test_runtime_rate_limit_bounds_retry_after_window(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), max_memory_writes_per_minute=10)
        now_values = iter([1000.0 + (index * 3.0) for index in range(10)] + [1030.0])
        monkeypatch.setattr("trw_memory.security.runtime.time", lambda: next(now_values))

        with create_backend_from_config(cfg, "project:default") as backend:
            for index in range(10):
                prepare_entry_for_store(
                    MemoryEntry(id=f"M-{index}", content=f"entry {index}", namespace="project:default"),
                    backend=backend,
                    config=cfg,
                    session_id="burst",
                )

            with pytest.raises(RateLimitError) as excinfo:
                prepare_entry_for_store(
                    MemoryEntry(id="M-over", content="overflow", namespace="project:default"),
                    backend=backend,
                    config=cfg,
                    session_id="burst",
                )

        assert 30.0 <= excinfo.value.retry_after <= 60.0

    def test_runtime_rate_limit_prunes_stale_sessions(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), max_memory_writes_per_minute=5)
        state_path = Path(cfg.rate_limit_state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("sessions:\n  old: [1.0]\n", encoding="utf-8")

        with create_backend_from_config(cfg, "project:default") as backend:
            prepare_entry_for_store(
                MemoryEntry(id="M-now", content="now", namespace="project:default"),
                backend=backend,
                config=cfg,
                session_id="current",
            )

        state = read_yaml(state_path)
        assert "old" not in state["sessions"]
        assert "current" in state["sessions"]

    def test_runtime_rate_limit_none_session_id_skips_limiting(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), max_memory_writes_per_minute=1)

        with create_backend_from_config(cfg, "project:default") as backend:
            prepare_entry_for_store(
                MemoryEntry(id="M-1", content="first", namespace="project:default"),
                backend=backend,
                config=cfg,
                session_id=None,
            )
            prepare_entry_for_store(
                MemoryEntry(id="M-2", content="second", namespace="project:default"),
                backend=backend,
                config=cfg,
                session_id=None,
            )

        assert Path(cfg.rate_limit_state_path).exists() is False

    def test_runtime_rate_limit_zero_threshold_disables_limiting(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), max_memory_writes_per_minute=0)

        with create_backend_from_config(cfg, "project:default") as backend:
            for index in range(3):
                prepare_entry_for_store(
                    MemoryEntry(id=f"M-{index}", content=f"entry {index}", namespace="project:default"),
                    backend=backend,
                    config=cfg,
                    session_id="s1",
                )

        assert Path(cfg.rate_limit_state_path).exists() is False

    def test_quarantine_storage_list_and_delete_round_trip(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entry = quarantine_entry(
            MemoryEntry(
                id="M-q1",
                content="quarantined",
                namespace="project:default",
                source_identity="alice",
            )
        )

        store_quarantined_entry(cfg, entry)
        listed = list_quarantined_entries(cfg, namespace="project:default", actor="alice")
        deleted = delete_quarantined_entries(cfg, namespace="project:default", actor="alice")
        after_delete = list_quarantined_entries(cfg, namespace="project:default", actor="alice")

        assert [candidate.id for candidate in listed] == ["M-q1"]
        assert deleted == 1
        assert after_delete == []

    def test_prepare_entry_for_store_respects_disabled_pii_checks(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), pii_enabled=False)
        entry = MemoryEntry(id="M-pii-off", content="user@example.com", namespace="project:default")

        with create_backend_from_config(cfg, "project:default") as backend:
            prepared = prepare_entry_for_store(entry, backend=backend, config=cfg)

        assert prepared.entry.content == "user@example.com"
        assert prepared.pii_matches == ()

    def test_prepare_entry_for_store_marks_high_entropy_metadata(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entry = MemoryEntry(
            id="M-entropy",
            content="token aB3cD9eF2gH5iJ8kL1mN4oP7qR6sT0",
            namespace="project:default",
        )

        with create_backend_from_config(cfg, "project:default") as backend:
            prepared = prepare_entry_for_store(entry, backend=backend, config=cfg)

        assert prepared.entry.metadata["contains_high_entropy_token"] == "true"

    def test_append_audit_event_noops_when_disabled(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), audit_enabled=False)
        append_audit_event(cfg, "store", entry_id="M-001", namespace="project:default")
        assert Path(cfg.audit_log_path).exists() is False
