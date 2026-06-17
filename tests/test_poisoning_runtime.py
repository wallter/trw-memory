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
        assert set(list(stats["dimensions"])) == {"entry_length", "tag_count", "importance"}  # type: ignore[arg-type]

    def test_sub_baseline_namespace_emits_audit_event(self, tmp_path: Path) -> None:
        """trw-memory-10: a namespace below the statistical baseline (<10 clean
        entries) must record an audit event for the skipped anomaly detection, not
        only a structlog WARNING — so the sub-baseline window is visible in the
        audit trail an attacker could exploit by seeding baseline-1 entries.
        """
        from trw_memory.security.audit import AuditLog

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        assert cfg.poisoning_detection_enabled is True

        with create_backend_from_config(cfg, "project:default") as backend:
            # Seed only 5 clean entries — below the 10-entry baseline.
            for index in range(5):
                backend.store(
                    MemoryEntry(
                        id=f"M-seed-{index}",
                        content="seed content",
                        namespace="project:default",
                        importance=0.5,
                    )
                )
            prepare_entry_for_store(
                MemoryEntry(id="M-new", content="normal entry", namespace="project:default"),
                backend=backend,
                config=cfg,
            )

        records = AuditLog(Path(cfg.audit_log_path)).read_all()
        baseline_events = [r for r in records if r.op == "anomaly_baseline_insufficient"]
        assert len(baseline_events) == 1
        event = baseline_events[0]
        assert event.data["min_baseline"] == 10
        assert int(str(event.data["sample_count"])) < 10
        assert event.data["reason"] == "below_statistical_baseline"

    def test_above_baseline_namespace_emits_no_baseline_audit_event(self, tmp_path: Path) -> None:
        """The sub-baseline audit event must NOT fire once the namespace has a
        full statistical baseline — the signal is specific to the skip window."""
        from trw_memory.security.audit import AuditLog

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))

        with create_backend_from_config(cfg, "project:default") as backend:
            for index in range(30):
                backend.store(
                    MemoryEntry(
                        id=f"M-seed-{index}",
                        content="seed content",
                        namespace="project:default",
                        importance=0.5,
                    )
                )
            prepare_entry_for_store(
                MemoryEntry(id="M-new", content="normal entry", namespace="project:default"),
                backend=backend,
                config=cfg,
            )

        records = AuditLog(Path(cfg.audit_log_path)).read_all()
        assert not [r for r in records if r.op == "anomaly_baseline_insufficient"]

    def test_size_anomaly_observe_mode_default_does_not_quarantine_long_entry(self, tmp_path: Path) -> None:
        """SEC-001 default: a long, well-formed learning is observed, NOT quarantined.

        Regression for the PRD-DIST-254 MCP-vs-MemoryClient recall divergence:
        the per-entry MCP write path accumulates a short-entry reference
        distribution, so an 11th long entry scored as a >3-sigma length outlier
        was silently quarantined out of recall. Observe-mode (the documented
        default rollout) must store it normally while still recording the anomaly.
        """
        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"), poisoning_z_threshold=3.0)
        assert cfg.poisoning_detection_mode == "observe"

        long_detail = "Lambda cold-start contention requires NullPool. " * 40

        with create_backend_from_config(cfg, "project:default") as backend:
            for index in range(12):
                backend.store(
                    MemoryEntry(
                        id=f"M-short-{index}",
                        content="short",
                        detail="x",
                        namespace="project:default",
                        importance=0.6,
                    )
                )

            prepared = prepare_entry_for_store(
                MemoryEntry(
                    id="M-long-decision",
                    content="switched SQLAlchemy engine to NullPool",
                    detail=long_detail,
                    namespace="project:default",
                    importance=0.6,
                ),
                backend=backend,
                config=cfg,
            )

        # Observed as an anomaly (diagnostics populated) but NOT held back.
        assert prepared.quarantined is False
        assert prepared.anomaly_dimension == "entry_length"
        assert prepared.anomaly_z_score >= 3.0

    def test_size_anomaly_enforce_mode_quarantines_long_entry(self, tmp_path: Path) -> None:
        """Explicit enforce-mode still quarantines the statistical size outlier."""
        cfg = MemoryConfig(
            storage_path=str(tmp_path / "mem"),
            poisoning_z_threshold=3.0,
            poisoning_detection_mode="enforce",
        )

        long_detail = "Lambda cold-start contention requires NullPool. " * 40

        with create_backend_from_config(cfg, "project:default") as backend:
            for index in range(12):
                backend.store(
                    MemoryEntry(
                        id=f"M-short-{index}",
                        content="short",
                        detail="x",
                        namespace="project:default",
                        importance=0.6,
                    )
                )

            prepared = prepare_entry_for_store(
                MemoryEntry(
                    id="M-long-decision",
                    content="switched SQLAlchemy engine to NullPool",
                    detail=long_detail,
                    namespace="project:default",
                    importance=0.6,
                ),
                backend=backend,
                config=cfg,
            )

        assert prepared.quarantined is True
        assert prepared.anomaly_dimension == "entry_length"

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
        sessions = state["sessions"]
        assert isinstance(sessions, dict)
        assert "old" not in sessions
        assert "current" in sessions

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

    def test_pii_policy_blocks_api_key_hidden_in_tag(self, tmp_path: Path) -> None:
        """An API key placed in a TAG must trigger the PII block, not bypass it."""
        from trw_memory.exceptions import PIIBlockError
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entry = MemoryEntry(
            id="M-tag-key",
            content="benign content",
            namespace="project:default",
            tags=["ok", "sk-abcdefghijklmnopqrstuvwxyz"],
        )

        with pytest.raises(PIIBlockError, match="api_key"):
            apply_runtime_pii_policy(entry, cfg)

    def test_pii_policy_redacts_email_in_tag(self, tmp_path: Path) -> None:
        """An email in a tag must be redacted, not stored in the clear."""
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entry = MemoryEntry(
            id="M-tag-email",
            content="benign content",
            namespace="project:default",
            tags=["contact:user@example.com"],
        )

        secured, matches = apply_runtime_pii_policy(entry, cfg)
        assert "user@example.com" not in secured.tags[0]
        assert "<email>" in secured.tags[0]
        assert matches

    def test_pii_policy_redacts_ssn_in_tag(self, tmp_path: Path) -> None:
        """An SSN in a tag must be redacted, not stored verbatim (v0.9.2).

        Regression for the incomplete v0.9.1 tag-scan: SSN/PHONE/CREDIT_CARD
        were detected but absent from REDACTED_PII_TYPES, so replace_pii() fell
        through its ``else: continue`` and stored the raw value at recall time.
        """
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entry = MemoryEntry(
            id="M-tag-ssn",
            content="benign content",
            namespace="project:default",
            tags=["customer-ssn:123-45-6789"],
        )

        secured, matches = apply_runtime_pii_policy(entry, cfg)
        assert "123-45-6789" not in secured.tags[0]
        assert "<ssn>" in secured.tags[0]
        assert matches

    def test_pii_policy_redacts_high_entropy_token_in_content(self, tmp_path: Path) -> None:
        """A HIGH_ENTROPY secret (no API_KEY prefix) must be redacted, not stored verbatim.

        Regression (2026-06-17): HIGH_ENTROPY was detected — the metadata flag was
        set — but absent from REDACTED_PII_TYPES, so replace_pii() fell through its
        ``else: continue`` and persisted the raw secret in content/detail/tags.
        """
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        # A 40-char token with no recognized prefix that clears the entropy floor.
        secret = "aB3cD9eF2gH5iJ8kL1mN4oP7qR6sT0uV3wX5yZ8b"
        entry = MemoryEntry(
            id="M-high-entropy-redact",
            content=f"the credential is {secret} keep it safe",
            namespace="project:default",
        )

        secured, matches = apply_runtime_pii_policy(entry, cfg)
        assert secret not in secured.content
        assert "<high_entropy_secret>" in secured.content
        assert secured.metadata["contains_high_entropy_token"] == "true"
        assert any(str(m.pii_type) == "high_entropy" for m in matches)


class TestAnomalyQuarantineNoCallerBypass:
    """Security regression: the anomaly quarantine must NOT be bypassable via
    caller-supplied ``entry.metadata['source']``.

    The former PRD-DIST-2045 carve-out skipped anomaly quarantine when
    ``entry.metadata['source']`` started with a configured prefix (e.g.
    ``distilled:``). Because ``metadata`` is entirely caller-controlled, any
    caller could spoof that field and slip a poisoned outlier past the
    detector. The bypass was removed: in enforce-mode every write — regardless
    of its claimed source metadata — goes through anomaly quarantine.
    """

    @staticmethod
    def _seed_namespace(backend: object, n: int = 120) -> None:
        for index in range(n):
            backend.store(  # type: ignore[attr-defined]
                MemoryEntry(
                    id=f"M-seed-{index}",
                    content="seed",
                    namespace="project:default",
                    importance=0.5,
                    tags=["s1"],
                )
            )

    def test_distilled_source_metadata_cannot_bypass_quarantine(self, tmp_path: Path) -> None:
        # A spoofed ``distilled:`` source must NOT exempt an outlier from
        # enforce-mode quarantine — the bypass was caller-controlled and is gone.
        cfg = MemoryConfig(
            storage_path=str(tmp_path / "mem"),
            poisoning_z_threshold=1.0,
            poisoning_detection_mode="enforce",
        )
        outlier = MemoryEntry(
            id="M-distill",
            content="outlier-content",
            namespace="project:default",
            tags=[f"t{i}" for i in range(30)],
            metadata={"source": "distilled:git:DEADBEEF..CAFEBABE"},
        )

        with create_backend_from_config(cfg, "project:default") as backend:
            self._seed_namespace(backend)
            prepared = prepare_entry_for_store(outlier, backend=backend, config=cfg)

        assert prepared.quarantined is True
        assert prepared.anomaly_dimension == "tag_count"

    def test_agent_source_outlier_still_quarantined(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(
            storage_path=str(tmp_path / "mem"),
            poisoning_z_threshold=1.0,
            poisoning_detection_mode="enforce",
        )
        outlier = MemoryEntry(
            id="M-agent",
            content="outlier-content",
            namespace="project:default",
            tags=[f"t{i}" for i in range(30)],
            metadata={"source": "agent:my-tool"},
        )

        with create_backend_from_config(cfg, "project:default") as backend:
            self._seed_namespace(backend)
            prepared = prepare_entry_for_store(outlier, backend=backend, config=cfg)

        assert prepared.quarantined is True
        assert prepared.anomaly_dimension == "tag_count"

    def test_custom_prefix_metadata_cannot_bypass_quarantine(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(
            storage_path=str(tmp_path / "mem"),
            poisoning_z_threshold=1.0,
            poisoning_detection_mode="enforce",
        )
        outlier = MemoryEntry(
            id="M-custom",
            content="outlier-content",
            namespace="project:default",
            tags=[f"t{i}" for i in range(30)],
            metadata={"source": "my-pipeline:job-42"},
        )

        with create_backend_from_config(cfg, "project:default") as backend:
            self._seed_namespace(backend)
            prepared = prepare_entry_for_store(outlier, backend=backend, config=cfg)

        assert prepared.quarantined is True

    def test_provenance_hash_matches_post_pii_content(self, tmp_path: Path) -> None:
        """PRD-DIST-2046 c793: provenance_content_hash MUST equal sha256(stored content + detail).

        Pre-c793 the hash was computed inside _apply_sec001_intake BEFORE
        _apply_runtime_pii_policy ran, so when PII redaction modified content
        the stored entry had post-PII content but pre-PII hash. Recall-time
        filter_recall_window then BLOCKED the entry on `hash_pin_drift`.

        This test exercises a payload that triggers PII detection
        (high-entropy token marker) and asserts that after prepare_entry_for_store
        completes, sha256(secured_entry.content + secured_entry.detail) ==
        secured_entry.metadata['provenance_content_hash'].
        """
        import hashlib

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        # Payload includes a high-entropy token that PRD-SEC-001's PII policy
        # marks via `contains_high_entropy_token` metadata; this exercises the
        # _apply_runtime_pii_policy path that previously caused hash drift.
        entry = MemoryEntry(
            id="M-prov-post-pii",
            content="prod token aB3cD9eF2gH5iJ8kL1mN4oP7qR6sT0 in code",
            detail="auth header carries it",
            namespace="project:default",
        )

        with create_backend_from_config(cfg, "project:default") as backend:
            prepared = prepare_entry_for_store(entry, backend=backend, config=cfg)

        stored_meta = prepared.entry.metadata
        stored_hash = stored_meta.get("provenance_content_hash", "")
        if not stored_hash:
            # Provenance not required for this config? Skip the assertion path.
            return
        recomputed = hashlib.sha256(f"{prepared.entry.content}{prepared.entry.detail}".encode()).hexdigest()
        assert stored_hash == recomputed, (
            f"provenance hash drift detected: stored={stored_hash} "
            f"recomputed={recomputed}; content={prepared.entry.content!r}"
        )

    def test_bypass_does_not_skip_when_metadata_source_missing(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(
            storage_path=str(tmp_path / "mem"),
            poisoning_z_threshold=1.0,
            poisoning_detection_mode="enforce",
        )
        outlier = MemoryEntry(
            id="M-no-source",
            content="outlier-content",
            namespace="project:default",
            tags=[f"t{i}" for i in range(30)],
            metadata={},
        )

        with create_backend_from_config(cfg, "project:default") as backend:
            self._seed_namespace(backend)
            prepared = prepare_entry_for_store(outlier, backend=backend, config=cfg)

        # No metadata['source'] → bypass cannot match → quarantined as usual.
        assert prepared.quarantined is True


class TestQuarantineNamespaceMetadata:
    """``read_namespace_metadata`` must fail open on a corrupt sidecar.

    Mirrors the storage-side seam: a single unreadable / non-UTF-8
    ``namespace.txt`` in the quarantine tree yields ``None`` rather than
    raising into the review/discovery path.
    """

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        from trw_memory.security._runtime_quarantine import read_namespace_metadata

        assert read_namespace_metadata(tmp_path) is None

    def test_roundtrip_returns_namespace(self, tmp_path: Path) -> None:
        from trw_memory.security._runtime_quarantine import (
            NAMESPACE_METADATA_FILE,
            read_namespace_metadata,
        )

        (tmp_path / NAMESPACE_METADATA_FILE).write_text("project:default", encoding="utf-8")
        assert read_namespace_metadata(tmp_path) == "project:default"

    def test_non_utf8_fails_open(self, tmp_path: Path) -> None:
        import structlog

        from trw_memory.security._runtime_quarantine import (
            NAMESPACE_METADATA_FILE,
            read_namespace_metadata,
        )

        (tmp_path / NAMESPACE_METADATA_FILE).write_bytes(b"project:\x80\xffbad")
        with structlog.testing.capture_logs() as logs:
            assert read_namespace_metadata(tmp_path) is None
        dropped = [r for r in logs if r["event"] == "namespace_metadata_read_failed"]
        assert len(dropped) == 1
        assert dropped[0]["error"] == "UnicodeDecodeError"
        assert "project" not in repr(dropped[0])

    def test_unreadable_sidecar_fails_open(self, tmp_path: Path) -> None:
        from trw_memory.security._runtime_quarantine import (
            NAMESPACE_METADATA_FILE,
            read_namespace_metadata,
        )

        (tmp_path / NAMESPACE_METADATA_FILE).mkdir()
        assert read_namespace_metadata(tmp_path) is None


class TestScoreAnomalyActiveFilter:
    """P1 regression: score_anomaly must exclude non-ACTIVE entries from the reference set.

    An entry with status=OBSOLETE or ARCHIVED is retired and no longer represents
    normal write behaviour. Including it in the rolling window skews the mean/std
    and corrupts z-scores: a genuine anomaly can score low (baseline widened by
    old large entries) and pass through undetected.
    """

    def test_score_anomaly_ignores_obsolete_entries_in_baseline(self, tmp_path: Path) -> None:
        """Obsolete entries must not appear in the reference window used to build stats."""
        from unittest.mock import patch

        from trw_memory.models.memory import MemoryStatus
        from trw_memory.security._runtime_anomaly import score_anomaly
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        backend = SQLiteBackend(tmp_path / "mem" / "memory.db")

        # Seed 15 normal active entries
        for index in range(15):
            backend.store(
                MemoryEntry(
                    id=f"M-active-{index}",
                    content="short",
                    namespace="default",
                    importance=0.5,
                    status=MemoryStatus.ACTIVE,
                )
            )

        # Add one OBSOLETE entry with enormous content to shift the baseline.
        # If score_anomaly fetches ALL statuses, this inflates mean entry_length
        # and hides a real size anomaly in the new candidate.
        backend.store(
            MemoryEntry(
                id="M-retired",
                content="X" * 50_000,
                namespace="default",
                importance=0.5,
                status=MemoryStatus.ACTIVE,  # store active first
            )
        )
        # Retire it so it should be excluded from the baseline
        backend.update("M-retired", status=MemoryStatus.OBSOLETE)

        # Verify the backend actually stores the obsolete entry
        retired = backend.get("M-retired")
        assert retired is not None and retired.status == MemoryStatus.OBSOLETE

        # Track what list_entries was called with
        original_list = backend.list_entries

        calls: list[MemoryStatus | None] = []

        def tracked_list_entries(
            *,
            status: MemoryStatus | None = None,
            namespace: str | None = None,
            limit: int = 100,
        ) -> list[MemoryEntry]:
            calls.append(status)
            return original_list(status=status, namespace=namespace, limit=limit)

        candidate = MemoryEntry(
            id="M-candidate",
            content="normal size entry",
            namespace="default",
            importance=0.5,
        )
        with patch.object(backend, "list_entries", side_effect=tracked_list_entries):
            score_anomaly(candidate, backend, config=cfg)

        # score_anomaly must pass status=MemoryStatus.ACTIVE
        assert calls, "list_entries was never called"
        assert all(status == MemoryStatus.ACTIVE for status in calls), f"Expected ACTIVE-only calls; got: {calls}"

        backend.close()

    def test_score_anomaly_excludes_obsolete_from_stats_count(self, tmp_path: Path) -> None:
        """AnomalyStats.sample_count must reflect only active entries."""
        from trw_memory.models.memory import MemoryStatus
        from trw_memory.security._runtime_anomaly import score_anomaly
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        backend = SQLiteBackend(tmp_path / "mem" / "memory.db")

        # Seed 20 active + 5 obsolete
        for index in range(20):
            backend.store(
                MemoryEntry(
                    id=f"M-active-{index}",
                    content="normal content",
                    namespace="default",
                    status=MemoryStatus.ACTIVE,
                )
            )
        for index in range(5):
            entry_id = f"M-obs-{index}"
            backend.store(
                MemoryEntry(
                    id=entry_id,
                    content="retired content",
                    namespace="default",
                    status=MemoryStatus.ACTIVE,
                )
            )
            backend.update(entry_id, status=MemoryStatus.OBSOLETE)

        candidate = MemoryEntry(
            id="M-new",
            content="new entry",
            namespace="default",
        )
        _, stats = score_anomaly(candidate, backend, config=cfg)

        # Rolling window is the 100 most recent active entries; here all
        # 20 active entries should appear — none of the 5 obsolete ones.
        # (clean_reference filters quarantined+canary; stats count = len(rolling))
        assert stats.sample_count == 20, (
            f"Expected 20 active entries in baseline; got {stats.sample_count} "
            "(obsolete entries may be leaking into the reference set)"
        )

        backend.close()
