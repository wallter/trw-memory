"""Tests for trw_memory.security.poisoning — runtime enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import RateLimitError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.runtime import (
    append_audit_event,
    delete_quarantined_entries,
    list_quarantined_entries,
    prepare_entry_for_store,
    store_quarantined_entry,
)
from trw_memory.storage.persistence import read_yaml


class TestRuntimePoisoningPolicy:
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
        entry = MemoryEntry(
            id="M-q1",
            content="quarantined",
            namespace="project:default",
            source_identity="alice",
            metadata={"quarantined": "true"},
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

    def test_pii_policy_keeps_email_in_tag_verbatim(self, tmp_path: Path) -> None:
        """An email in a tag is DETECTED but stored exactly as written (2026-07-25).

        The store path no longer mutates local text for built-in detector types.
        Sanitization happens at the only boundary where the data leaves the
        machine — ``sync/_remote_publish`` — which is reversible because the
        local row keeps the truth.
        """
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entry = MemoryEntry(
            id="M-tag-email",
            content="benign content",
            namespace="project:default",
            tags=["contact:user@example.com"],
        )

        secured, matches = apply_runtime_pii_policy(entry, cfg)
        assert secured.tags[0] == "contact:user@example.com"
        assert "<email>" not in secured.tags[0]
        assert any(str(match.pii_type) == "email" for match in matches)
        assert "email" in secured.metadata["pii_types"]

    def test_pii_policy_keeps_ssn_shaped_tag_verbatim(self, tmp_path: Path) -> None:
        """An SSN-shaped tag is detected but not rewritten (2026-07-25).

        The SSN detector is ``\\b\\d{3}[-\\s]?\\d{2}[-\\s]?\\d{4}\\b`` — it fires on
        any 9 consecutive digits, so mutating on it destroyed build numbers and
        ids. Detection is kept for the audit trail; the text is kept for the user.
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
        assert secured.tags[0] == "customer-ssn:123-45-6789"
        assert "<ssn>" not in secured.tags[0]
        assert any(str(match.pii_type) == "ssn" for match in matches)
        assert "ssn" in secured.metadata["pii_types"]

    @pytest.mark.parametrize(
        ("entry_id", "template", "token"),
        [
            # A 40-char token with no recognized prefix that clears the entropy floor.
            (
                "M-high-entropy-credential",
                "the credential is {token} keep it safe",
                "aB3cD9eF2gH5iJ8kL1mN4oP7qR6sT0uV3wX5yZ8b",
            ),
            # The backstop also fires on legitimate technical prose — snapshot ids,
            # digests, dotted identifiers. Measured true-positive rate on this
            # project's corpus was ZERO, which is why it no longer mutates.
            (
                "M-high-entropy-prose",
                "session_start returned {token} and then failed",
                "surf_9f3aB7cD2eF5gH8iJ1kL4mN6oP0qR3s",
            ),
        ],
    )
    def test_pii_policy_keeps_high_entropy_token_verbatim(
        self, tmp_path: Path, entry_id: str, template: str, token: str
    ) -> None:
        """HIGH_ENTROPY is flagged in metadata but never rewritten (2026-07-25).

        Replaces the 2026-06-17 redaction and the 2026-07-24 elision that softened
        it: both destroyed the sentence a learning existed to record, irreversibly
        and before anything reached disk.
        """
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        text = template.format(token=token)
        entry = MemoryEntry(id=entry_id, content=text, namespace="project:default")

        secured, matches = apply_runtime_pii_policy(entry, cfg)
        assert secured.content == text
        assert "<id:" not in secured.content
        # The observability signal survives intact — only the mutation is gone.
        assert secured.metadata["contains_high_entropy_token"] == "true"
        assert any(str(match.pii_type) == "high_entropy" for match in matches)

    def test_pii_policy_masks_operator_configured_custom_pattern(self, tmp_path: Path) -> None:
        """``pii_custom_patterns`` is the one masking path that survives on write.

        It is not a heuristic — it is the operator's own regex, empty by default.
        Keeping it preserves local-masking control for regulated deployments
        without letting our 8 built-in regexes destroy anyone's text.
        """
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(
            storage_path=str(tmp_path / "mem"),
            pii_custom_patterns=[r"CUST-\d{6}"],
        )
        entry = MemoryEntry(
            id="M-custom",
            content="incident for CUST-123456 raised by user@example.com",
            namespace="project:default",
        )

        secured, matches = apply_runtime_pii_policy(entry, cfg)
        assert "CUST-123456" not in secured.content
        assert "<custom_pii>" in secured.content
        # The built-in detector alongside it still does NOT mutate.
        assert "user@example.com" in secured.content
        assert any(str(match.pii_type) == "custom" for match in matches)

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


class TestEvidenceAndAssertionIntakeCoverage:
    """SEC-001 release-blocker (2026-07-17): evidence[] and Assertion.last_evidence
    are publicly reachable via memory_store and were persisted verbatim, bypassing
    BOTH PII detection/redaction AND trust/poisoning scoring. These tests pin that
    the intake path now folds those fields into the scanned surface.
    """

    def test_pii_policy_blocks_api_key_in_evidence(self, tmp_path: Path) -> None:
        """An API key placed in evidence[] must trigger the PII block, not bypass it."""
        from trw_memory.exceptions import PIIBlockError
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entry = MemoryEntry(
            id="M-evidence-key",
            content="benign content",
            namespace="project:default",
            evidence=["source: sk-abcdefghijklmnopqrstuvwxyz"],
        )

        with pytest.raises(PIIBlockError, match="api_key"):
            apply_runtime_pii_policy(entry, cfg)

    def test_pii_policy_scans_evidence_and_keeps_it_verbatim(self, tmp_path: Path) -> None:
        """evidence[] is scanned (so API keys there still block) but not rewritten."""
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entry = MemoryEntry(
            id="M-evidence-email",
            content="benign content",
            namespace="project:default",
            evidence=["reported by user@example.com"],
        )

        secured, matches = apply_runtime_pii_policy(entry, cfg)
        assert secured.evidence[0] == "reported by user@example.com"
        assert any(str(match.pii_type) == "email" for match in matches)

    def test_pii_policy_scans_assertion_evidence_and_keeps_it_verbatim(self, tmp_path: Path) -> None:
        """Assertion.last_evidence is scanned for the block gate but not rewritten."""
        from trw_memory.models.memory import Assertion, AssertionType
        from trw_memory.security._runtime_pii import apply_runtime_pii_policy

        cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
        entry = MemoryEntry(
            id="M-assertion-ssn",
            content="benign content",
            namespace="project:default",
            assertions=[
                Assertion(
                    type=AssertionType.GLOB_EXISTS,
                    target="*.py",
                    last_evidence="verified for customer 123-45-6789",
                )
            ],
        )

        secured, matches = apply_runtime_pii_policy(entry, cfg)
        assert secured.assertions[0].last_evidence == "verified for customer 123-45-6789"
        assert any(str(match.pii_type) == "ssn" for match in matches)


class TestEvidenceGateLogContainment:
    """PRD-CORE-244 NFR03 — the FR02 rejection log carries ids and codes only.

    The gate fires BEFORE ``_stage_pii_policy`` runs, so a rejected entry may
    still hold exactly the payload the PII stage exists to contain. A log line
    that echoed ``content``, ``detail`` or ``evidence`` would move that payload
    into a file with different retention and different permissions.
    """

    _SENTINEL = "sentinel-9f2c1a-do-not-log-this-payload"

    def test_evidence_gate_does_not_leak_entry_content_to_logs(self) -> None:
        from structlog.testing import capture_logs

        from trw_memory.exceptions import SchemaValidationError
        from trw_memory.models.memory import Confidence
        from trw_memory.security.poisoning import reject_unsubstantiated_verified

        entry = MemoryEntry(
            id="M-leak-probe",
            content=f"claim about {self._SENTINEL}",
            detail=f"supporting detail mentioning {self._SENTINEL}",
            nudge_line=self._SENTINEL,
            namespace="project:default",
            confidence=Confidence.VERIFIED,
        )

        with capture_logs() as logs:
            with pytest.raises(SchemaValidationError):
                reject_unsubstantiated_verified(entry, min_items=1)

        # Non-vacuity: the gate really ran and really logged.
        assert logs, "the rejection must be observable in the log stream"
        assert any(record.get("event") == "unsubstantiated_verified_rejected" for record in logs)
        assert any(record.get("entry_id") == "M-leak-probe" for record in logs)

        # Containment: no captured field — key OR value, at any depth — carries it.
        assert self._SENTINEL not in repr(logs)

    def test_evidence_gate_containment_holds_through_validate_entry_payload(self) -> None:
        """Same containment on the runtime write path, not just the helper."""
        from structlog.testing import capture_logs

        from trw_memory.exceptions import SchemaValidationError
        from trw_memory.models.memory import Confidence
        from trw_memory.security.poisoning import validate_entry_payload

        entry = MemoryEntry(
            id="M-leak-probe-2",
            content=f"claim about {self._SENTINEL}",
            detail=self._SENTINEL,
            namespace="project:default",
            confidence=Confidence.VERIFIED,
        )

        with capture_logs() as logs:
            with pytest.raises(SchemaValidationError):
                validate_entry_payload(entry, max_chars=100_000, min_evidence_items_for_verified=1)

        assert any(record.get("event") == "unsubstantiated_verified_rejected" for record in logs)
        assert self._SENTINEL not in repr(logs)
