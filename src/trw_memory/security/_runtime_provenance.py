"""SEC-001 trust/provenance intake helpers for runtime writes."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from trw_memory.exceptions import ProvenanceKeyUnavailableError, ScorerUnavailableError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.poisoning import quarantine_entry
from trw_memory.security.provenance import build_entry_provenance
from trw_memory.security.startup import _discover_anchor, resolve_security_path, verify_defaults
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event
from trw_memory.security.trust_scorer import score_intake

__all__ = ["apply_provenance_hash", "apply_sec001_intake"]


def _actor_for_entry(entry: MemoryEntry) -> str:
    return entry.source_identity or entry.source or "system"


def _resolve_provenance_session_id(entry: MemoryEntry, session_id: str | None) -> str:
    return (
        session_id
        or entry.metadata.get("session_id", "")
        or entry.metadata.get("installation_id", "")
        or os.environ.get("TRW_SESSION_ID", "").strip()
        or entry.source_identity
        or "unknown-session"
    )


def _resolve_security_trace_context(*, session_id: str | None = None) -> tuple[str, str | None]:
    resolved_session_id = session_id or os.environ.get("TRW_SESSION_ID", "").strip() or "memory-security"
    run_id = os.environ.get("TRW_RUN_ID", "").strip() or None
    return resolved_session_id, run_id


def apply_sec001_intake(
    entry: MemoryEntry,
    *,
    config: MemoryConfig,
    session_id: str | None,
    trw_dir: Path | None = None,
) -> MemoryEntry:
    anchor_dir = trw_dir or _discover_anchor(config)
    verify_defaults(config, trw_dir=anchor_dir)
    trust_metadata = {**entry.metadata, "source_identity": entry.source_identity}
    if config.enable_trust_scoring:
        try:
            trust_result = score_intake(
                f"{entry.content}\n{entry.detail}",
                trust_metadata,
                observe_mode=config.trust_scoring_mode == "observe",
                trw_dir=anchor_dir,
            )
        except Exception as exc:
            raise ScorerUnavailableError(f"trust scorer unavailable: {exc}") from exc
        updated_metadata = {
            **entry.metadata,
            "trust_score": f"{trust_result.score:.4f}",
            "trust_flags": "|".join(trust_result.reasons),
        }
        entry = entry.model_copy(update={"metadata": updated_metadata})
        would_be_decision = next(
            (reason.removeprefix("WOULD-BE:") for reason in trust_result.reasons if reason.startswith("WOULD-BE:")),
            trust_result.decision,
        )
        telemetry_session_id, telemetry_run_id = _resolve_security_trace_context(
            session_id=session_id or _resolve_provenance_session_id(entry, session_id)
        )
        emit_security_event(
            config,
            emitter="trust_scorer",
            session_id=telemetry_session_id,
            run_id=telemetry_run_id,
            payload={
                "event_name": "trust_score_decision",
                "entry_id": entry.id,
                "namespace": entry.namespace,
                "mode": config.trust_scoring_mode,
                "score": trust_result.score,
                "decision": trust_result.decision,
                "would_be_decision": would_be_decision,
                "flags": list(trust_result.reasons),
                "traceability": build_security_traceability(
                    live_path="security.runtime.prepare_entry_for_store",
                    requirement_ids=["FR-001", "FR-008", "NFR-010", "NFR-011"],
                ),
            },
        )
        if config.trust_scoring_mode == "strict" and trust_result.score < config.trust_score_threshold:
            from trw_memory.exceptions import PoisoningError

            raise PoisoningError("trust score below threshold", reason="trust_score_below_threshold")
        if config.trust_scoring_mode == "enforce" and trust_result.score < config.trust_score_threshold:
            entry = quarantine_entry(entry)

    # Provenance hash + signature moved to _apply_provenance_hash (PRD-DIST-2046 c793)
    # so it runs AFTER _apply_runtime_pii_policy, ensuring the stored hash reflects
    # the stored content (eliminating the c792 hash_pin_drift recall-time block).
    return entry


def apply_provenance_hash(
    entry: MemoryEntry,
    *,
    config: MemoryConfig,
    session_id: str | None,
    trw_dir: Path | None = None,
) -> MemoryEntry:
    """Compute the provenance content hash + signature on the FINAL stored content.

    PRD-DIST-2046 c793: must be called AFTER _apply_runtime_pii_policy in
    prepare_entry_for_store so the stored hash reflects what's actually
    stored. Previously this step ran inside _apply_sec001_intake (before
    PII redaction), causing hash drift at recall time when PII modified
    content (c792 root cause: 12/39 baseline drops on c763 via
    filter_recall_window hash_pin_drift block).
    """
    if not config.provenance_required:
        return entry
    anchor_dir = trw_dir or _discover_anchor(config)
    try:
        from trw_memory.security.keys import get_or_create_ed25519_key_at_path

        signing_key = get_or_create_ed25519_key_at_path(
            resolve_security_path(
                config,
                "provenance_signing_key_path",
                trw_dir=anchor_dir,
                create_parent=True,
            )
        )
        if signing_key is None:
            raise ProvenanceKeyUnavailableError("provenance signing key unavailable")
    except Exception as exc:
        if isinstance(exc, ProvenanceKeyUnavailableError):
            raise
        raise ProvenanceKeyUnavailableError(f"unable to load provenance key: {exc}") from exc
    provenance_metadata = build_entry_provenance(
        learning_id=entry.id,
        content=entry.content,
        detail=entry.detail,
        author=_actor_for_entry(entry),
        session_id=_resolve_provenance_session_id(entry, session_id),
        ts=datetime.now(timezone.utc).isoformat(),
        signing_key=signing_key,
    )
    return entry.model_copy(update={"metadata": {**entry.metadata, **provenance_metadata}})
