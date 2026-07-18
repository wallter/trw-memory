# ruff: noqa: I001
"""Ordered intake pipeline for ``prepare_entry_for_store``.

Belongs to the ``runtime.py`` facade: ``runtime`` re-exports
``prepare_entry_for_store``, ``PreparedStoreEntry`` and the SEC-001 intake
helpers from here with ``X as X`` so every existing import site keeps working.

The store intake is a sequence of check stages whose ORDER IS SEMANTIC. This
module encodes that order as *data* — two explicit ordered stage lists,
``_PRE_QUARANTINE_STAGES`` and ``_AUDITED_STAGES`` — rather than as an opaque
straight-line function body, so the sequence is auditable and test-pinnable
(``tests/test_security_intake_pipeline_order.py``).

Order-dependence a naive uniform pipeline would break (do NOT reorder):
  1. classify (op/actor) reads pre-mutation backend state BEFORE any model_copy.
  2. trust-quarantine short-circuit signs provenance on UN-redacted content and
     SKIPS rate-limit / PII / anomaly — running those uniformly on a held entry
     would change stored content + provenance timing.
  3. PII redaction (``_stage_pii_policy``) MUST precede provenance hashing
     (``_stage_provenance_hash``) so the stored hash reflects stored content
     (PRD-DIST-2046 c793 — prevents recall-time hash_pin_drift).
  4. rate-limit .. anomaly all sit inside ONE try whose except emits the
     ``store_rejected`` audit (carrying ``retry_after`` / ``failed_fields``)
     then re-raises. Audit is NOT pushed into individual stages.
  5. the anomaly-stats write runs ONLY on success (after the try).

``enforce_write_rate_limit`` / ``append_audit_event`` /
``ensure_security_maintenance`` deliberately stay in ``runtime`` and are reached
via ``_rt()`` at call time — this preserves the ``trw_memory.security.runtime.time``
monkeypatch seam (``enforce_write_rate_limit`` reads the ``time`` global of the
``runtime`` module where it is defined).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from types import ModuleType

import structlog

from trw_memory.exceptions import ProvenanceKeyUnavailableError, ScorerUnavailableError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.pii import PIIMatch
from trw_memory.security.poisoning import MIN_ANOMALY_BASELINE, quarantine_entry, validate_entry_payload
from trw_memory.security.provenance import build_entry_provenance
from trw_memory.security.startup import _discover_anchor, resolve_security_path, verify_defaults
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event
from trw_memory.security.trust_scorer import score_intake
from trw_memory.storage.interface import StorageBackend

from trw_memory.security._runtime_anomaly import (
    AnomalyStats,
    score_anomaly as _score_entry_anomaly,
    write_anomaly_stats as _write_anomaly_stats,
)
from trw_memory.security._runtime_pii import (
    apply_runtime_pii_policy as _apply_runtime_pii_policy,
    flag_code_snippet as _flag_code_snippet,
)

logger = structlog.get_logger(__name__)


def _rt() -> ModuleType:
    """Return the ``runtime`` facade module (call-time, to preserve seams)."""
    from trw_memory.security import runtime as _runtime

    return _runtime


@dataclass(frozen=True)
class PreparedStoreEntry:
    """Entry plus the security decisions made before persistence."""

    entry: MemoryEntry
    op: str
    pii_matches: tuple[PIIMatch, ...]
    quarantined: bool = False
    anomaly_dimension: str = ""
    anomaly_z_score: float = 0.0


@dataclass
class _StoreContext:
    """Mutable state threaded through the ordered intake stages."""

    entry: MemoryEntry
    backend: StorageBackend
    config: MemoryConfig
    session_id: str | None
    trw_dir: Path | None
    actor: str = ""
    op: str = "store"
    pii_matches: tuple[PIIMatch, ...] = ()
    anomaly: tuple[str, float] | None = None
    anomaly_stats: AnomalyStats | None = None


# --------------------------------------------------------------------------- #
# Small SEC-001 helpers (migrated from runtime.py; re-exported there for the   #
# _runtime_canary lazy lookup of _resolve_security_trace_context).             #
# --------------------------------------------------------------------------- #
def _actor_for_entry(entry: MemoryEntry) -> str:
    return entry.source_identity or entry.source or "system"


def _intake_scannable_text(entry: MemoryEntry) -> str:
    """Concatenate every free-form user-writable text field SEC-001 intake must scan.

    Beyond ``content`` + ``detail`` this folds in each ``entry.evidence`` item and
    each ``Assertion.last_evidence`` string. These fields are publicly reachable via
    ``memory_store`` and are persisted verbatim, so without folding them in here a
    caller could smuggle poisoning-pattern text past the trust scorer (release-blocker
    SEC-001). The PII stage covers the same surface independently (see
    ``_runtime_pii.apply_runtime_pii_policy``).
    """
    parts: list[str] = [entry.content, entry.detail]
    parts.extend(entry.evidence)
    parts.extend(assertion.last_evidence for assertion in entry.assertions)
    return "\n".join(part for part in parts if part)


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


def _rejection_reason(exc: Exception) -> str:
    from trw_memory.exceptions import PIIBlockError, RateLimitError

    if isinstance(exc, RateLimitError):
        return "rate_limited"
    if isinstance(exc, PIIBlockError):
        return "pii_detected"
    if exc.__class__.__name__ == "SchemaValidationError":
        return "schema_invalid"
    return getattr(exc, "reason", exc.__class__.__name__)


def _apply_sec001_intake(
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
                _intake_scannable_text(entry),
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

    # Provenance hash + signature run in _apply_provenance_hash (PRD-DIST-2046
    # c793) so they follow _apply_runtime_pii_policy and the stored hash
    # reflects the stored content.
    return entry


def _apply_provenance_hash(
    entry: MemoryEntry,
    *,
    config: MemoryConfig,
    session_id: str | None,
    trw_dir: Path | None = None,
) -> MemoryEntry:
    """Compute the provenance content hash + signature on the FINAL stored content.

    PRD-DIST-2046 c793: must be called AFTER _apply_runtime_pii_policy so the
    stored hash reflects what is actually stored (eliminating the c792
    filter_recall_window hash_pin_drift recall-time block).
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
                reject_leaf_symlink=True,
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


# --------------------------------------------------------------------------- #
# Ordered stages. Sequence is SEMANTIC — see module docstring. Do not reorder. #
# --------------------------------------------------------------------------- #
def _stage_queue_drain(ctx: _StoreContext) -> None:
    _rt().ensure_security_maintenance(ctx.config)


def _stage_classify(ctx: _StoreContext) -> None:
    # Reads pre-mutation backend state; MUST precede any model_copy below.
    ctx.actor = _actor_for_entry(ctx.entry)
    existing = ctx.backend.get(ctx.entry.id)
    ctx.op = "update" if existing is not None and existing.namespace == ctx.entry.namespace else "store"


def _stage_flag_code(ctx: _StoreContext) -> None:
    ctx.entry = _flag_code_snippet(ctx.entry)


def _stage_trust_intake(ctx: _StoreContext) -> None:
    ctx.entry = _apply_sec001_intake(ctx.entry, config=ctx.config, session_id=ctx.session_id, trw_dir=ctx.trw_dir)


_PRE_QUARANTINE_STAGES: list[Callable[[_StoreContext], None]] = [
    _stage_queue_drain,
    _stage_classify,
    _stage_flag_code,
    _stage_trust_intake,
]


def _stage_rate_limit(ctx: _StoreContext) -> None:
    _rt().enforce_write_rate_limit(
        ctx.config,
        session_id=ctx.session_id,
        actor=ctx.actor,
        namespace=ctx.entry.namespace,
        entry_id=ctx.entry.id,
    )


def _stage_validate_payload(ctx: _StoreContext) -> None:
    validate_entry_payload(ctx.entry, max_chars=ctx.config.max_entry_chars)


def _stage_pii_policy(ctx: _StoreContext) -> None:
    ctx.entry, pii_matches = _apply_runtime_pii_policy(ctx.entry, ctx.config)
    ctx.pii_matches = tuple(pii_matches)


def _stage_provenance_hash(ctx: _StoreContext) -> None:
    # PRD-DIST-2046 c793: MUST follow _stage_pii_policy.
    ctx.entry = _apply_provenance_hash(ctx.entry, config=ctx.config, session_id=ctx.session_id, trw_dir=ctx.trw_dir)


def _stage_anomaly_score(ctx: _StoreContext) -> None:
    ctx.anomaly, ctx.anomaly_stats = _score_entry_anomaly(ctx.entry, ctx.backend, config=ctx.config)


_AUDITED_STAGES: list[Callable[[_StoreContext], None]] = [
    _stage_rate_limit,
    _stage_validate_payload,
    _stage_pii_policy,
    _stage_provenance_hash,
    _stage_anomaly_score,
]


def _finalize_trust_quarantine(ctx: _StoreContext) -> PreparedStoreEntry:
    # A trust-score quarantine must still carry provenance so it stays auditable
    # (audit_entry() reports `quarantined` + verified rather than
    # `legacy_unsigned`). No PII redaction runs on this path, so signing the
    # quarantined content as-is keeps hash + signature internally consistent.
    # Best-effort: if the signing key is unavailable, keep the (still-quarantined)
    # entry unsigned rather than failing the store — a held entry is never
    # recalled until a review approves it.
    try:
        ctx.entry = _apply_provenance_hash(ctx.entry, config=ctx.config, session_id=ctx.session_id, trw_dir=ctx.trw_dir)
    except ProvenanceKeyUnavailableError:
        logger.warning(
            "quarantine_provenance_sign_skipped",
            entry_id=ctx.entry.id,
            namespace=ctx.entry.namespace,
            outcome="signing_key_unavailable",
        )
    trust_score = float(ctx.entry.metadata.get("trust_score", "0.0") or "0.0")
    return PreparedStoreEntry(
        entry=ctx.entry,
        op=ctx.op,
        pii_matches=(),
        quarantined=True,
        anomaly_dimension="trust_score",
        anomaly_z_score=trust_score,
    )


def _finalize_anomaly_decision(ctx: _StoreContext) -> PreparedStoreEntry:
    config = ctx.config
    # _stage_anomaly_score always sets anomaly_stats. A bare `assert` would be
    # stripped under `python -O`, silently admitting an unscored entry into the
    # store; on a security intake path the guard must fail closed for real.
    if ctx.anomaly_stats is None:
        raise ScorerUnavailableError("anomaly_stats missing after the anomaly-scoring stage")
    if ctx.anomaly is None or not config.poisoning_detection_enabled:
        # trw-memory-10: emit an AUDIT event for the sub-baseline condition so a
        # namespace with < MIN_ANOMALY_BASELINE clean entries (detector silently
        # skipped) is visible to audit-trail analysis, not only structured logs.
        if (
            ctx.anomaly is None
            and config.poisoning_detection_enabled
            and ctx.anomaly_stats.sample_count < MIN_ANOMALY_BASELINE
        ):
            _rt().append_audit_event(
                config,
                "anomaly_baseline_insufficient",
                entry_id=ctx.entry.id,
                actor=ctx.actor,
                namespace=ctx.entry.namespace,
                data={
                    "sample_count": ctx.anomaly_stats.sample_count,
                    "min_baseline": MIN_ANOMALY_BASELINE,
                    "reason": "below_statistical_baseline",
                },
            )
        return PreparedStoreEntry(entry=ctx.entry, op=ctx.op, pii_matches=ctx.pii_matches)

    dimension, z_score = ctx.anomaly

    # SEC-001 observe-mode (documented default): an anomaly was detected but the
    # detector is NOT promoted to enforce. Record + audit the would-be quarantine
    # and store normally. Quarantine only fires under explicit enforce-mode.
    if config.poisoning_detection_mode != "enforce":
        logger.info(
            "anomaly_observed_not_quarantined",
            op="store",
            outcome="observe_mode",
            entry_id=ctx.entry.id,
            namespace=ctx.entry.namespace,
            anomaly_dimension=dimension,
            z_score=z_score,
        )
        _rt().append_audit_event(
            config,
            "anomaly_observed",
            entry_id=ctx.entry.id,
            actor=ctx.actor,
            namespace=ctx.entry.namespace,
            data={
                "anomaly_dimension": dimension,
                "z_score": z_score,
                "mode": "observe",
                "would_quarantine": True,
            },
        )
        return PreparedStoreEntry(
            entry=ctx.entry,
            op=ctx.op,
            pii_matches=ctx.pii_matches,
            quarantined=False,
            anomaly_dimension=dimension,
            anomaly_z_score=z_score,
        )

    quarantined = quarantine_entry(
        ctx.entry.model_copy(
            update={
                "metadata": {
                    **ctx.entry.metadata,
                    "anomaly_dimension": dimension,
                    "z_score": f"{z_score:.2f}",
                }
            }
        )
    )
    return PreparedStoreEntry(
        entry=quarantined,
        op=ctx.op,
        pii_matches=ctx.pii_matches,
        quarantined=True,
        anomaly_dimension=dimension,
        anomaly_z_score=z_score,
    )


def prepare_entry_for_store(
    entry: MemoryEntry,
    *,
    backend: StorageBackend,
    config: MemoryConfig,
    session_id: str | None = None,
    trw_dir: Path | None = None,
) -> PreparedStoreEntry:
    """Apply rate limits, PII handling, and anomaly scoring before a write."""
    ctx = _StoreContext(entry=entry, backend=backend, config=config, session_id=session_id, trw_dir=trw_dir)
    for stage in _PRE_QUARANTINE_STAGES:
        stage(ctx)

    if ctx.entry.metadata.get("quarantined") == "true":
        return _finalize_trust_quarantine(ctx)

    try:
        for stage in _AUDITED_STAGES:
            stage(ctx)
    except Exception as exc:
        _rt().append_audit_event(
            config,
            "store_rejected",
            entry_id=entry.id,
            actor=ctx.actor,
            namespace=entry.namespace,
            data={
                "reason": _rejection_reason(exc),
                "session_id": session_id,
                "retry_after": getattr(exc, "retry_after", 0.0),
                "failed_fields": getattr(exc, "failed_fields", []),
            },
        )
        raise

    # See _finalize_anomaly_decision: fail closed rather than assert, so the
    # guard survives `python -O`.
    if ctx.anomaly_stats is None:
        raise ScorerUnavailableError("anomaly_stats missing after the anomaly-scoring stage")
    _write_anomaly_stats(config, ctx.anomaly_stats)
    return _finalize_anomaly_decision(ctx)
