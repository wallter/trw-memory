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
  2. PII redaction (``_stage_pii_policy``) MUST precede provenance hashing
     (``_stage_provenance_hash``) so the stored hash reflects stored content
     (PRD-DIST-2046 c793 — prevents recall-time hash_pin_drift).
  3. rate-limit .. provenance all sit inside ONE try whose except emits the
     ``store_rejected`` audit (carrying ``retry_after`` / ``failed_fields``)
     then re-raises. Audit is NOT pushed into individual stages.

The intake trust scorer, the statistical anomaly quarantine and their observe clock were removed (UF-MEM-03,
2026-10-01): in 15 days of observe data they caught nothing and flagged only legitimate learnings. The write
gate (``validate_entry_payload``) still refuses known injection shapes, and PII/API keys are still blocked.

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

from trw_memory.exceptions import PIIBlockError, ProvenanceKeyUnavailableError, RateLimitError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.pii import PIIMatch
from trw_memory.security.poisoning import (
    strip_reserved_metadata as _strip,
    validate_entry_payload,
)
from trw_memory.security.provenance import build_entry_provenance
from trw_memory.security.startup import _discover_anchor, resolve_security_path, verify_defaults
from trw_memory.storage.interface import StorageBackend

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
    rate_receipt: float | None = None  # the charged slot, refunded if the write then stores nothing


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
    rate_receipt: float | None = None


# --------------------------------------------------------------------------- #
# Small SEC-001 helpers (migrated from runtime.py; re-exported there for the   #
# _runtime_canary lazy lookup of _resolve_security_trace_context).             #
# --------------------------------------------------------------------------- #
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


def _rejection_reason(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "rate_limited"
    if isinstance(exc, PIIBlockError):
        return "pii_detected"
    if exc.__class__.__name__ == "SchemaValidationError":
        return "schema_invalid"
    return getattr(exc, "reason", exc.__class__.__name__)


def _apply_sec001_intake(entry: MemoryEntry, *, config: MemoryConfig, trw_dir: Path | None = None) -> MemoryEntry:
    """Fail closed when the SEC-001 security defaults are not in place (``verify_defaults``)."""
    verify_defaults(config, trw_dir=trw_dir or _discover_anchor(config))
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
    # PRD-CORE-245 FR03: the namespace predicate now lives in the read, so a
    # row with the same id in another namespace is not a candidate "update".
    existing = ctx.backend.get(ctx.entry.id, namespace=ctx.entry.namespace)
    # ``isinstance`` rather than ``is not None``: a backend that returns a
    # placeholder (a test double, a partially-initialised adapter) must not be
    # read as "this entry already exists" and silently downgrade a store to an
    # update. Mirrors ``_existing_entry_for_namespace``.
    ctx.op = "update" if isinstance(existing, MemoryEntry) else "store"


def _stage_flag_code(ctx: _StoreContext) -> None:
    ctx.entry = _flag_code_snippet(ctx.entry)


def _stage_verify_defaults(ctx: _StoreContext) -> None:
    ctx.entry = _apply_sec001_intake(ctx.entry, config=ctx.config, trw_dir=ctx.trw_dir)


_PRE_AUDIT_STAGES: list[Callable[[_StoreContext], None]] = [
    _stage_queue_drain,
    _stage_classify,
    _stage_flag_code,
    _stage_verify_defaults,
]


def _stage_rate_limit(ctx: _StoreContext) -> None:
    ctx.rate_receipt = _rt().enforce_write_rate_limit(
        ctx.config,
        session_id=ctx.session_id,
        actor=ctx.actor,
        namespace=ctx.entry.namespace,
        entry_id=ctx.entry.id,
    )


def _stage_validate_payload(ctx: _StoreContext) -> None:
    validate_entry_payload(
        ctx.entry,
        max_chars=ctx.config.max_entry_chars,
        min_evidence_items_for_verified=ctx.config.min_evidence_items_for_verified,
    )


def _stage_pii_policy(ctx: _StoreContext) -> None:
    ctx.entry, pii_matches = _apply_runtime_pii_policy(ctx.entry, ctx.config)
    ctx.pii_matches = tuple(pii_matches)


def _stage_provenance_hash(ctx: _StoreContext) -> None:
    # PRD-DIST-2046 c793: MUST follow _stage_pii_policy.
    ctx.entry = _apply_provenance_hash(ctx.entry, config=ctx.config, session_id=ctx.session_id, trw_dir=ctx.trw_dir)


_AUDITED_STAGES: list[Callable[[_StoreContext], None]] = [
    _stage_rate_limit,
    _stage_validate_payload,
    _stage_pii_policy,
    _stage_provenance_hash,
]


def prepare_entry_for_store(
    entry: MemoryEntry,
    *,
    backend: StorageBackend,
    config: MemoryConfig,
    session_id: str | None = None,
    trw_dir: Path | None = None,
) -> PreparedStoreEntry:
    """Apply the security defaults check, rate limits, the write gate, PII handling and provenance before a write.

    ``strip_reserved_metadata`` (Q1) runs UNCONDITIONALLY, before any stage — not as a list entry — so a
    caller-set ``quarantined``/etc. metadata key can never be stored as if the system had set it.
    """
    ctx = _StoreContext(entry=_strip(entry), backend=backend, config=config, session_id=session_id, trw_dir=trw_dir)
    for stage in _PRE_AUDIT_STAGES:
        stage(ctx)

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

    return PreparedStoreEntry(entry=ctx.entry, op=ctx.op, pii_matches=ctx.pii_matches, rate_receipt=ctx.rate_receipt)
