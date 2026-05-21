# ruff: noqa: E402,F401,I001
"""Shared runtime security helpers for store/search/forget paths."""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import time

import structlog

from trw_memory.exceptions import (
    PIIBlockError,
    ProvenanceKeyUnavailableError,
    RateLimitError,
    ScorerUnavailableError,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.audit import AuditLog
from trw_memory.security.pii import PIIMatch
from trw_memory.security.poisoning import quarantine_entry, validate_entry_payload
from trw_memory.security.provenance import build_entry_provenance, derive_verify_key, verify_entry_provenance
from trw_memory.security.startup import _discover_anchor, resolve_security_path, verify_defaults
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event
from trw_memory.security.trust_scorer import score_intake
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import lock_for_rmw, read_yaml, write_yaml

_AUDIT_MAINTENANCE_CACHE: set[str] = set()
_AUDIT_MAINTENANCE_QUEUE: deque[str] = deque(maxlen=128)
_AUDIT_MAINTENANCE_LOCK = threading.RLock()
logger = structlog.get_logger(__name__)


# Anomaly-stats helpers extracted to _runtime_anomaly.py
# (PRD-DIST-245 batch 102). Re-exports preserve back-compat names.
from trw_memory.security._runtime_anomaly import (
    AnomalyStats as AnomalyStats,
    build_anomaly_stats as _build_anomaly_stats,
    score_anomaly as _score_entry_anomaly,
    series_stats as _series_stats,
    write_anomaly_stats as _write_anomaly_stats,
)


@dataclass(frozen=True)
class PreparedStoreEntry:
    """Entry plus the security decisions made before persistence."""

    entry: MemoryEntry
    op: str
    pii_matches: tuple[PIIMatch, ...]
    quarantined: bool = False
    anomaly_dimension: str = ""
    anomaly_z_score: float = 0.0


def get_audit_log(config: MemoryConfig) -> AuditLog:
    """Return the configured audit log."""
    return AuditLog(Path(config.audit_log_path), fsync=config.fsync_on_append)


def append_audit_event(
    config: MemoryConfig,
    op: str,
    *,
    entry_id: str = "",
    actor: str = "",
    namespace: str = "default",
    data: dict[str, object] | None = None,
) -> None:
    """Append an audit event when auditing is enabled."""
    if not config.audit_enabled:
        return
    ensure_security_maintenance(config)
    get_audit_log(config).append(op, entry_id=entry_id, actor=actor, namespace=namespace, data=data or {})


def _should_bypass_anomaly_quarantine(entry: MemoryEntry, config: MemoryConfig) -> bool:
    """Return True when ``entry.metadata['source']`` starts with one of the
    configured bypass prefixes (PRD-DIST-2045).

    The bypass exists for source-grounded automated ingestion paths whose
    producer pipeline has already validated record provenance (assertions,
    file paths, commit SHAs). Empty prefix list disables the bypass and
    restores the original "every write goes through anomaly quarantine"
    behavior. PRD-SEC-001 trust-scoring and PII redaction still apply.
    """

    prefixes = config.anomaly_bypass_source_prefixes
    if not prefixes:
        return False
    metadata_source = entry.metadata.get("source", "")
    if not metadata_source:
        return False
    return any(metadata_source.startswith(prefix) for prefix in prefixes)


def prepare_entry_for_store(
    entry: MemoryEntry,
    *,
    backend: StorageBackend,
    config: MemoryConfig,
    session_id: str | None = None,
    trw_dir: Path | None = None,
) -> PreparedStoreEntry:
    """Apply rate limits, PII handling, and anomaly scoring before a write."""
    ensure_security_maintenance(config)
    actor = _actor_for_entry(entry)
    existing = backend.get(entry.id)
    op = "update" if existing is not None and existing.namespace == entry.namespace else "store"
    flagged_entry = _flag_code_snippet(entry)
    secured_entry = _apply_sec001_intake(flagged_entry, config=config, session_id=session_id, trw_dir=trw_dir)
    if secured_entry.metadata.get("quarantined") == "true":
        trust_score = float(secured_entry.metadata.get("trust_score", "0.0") or "0.0")
        return PreparedStoreEntry(
            entry=secured_entry,
            op=op,
            pii_matches=(),
            quarantined=True,
            anomaly_dimension="trust_score",
            anomaly_z_score=trust_score,
        )

    try:
        enforce_write_rate_limit(
            config, session_id=session_id, actor=actor, namespace=entry.namespace, entry_id=entry.id
        )
        validate_entry_payload(secured_entry, max_chars=config.max_entry_chars)
        secured_entry, pii_matches = _apply_runtime_pii_policy(secured_entry, config)
        # PRD-DIST-2046 c793: compute provenance hash AFTER PII redaction so the
        # stored hash reflects the stored content. Prevents recall-time
        # hash_pin_drift block in filter_recall_window mode=redact.
        secured_entry = _apply_provenance_hash(
            secured_entry,
            config=config,
            session_id=session_id,
            trw_dir=trw_dir,
        )
        anomaly, anomaly_stats = _score_entry_anomaly(
            secured_entry,
            backend,
            config=config,
        )
    except Exception as exc:
        append_audit_event(
            config,
            "store_rejected",
            entry_id=entry.id,
            actor=actor,
            namespace=entry.namespace,
            data={
                "reason": _rejection_reason(exc),
                "session_id": session_id,
                "retry_after": getattr(exc, "retry_after", 0.0),
                "failed_fields": getattr(exc, "failed_fields", []),
            },
        )
        raise

    _write_anomaly_stats(config, anomaly_stats)

    if anomaly is None or not config.poisoning_detection_enabled:
        return PreparedStoreEntry(entry=secured_entry, op=op, pii_matches=tuple(pii_matches))

    if _should_bypass_anomaly_quarantine(secured_entry, config):
        dimension, z_score = anomaly
        logger.debug(
            "anomaly_quarantine_bypass",
            entry_id=secured_entry.id,
            namespace=secured_entry.namespace,
            metadata_source=secured_entry.metadata.get("source", ""),
            anomaly_dimension=dimension,
            z_score=z_score,
            outcome="bypassed_via_anomaly_bypass_source_prefixes",
        )
        return PreparedStoreEntry(entry=secured_entry, op=op, pii_matches=tuple(pii_matches))

    dimension, z_score = anomaly
    quarantined = quarantine_entry(
        secured_entry.model_copy(
            update={
                "metadata": {
                    **secured_entry.metadata,
                    "anomaly_dimension": dimension,
                    "z_score": f"{z_score:.2f}",
                }
            }
        )
    )
    return PreparedStoreEntry(
        entry=quarantined,
        op=op,
        pii_matches=tuple(pii_matches),
        quarantined=True,
        anomaly_dimension=dimension,
        anomaly_z_score=z_score,
    )


# Quarantine + review-log helpers extracted to _runtime_quarantine.py
# (PRD-DIST-245 batch 100). Re-exports preserve back-compat names.
from trw_memory.security._runtime_quarantine import (
    delete_quarantined_entries as delete_quarantined_entries,
    list_quarantined_entries as list_quarantined_entries,
    store_quarantined_entry as store_quarantined_entry,
)


def enforce_write_rate_limit(
    config: MemoryConfig,
    *,
    session_id: str | None,
    actor: str,
    namespace: str,
    entry_id: str,
) -> None:
    """Apply the configured rolling write-rate limit."""
    if not session_id or config.max_memory_writes_per_minute <= 0:
        return

    state_path = Path(config.rate_limit_state_path)
    now = time()
    with lock_for_rmw(state_path):
        raw_state: dict[str, object] = read_yaml(state_path) if state_path.exists() else {}
        sessions_raw = raw_state.get("sessions", {})
        sessions: dict[str, list[float]] = {}
        if isinstance(sessions_raw, dict):
            for key, value in sessions_raw.items():
                if not isinstance(key, str) or not isinstance(value, list):
                    continue
                recent_for_session = [
                    float(item) for item in value if isinstance(item, (int, float)) and now - float(item) < 60.0
                ]
                if recent_for_session:
                    sessions[key] = recent_for_session

        recent = [stamp for stamp in sessions.get(session_id, []) if now - stamp < 60.0]
        if len(recent) >= config.max_memory_writes_per_minute:
            retry_after = max(0.0, 60.0 - (now - recent[0])) if recent else 60.0
            raise RateLimitError(
                f"session {session_id!r} exceeded {config.max_memory_writes_per_minute} memory writes per minute",
                retry_after=retry_after,
            )
        recent.append(now)
        sessions[session_id] = recent
        sessions = {key: value for key, value in sessions.items() if value}
        write_yaml(state_path, {"sessions": sessions})


# PII redaction helpers extracted to _runtime_pii.py (PRD-DIST-245 batch 99).
from trw_memory.security._runtime_pii import (
    apply_runtime_pii_policy as _apply_runtime_pii_policy,
    flag_code_snippet as _flag_code_snippet,
    hash_path_components as _hash_path_components,
    redaction_marker as _redaction_marker,
    replace_pii as _replace_pii,
)


def _actor_for_entry(entry: MemoryEntry) -> str:
    return entry.source_identity or entry.source or "system"


from trw_memory.security._runtime_quarantine import (
    open_quarantine_backend as _open_quarantine_backend,
    quarantine_namespace_dir as _quarantine_namespace_dir,
    read_namespace_metadata as _read_namespace_metadata,
)


def ensure_security_maintenance(config: MemoryConfig) -> None:
    """Run or enqueue once-per-process audit retention maintenance for a config path."""
    cache_key = f"{config.audit_log_path}:{config.audit_retention_days}"
    with _AUDIT_MAINTENANCE_LOCK:
        if cache_key in _AUDIT_MAINTENANCE_CACHE:
            return
        if not config.security_maintenance_inline:
            if cache_key not in _AUDIT_MAINTENANCE_QUEUE:
                _AUDIT_MAINTENANCE_QUEUE.append(cache_key)
                logger.debug("security_maintenance_enqueued", audit_log_path=config.audit_log_path)
            return
        _drain_security_maintenance_key(config, cache_key)


def _drain_security_maintenance_key(config: MemoryConfig, cache_key: str) -> None:
    """Drain one maintenance item outside scoring/write-lock paths."""
    get_audit_log(config).compact(config.audit_retention_days)
    _AUDIT_MAINTENANCE_CACHE.add(cache_key)


def drain_security_maintenance_queue(config: MemoryConfig) -> dict[str, object]:
    """Drain queued audit-retention maintenance and report compact status."""
    drained = 0
    cache_key = f"{config.audit_log_path}:{config.audit_retention_days}"
    retained: list[str] = []
    with _AUDIT_MAINTENANCE_LOCK:
        while _AUDIT_MAINTENANCE_QUEUE:
            queued_key = _AUDIT_MAINTENANCE_QUEUE.popleft()
            if queued_key != cache_key:
                retained.append(queued_key)
                continue
            _drain_security_maintenance_key(config, queued_key)
            drained += 1
        _AUDIT_MAINTENANCE_QUEUE.extend(retained)
        queued = len(_AUDIT_MAINTENANCE_QUEUE)
    return {"drained": drained, "queued": queued}


def security_maintenance_status() -> dict[str, object]:
    """Return compact process-local maintenance queue state."""
    with _AUDIT_MAINTENANCE_LOCK:
        return {
            "queued": len(_AUDIT_MAINTENANCE_QUEUE),
            "processed": len(_AUDIT_MAINTENANCE_CACHE),
            "bounded": True,
            "max_queue_size": _AUDIT_MAINTENANCE_QUEUE.maxlen,
        }


def _rejection_reason(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "rate_limited"
    if isinstance(exc, PIIBlockError):
        return "pii_detected"
    if exc.__class__.__name__ == "SchemaValidationError":
        return "schema_invalid"
    return getattr(exc, "reason", exc.__class__.__name__)


from trw_memory.security._runtime_quarantine import (
    append_review_log as _append_review_log,
    get_status_history as get_status_history,
    review_quarantined_entry as review_quarantined_entry,
)


def audit_entry(
    config: MemoryConfig,
    *,
    learning_id: str,
    active_backend: StorageBackend,
) -> dict[str, object]:
    entry = active_backend.get(learning_id)
    current_status = "active"
    if entry is None:
        quarantined = list_quarantined_entries(config, limit=10_000)
        entry = next((candidate for candidate in quarantined if candidate.id == learning_id), None)
        current_status = "quarantined" if entry is not None else "legacy_unsigned"
    if entry is None:
        return {"learning_id": learning_id, "status": "not_found", "status_history": []}
    metadata = dict(entry.metadata)
    if not metadata.get("provenance_signature"):
        return {
            "learning_id": learning_id,
            "status": "legacy_unsigned",
            "status_history": get_status_history(config, learning_id),
        }
    verify_key = None
    try:
        from trw_memory.security.keys import get_or_create_ed25519_key_at_path

        verify_key = derive_verify_key(
            get_or_create_ed25519_key_at_path(
                resolve_security_path(config, "provenance_signing_key_path", create_parent=True)
            )
        )
    except Exception:
        verify_key = None
    return {
        "learning_id": learning_id,
        "status": current_status,
        "author": metadata.get("provenance_author", entry.source_identity),
        "session_id": metadata.get("provenance_session_id", ""),
        "ts": metadata.get("provenance_ts", ""),
        "content_hash": metadata.get("provenance_content_hash", ""),
        "signature": metadata.get("provenance_signature", ""),
        "verified": verify_entry_provenance(entry, verify_key),
        "status_history": get_status_history(config, learning_id),
    }


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


def _apply_provenance_hash(
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


# Canary FR-007 helpers extracted to _runtime_canary.py (PRD-DIST-245 batch 101).
from trw_memory.security._runtime_canary import (
    CANARY_STATE as _CANARY_STATE,
    initialize_canaries as initialize_canaries,
    probe_canaries as probe_canaries,
    should_halt_recalls as should_halt_recalls,
)
