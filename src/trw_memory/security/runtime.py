# ruff: noqa: E402,F401,I001
"""Shared runtime security helpers for store/search/forget paths."""

from __future__ import annotations

import hashlib
import threading
from collections import deque
from math import isfinite
from pathlib import Path
from time import time

import structlog

from trw_memory.exceptions import RateLimitError
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.validation import DEFAULT_NAMESPACE
from trw_memory.security.audit import AuditLog
from trw_memory.security.provenance import derive_verify_key, verify_entry_provenance
from trw_memory.security.startup import resolve_security_path
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import lock_for_rmw, read_yaml, write_yaml

# Hard cap on the once-per-process maintenance dedup set. A long-lived process
# (or a test suite) that touches many distinct (audit_log_path, retention_days)
# tuples would otherwise grow this set without bound. On overflow we clear it:
# the only cost is re-running an idempotent retention compaction for a key whose
# membership was evicted, never a correctness problem. This makes the
# ``bounded`` claim in ``security_maintenance_status`` provably true.
_AUDIT_MAINTENANCE_CACHE_MAX = 256
_AUDIT_MAINTENANCE_CACHE: set[str] = set()
_AUDIT_MAINTENANCE_QUEUE: deque[str] = deque(maxlen=128)
_AUDIT_MAINTENANCE_LOCK = threading.RLock()
_MAX_LIVE_RATE_LIMIT_SESSIONS = 10_000
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


# The store intake body (prepare_entry_for_store) is an explicit ORDERED stage
# pipeline in _runtime_pipeline.py (order is semantic — see that module). It is
# re-exported at the bottom of this facade so import sites keep working.

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

    # Hash a caller-controlled long ID instead of truncating it. Truncation made
    # distinct IDs sharing the first 256 characters collide into one bucket.
    if len(session_id) > 256:
        session_id = "sha256:" + hashlib.sha256(session_id.encode()).hexdigest()

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
                recent_for_session: list[float] = []
                for item in value:
                    if not isinstance(item, (int, float)):
                        continue
                    stamp = float(item)
                    age = now - stamp
                    if isfinite(stamp) and 0.0 <= age < 60.0:
                        recent_for_session.append(stamp)
                if recent_for_session:
                    sessions[key] = recent_for_session

        recent = sessions.get(session_id, [])
        if session_id not in sessions and len(sessions) >= _MAX_LIVE_RATE_LIMIT_SESSIONS:
            oldest = min((stamp for values in sessions.values() for stamp in values), default=now)
            raise RateLimitError(
                "rate-limit session capacity exceeded",
                retry_after=min(60.0, max(0.0, 60.0 - (now - oldest))),
            )
        if len(recent) >= config.max_memory_writes_per_minute:
            retry_after = min(60.0, max(0.0, 60.0 - (now - recent[0]))) if recent else 60.0
            raise RateLimitError(
                f"session {session_id!r} exceeded {config.max_memory_writes_per_minute} memory writes per minute",
                retry_after=retry_after,
            )
        recent.append(now)
        sessions[session_id] = recent
        sessions = {key: value for key, value in sessions.items() if value}
        write_yaml(state_path, {"sessions": sessions})


# PII policy helpers extracted to _runtime_pii.py (PRD-DIST-245 batch 99).
# ``hash_path_components`` / ``redaction_marker`` were deleted with the
# write-path redaction action (2026-07-25) — see _runtime_pii.REDACTED_PII_TYPES.
from trw_memory.security._runtime_pii import (
    apply_runtime_pii_policy as _apply_runtime_pii_policy,
    flag_code_snippet as _flag_code_snippet,
    replace_pii as _replace_pii,
)


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
    """Drain one maintenance item outside scoring/write-lock paths.

    Caller holds ``_AUDIT_MAINTENANCE_LOCK`` (both call sites — ``ensure_*``
    and ``drain_security_maintenance_queue`` — hold it), so the bounded-set
    eviction below is race-free.
    """
    get_audit_log(config).compact(config.audit_retention_days)
    # Clear-on-overflow eviction keeps the dedup set bounded. Re-running an
    # idempotent compaction for an evicted key is the only cost.
    if len(_AUDIT_MAINTENANCE_CACHE) >= _AUDIT_MAINTENANCE_CACHE_MAX:
        logger.debug(
            "security_maintenance_cache_evicted",
            size=len(_AUDIT_MAINTENANCE_CACHE),
            cap=_AUDIT_MAINTENANCE_CACHE_MAX,
        )
        _AUDIT_MAINTENANCE_CACHE.clear()
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
        processed = len(_AUDIT_MAINTENANCE_CACHE)
        queued = len(_AUDIT_MAINTENANCE_QUEUE)
        queue_max = _AUDIT_MAINTENANCE_QUEUE.maxlen or 0
        # ``bounded`` reflects ACTUAL state: both the dedup set (clear-on-overflow
        # at _AUDIT_MAINTENANCE_CACHE_MAX) and the queue (deque maxlen) cannot
        # grow past their caps. It is no longer a hardcoded True.
        bounded = processed <= _AUDIT_MAINTENANCE_CACHE_MAX and queued <= queue_max
        return {
            "queued": queued,
            "processed": processed,
            "bounded": bounded,
            "max_queue_size": _AUDIT_MAINTENANCE_QUEUE.maxlen,
            "max_processed_size": _AUDIT_MAINTENANCE_CACHE_MAX,
        }


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
    namespace: str | None = None,
) -> dict[str, object]:
    entry = active_backend.get(learning_id, namespace=namespace if namespace is not None else DEFAULT_NAMESPACE)
    current_status = "active"
    if entry is None:
        quarantined = list_quarantined_entries(config, namespace=namespace, limit=10_000)
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
                resolve_security_path(
                    config,
                    "provenance_signing_key_path",
                    create_parent=True,
                    reject_leaf_symlink=True,
                )
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


# Canary FR-007 helpers extracted to _runtime_canary.py (PRD-DIST-245 batch 101).
from trw_memory.security._runtime_canary import (
    CANARY_STATE as _CANARY_STATE,
    initialize_canaries as initialize_canaries,
    probe_canaries as probe_canaries,
    should_halt_recalls as should_halt_recalls,
)


# Store-intake ordered pipeline extracted to _runtime_pipeline.py. Re-exported
# here so every import site (tests + MCP) keeps resolving these off the runtime
# facade. `_resolve_security_trace_context` in particular is looked up lazily by
# _runtime_canary as `runtime._resolve_security_trace_context`.
from trw_memory.security._runtime_pipeline import (
    PreparedStoreEntry as PreparedStoreEntry,
    prepare_entry_for_store as prepare_entry_for_store,
    _actor_for_entry as _actor_for_entry,
    _apply_provenance_hash as _apply_provenance_hash,
    _apply_sec001_intake as _apply_sec001_intake,
    _rejection_reason as _rejection_reason,
    _resolve_provenance_session_id as _resolve_provenance_session_id,
    _resolve_security_trace_context as _resolve_security_trace_context,
)
