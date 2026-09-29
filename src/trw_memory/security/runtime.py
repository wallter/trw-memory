# ruff: noqa: E402,F401,I001
"""Shared runtime security helpers for store/search/forget paths."""

from __future__ import annotations

import hashlib
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import time

import structlog

from trw_memory._project_anchor import resolve_state_path
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


@dataclass(frozen=True)
class _QueuedMaintenance:
    """One config's audit log, queued while ``security_maintenance_inline`` was ``False``.

    Carries what :func:`_drain_security_maintenance_key` needs directly (not the
    ``MemoryConfig`` that enqueued it, which may be long gone by drain time).
    """

    audit_log_path: str
    retention_days: int
    fsync: bool

    @property
    def cache_key(self) -> str:
        return f"{self.audit_log_path}:{self.retention_days}"


_AUDIT_MAINTENANCE_QUEUE: deque[_QueuedMaintenance] = deque(maxlen=128)
_AUDIT_MAINTENANCE_LOCK = threading.RLock()
_MAX_LIVE_RATE_LIMIT_SESSIONS = 10_000
logger = structlog.get_logger(__name__)


def get_audit_log(config: MemoryConfig) -> AuditLog:
    """Return the configured audit log."""
    return AuditLog(resolve_state_path(config, "audit_log_path"), fsync=config.fsync_on_append)


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


# session key -> None (charged and admitted) or the RateLimitError its charge raised.
_WRITE_OPERATION: ContextVar[dict[str, RateLimitError | None] | None] = ContextVar("_WRITE_OPERATION", default=None)


@contextmanager
def single_write_operation() -> Iterator[None]:
    """Charge each writer session ONE rate-limit slot for everything written in this block.

    The limiter stops an agent flooding memory with many independent writes. A
    ``bulk_store`` batch is one caller operation, so it costs one slot per session,
    not one per row: per-row charging rejected every row after the tenth of a batch
    (10/min default). Inside the block the first charge of a session decides the
    outcome and every later row of that session shares it -- admitted rows stay
    admitted, and a refused session refuses ALL its rows with the same
    ``RateLimitError`` rather than admitting an arbitrary prefix.
    """
    token = _WRITE_OPERATION.set({})
    try:
        yield
    finally:
        _WRITE_OPERATION.reset(token)


def enforce_write_rate_limit(
    config: MemoryConfig,
    *,
    session_id: str | None,
    actor: str,
    namespace: str,
    entry_id: str,
) -> float | None:
    """Apply the rolling write-rate limit; once per session inside ``single_write_operation``.

    Returns the charged slot's receipt for :func:`refund_write_slot`, or ``None`` when
    nothing refundable was charged (no session, limiter off, or a batch's shared slot).
    """
    if not session_id or config.max_memory_writes_per_minute <= 0:
        return None
    session_id = _session_key(session_id)
    operation = _WRITE_OPERATION.get()
    if operation is not None and session_id in operation:
        prior = operation[session_id]
        if prior is not None:
            raise RateLimitError(str(prior), retry_after=prior.retry_after)
        return None
    try:
        receipt = _charge_write_slot(config, session_id)
    except RateLimitError as exc:
        if operation is not None:
            operation[session_id] = exc
        raise
    if operation is not None:
        operation[session_id] = None
        return None
    return receipt


def _session_key(session_id: str) -> str:
    # Hash a caller-controlled long ID instead of truncating it. Truncation made
    # distinct IDs sharing the first 256 characters collide into one bucket.
    return "sha256:" + hashlib.sha256(session_id.encode()).hexdigest() if len(session_id) > 256 else session_id


def refund_write_slot(config: MemoryConfig, *, session_id: str | None, receipt: float | None) -> None:
    """Return the slot a write charged when that write then stored nothing (B71-81).

    A ``memory_store`` refused as ``conflict`` is charged by the intake pipeline before
    its revision check, so without a refund repeated collisions on one id would delay
    the caller's retry. Only the stamp *receipt* names is removed, and only while it is
    still in the window: a charge that already expired frees nothing, so a stalled
    write can never refund a newer request's slot. Two equal stamps are interchangeable.
    """
    if not session_id or receipt is None:
        return
    key, state_path = _session_key(session_id), resolve_state_path(config, "rate_limit_state_path")
    with lock_for_rmw(state_path):
        state: dict[str, object] = read_yaml(state_path) if state_path.exists() else {}
        sessions = state.get("sessions")
        stamps = sessions.get(key) if isinstance(sessions, dict) else None
        if isinstance(stamps, list) and receipt in stamps:
            stamps.remove(receipt)
            write_yaml(state_path, state)


def _charge_write_slot(config: MemoryConfig, session_id: str) -> float:
    state_path = resolve_state_path(config, "rate_limit_state_path")
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
    return now


# PII policy helpers extracted to _runtime_pii.py (PRD-DIST-245 batch 99).
# ``hash_path_components`` / ``redaction_marker`` were deleted with the
# write-path redaction action (2026-07-25) — see _runtime_pii.REDACTED_PII_TYPES.
from trw_memory.security._runtime_pii import (
    flag_code_snippet as _flag_code_snippet,
)


def ensure_security_maintenance(config: MemoryConfig) -> None:
    """Run or enqueue once-per-process audit retention maintenance for a config path."""
    audit_log_path = str(resolve_state_path(config, "audit_log_path"))
    cache_key = f"{audit_log_path}:{config.audit_retention_days}"
    with _AUDIT_MAINTENANCE_LOCK:
        if cache_key in _AUDIT_MAINTENANCE_CACHE:
            return
        if not config.security_maintenance_inline:
            if not any(queued.cache_key == cache_key for queued in _AUDIT_MAINTENANCE_QUEUE):
                _AUDIT_MAINTENANCE_QUEUE.append(
                    _QueuedMaintenance(audit_log_path, config.audit_retention_days, config.fsync_on_append)
                )
                logger.debug("security_maintenance_enqueued", audit_log_path=audit_log_path)
            return
        _compact_and_mark(audit_log_path, config.audit_retention_days, config.fsync_on_append, cache_key)


def _compact_and_mark(audit_log_path: str, retention_days: int, fsync: bool, cache_key: str) -> None:
    """Compact one audit log and mark *cache_key* processed.

    Caller holds ``_AUDIT_MAINTENANCE_LOCK`` (both callers do), so the
    bounded-set eviction below is race-free.
    """
    AuditLog(Path(audit_log_path), fsync=fsync).compact(retention_days)
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


def drain_security_maintenance() -> dict[str, object]:
    """Compact every audit log queued while ``security_maintenance_inline`` was ``False`` (B71-97).

    Nothing else drains this queue: with the switch off, ``ensure_security_maintenance`` only
    enqueues, so an operator running that way accumulated queued keys that fell off the
    bounded deque (maxlen 128) and were never compacted once the queue filled. Call this from a
    maintenance sweep (``memory_maintain``); it is idempotent and safe to call with an empty
    queue.

    Returns:
        ``{"drained": int}`` -- the number of distinct audit logs compacted this call.
    """
    with _AUDIT_MAINTENANCE_LOCK:
        pending = list(_AUDIT_MAINTENANCE_QUEUE)
        _AUDIT_MAINTENANCE_QUEUE.clear()
    for item in pending:
        with _AUDIT_MAINTENANCE_LOCK:
            _compact_and_mark(item.audit_log_path, item.retention_days, item.fsync, item.cache_key)
    return {"drained": len(pending)}


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
    effective_namespace = namespace if namespace is not None else DEFAULT_NAMESPACE
    entry = active_backend.get(learning_id, namespace=effective_namespace)
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
            "status_history": get_status_history(config, learning_id, namespace=effective_namespace),
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
        "status_history": get_status_history(config, learning_id, namespace=effective_namespace),
    }


# Canary FR-007 helpers extracted to _runtime_canary.py (PRD-DIST-245 batch 101).
from trw_memory.security._runtime_canary import (
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
    _resolve_security_trace_context as _resolve_security_trace_context,
)
