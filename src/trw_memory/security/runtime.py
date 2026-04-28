"""Shared runtime security helpers for store/search/forget paths."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import time

import structlog

from trw_memory.exceptions import (
    CanaryTamperError,
    PIIBlockError,
    ProvenanceKeyUnavailableError,
    QuarantineUnreachableError,
    RateLimitError,
    ScorerUnavailableError,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.audit import AuditLog
from trw_memory.security.canary import _CANARY_FIXTURES, PINNED_HASHES
from trw_memory.security.pii import PIIMatch, PIIType, detect_pii
from trw_memory.security.poisoning import quarantine_entry, score_entry_anomaly, validate_entry_payload
from trw_memory.security.provenance import build_entry_provenance, derive_verify_key, verify_entry_provenance
from trw_memory.security.startup import _discover_anchor, resolve_security_path, verify_defaults
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event
from trw_memory.security.trust_scorer import score_intake
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import lock_for_rmw, read_yaml, write_yaml
from trw_memory.storage.sqlite_backend import SQLiteBackend

_NAMESPACE_METADATA_FILE = "namespace.txt"
_BLOCKING_PII_TYPES = frozenset({PIIType.API_KEY})
_REDACTED_PII_TYPES = frozenset(
    {
        PIIType.EMAIL,
        PIIType.IP_ADDRESS,
        PIIType.CUSTOM,
    }
)
_CODE_SNIPPET_PATTERNS = (
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+"),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfunction\s+\w+\s*\("),
)
_AUDIT_MAINTENANCE_CACHE: set[str] = set()
logger = structlog.get_logger(__name__)
_CANARY_STATE: dict[str, dict[str, object]] = {}


@dataclass(frozen=True)
class AnomalyStats:
    """Rolling anomaly statistics persisted alongside the quarantine store."""

    sample_count: int
    dimensions: dict[str, dict[str, float]]


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


def store_quarantined_entry(config: MemoryConfig, entry: MemoryEntry) -> None:
    """Persist a quarantined entry in the SEC-001 quarantine SQLite store."""
    try:
        with _open_quarantine_backend(config) as backend:
            backend.store(
                entry.model_copy(
                    update={
                        "metadata": {
                            **entry.metadata,
                            "quarantined": "true",
                            "quarantined_at": datetime.now(timezone.utc).isoformat(),
                        }
                    }
                )
            )
            _append_review_log(config, entry.id, "quarantined", reviewer_id="system")
    except OSError as exc:
        raise QuarantineUnreachableError(f"quarantine DB unavailable: {exc}") from exc


def list_quarantined_entries(
    config: MemoryConfig,
    *,
    namespace: str | None = None,
    actor: str | None = None,
    limit: int = 100,
) -> list[MemoryEntry]:
    """Return quarantined entries filtered by namespace and actor."""
    ensure_security_maintenance(config)
    entries: list[MemoryEntry] = []
    with _open_quarantine_backend(config) as backend:
        for entry in backend.list_entries(namespace=namespace, limit=limit * 5):
            if entry.metadata.get("quarantined") != "true":
                continue
            if actor is not None and entry.source_identity != actor:
                continue
            entries.append(entry)
    entries.sort(key=lambda item: item.updated_at, reverse=True)
    return entries[:limit]


def delete_quarantined_entries(
    config: MemoryConfig,
    *,
    namespace: str,
    actor: str | None = None,
    memory_id: str | None = None,
) -> int:
    """Delete matching quarantined entries and return the count removed."""
    ensure_security_maintenance(config)
    deleted = 0
    with _open_quarantine_backend(config) as backend:
        if memory_id is not None:
            return 1 if backend.delete(memory_id) else 0
        for entry in backend.list_entries(namespace=namespace, limit=10_000):
            if actor is not None and entry.source_identity != actor:
                continue
            if entry.metadata.get("quarantined") != "true":
                continue
            if backend.delete(entry.id):
                deleted += 1
    return deleted


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


def _apply_runtime_pii_policy(entry: MemoryEntry, config: MemoryConfig) -> tuple[MemoryEntry, list[PIIMatch]]:
    if not config.pii_enabled:
        return entry, []

    content_matches = detect_pii(
        entry.content,
        entropy_threshold=config.pii_entropy_threshold,
        custom_patterns=config.pii_custom_patterns,
    )
    detail_matches = detect_pii(
        entry.detail,
        entropy_threshold=config.pii_entropy_threshold,
        custom_patterns=config.pii_custom_patterns,
    )
    all_matches = content_matches + detail_matches
    if not all_matches:
        return entry, []

    blocking = [match for match in all_matches if match.pii_type in _BLOCKING_PII_TYPES]
    if blocking:
        detected_type = str(blocking[0].pii_type)
        logger.warning(
            "memory_store_pii_blocked", detected_type=detected_type, namespace=entry.namespace, entry_id=entry.id
        )
        raise PIIBlockError(f"memory entry blocked by PII policy: {detected_type}", detected_type=detected_type)

    new_content = _replace_pii(entry.content, content_matches)
    new_detail = _replace_pii(entry.detail, detail_matches)
    metadata = dict(entry.metadata)
    metadata["pii_types"] = ",".join(sorted({match.pii_type for match in all_matches}))
    if any(match.pii_type == PIIType.HIGH_ENTROPY for match in all_matches):
        metadata["contains_high_entropy_token"] = "true"  # noqa: S105 — flag value, not a credential
    return entry.model_copy(update={"content": new_content, "detail": new_detail, "metadata": metadata}), all_matches


def _replace_pii(text: str, matches: list[PIIMatch]) -> str:
    if not matches:
        return text
    result = text
    for match in sorted(matches, key=lambda item: item.start, reverse=True):
        if match.pii_type == PIIType.FILE_PATH:
            replacement = _hash_path_components(match.value)
        elif match.pii_type in _REDACTED_PII_TYPES:
            replacement = _redaction_marker(match.pii_type)
        else:
            continue
        result = result[: match.start] + replacement + result[match.end :]
    return result


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _hash_path_components(path_value: str) -> str:
    is_windows = ":\\" in path_value
    separator = "\\" if is_windows else "/"
    components = [component for component in re.split(r"[\\/]+", path_value) if component]
    hashed = [hashlib.sha256(component.encode("utf-8")).hexdigest()[:8] for component in components]
    prefix = "C:\\" if is_windows and path_value[:2].isalpha() else ("/" if path_value.startswith("/") else "")
    return prefix + separator.join(hashed)


def _redaction_marker(pii_type: PIIType) -> str:
    if pii_type == PIIType.EMAIL:
        return "<email>"
    if pii_type == PIIType.IP_ADDRESS:
        return "<ip>"
    if pii_type == PIIType.CUSTOM:
        return "<custom_pii>"
    return f"<{pii_type}>"


def _flag_code_snippet(entry: MemoryEntry) -> MemoryEntry:
    """Authoritatively set the system code-flag metadata key.

    Strips any caller-provided value of ``SYSTEM_CODE_FLAG_KEY`` and sets
    it to "true" iff the combined content matches a code-snippet pattern.
    This guarantees callers cannot pre-seed the bypass flag — see security
    audit 2026-04-18 H2. The descriptive ``"code_snippet_flagged"`` tag is
    still appended for backward-compat visibility in UI listings.
    """
    from trw_memory.security.poisoning import SYSTEM_CODE_FLAG_KEY

    combined = f"{entry.content}\n{entry.detail}"
    is_code = any(pattern.search(combined) for pattern in _CODE_SNIPPET_PATTERNS)

    # Start by stripping any caller-supplied system-flag key.
    metadata = {k: v for k, v in entry.metadata.items() if k != SYSTEM_CODE_FLAG_KEY}
    tags = list(entry.tags)
    updates: dict[str, object] = {}

    if is_code:
        metadata[SYSTEM_CODE_FLAG_KEY] = "true"
        if "code_snippet_flagged" not in tags:
            tags.append("code_snippet_flagged")

    if metadata != entry.metadata:
        updates["metadata"] = metadata
    if tags != list(entry.tags):
        updates["tags"] = tags

    if updates:
        return entry.model_copy(update=updates)
    return entry


def _actor_for_entry(entry: MemoryEntry) -> str:
    return entry.source_identity or entry.source or "system"


def _quarantine_namespace_dir(config: MemoryConfig, namespace: str) -> Path:
    return Path(config.quarantine_path) / namespace.replace(":", "_")


def _read_namespace_metadata(namespace_dir: Path) -> str | None:
    metadata_path = namespace_dir / _NAMESPACE_METADATA_FILE
    if not metadata_path.exists():
        return None
    namespace = metadata_path.read_text(encoding="utf-8").strip()
    return namespace or None


def _score_entry_anomaly(
    entry: MemoryEntry,
    backend: StorageBackend,
    *,
    config: MemoryConfig,
) -> tuple[tuple[str, float] | None, AnomalyStats]:
    reference_entries = backend.list_entries(namespace=entry.namespace, limit=1_000)
    clean_reference = [
        candidate
        for candidate in reference_entries
        if candidate.metadata.get("quarantined") != "true" and candidate.metadata.get("system_canary") != "true"
    ]
    clean_reference.sort(key=lambda candidate: candidate.updated_at, reverse=True)
    rolling = clean_reference[:100]
    stats = _build_anomaly_stats(rolling)
    anomaly_reference = [candidate for candidate in rolling if (candidate.content + candidate.detail).strip()]
    anomaly = score_entry_anomaly(entry, anomaly_reference, z_threshold=config.poisoning_z_threshold)
    return anomaly, stats


def _build_anomaly_stats(entries: list[MemoryEntry]) -> AnomalyStats:
    if not entries:
        return AnomalyStats(sample_count=0, dimensions={})
    lengths = [float(len((entry.content + entry.detail).encode("utf-8"))) for entry in entries]
    tag_counts = [float(len(entry.tags)) for entry in entries]
    importances = [float(entry.importance) for entry in entries]
    return AnomalyStats(
        sample_count=len(entries),
        dimensions={
            "entry_length": _series_stats(lengths),
            "tag_count": _series_stats(tag_counts),
            "importance": _series_stats(importances),
        },
    )


def _series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std_dev": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std_dev": math.sqrt(variance)}


def _write_anomaly_stats(config: MemoryConfig, stats: AnomalyStats) -> None:
    stats_path = Path(config.quarantine_path).parent / "anomaly_stats.yaml"
    payload: dict[str, object] = {
        "version": "1.0",
        "updated": datetime.now(timezone.utc).isoformat(),
        "sample_count": stats.sample_count,
        "dimensions": stats.dimensions,
    }
    write_yaml(stats_path, payload)


def ensure_security_maintenance(config: MemoryConfig) -> None:
    """Run once-per-process audit retention maintenance for a config path."""
    cache_key = f"{config.audit_log_path}:{config.audit_retention_days}"
    if cache_key in _AUDIT_MAINTENANCE_CACHE:
        return
    get_audit_log(config).compact(config.audit_retention_days)
    _AUDIT_MAINTENANCE_CACHE.add(cache_key)


def _rejection_reason(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "rate_limited"
    if isinstance(exc, PIIBlockError):
        return "pii_detected"
    if exc.__class__.__name__ == "SchemaValidationError":
        return "schema_invalid"
    return getattr(exc, "reason", exc.__class__.__name__)


def _open_quarantine_backend(config: MemoryConfig) -> SQLiteBackend:
    path = resolve_security_path(config, "quarantine_db_path", create_parent=True)
    return SQLiteBackend(
        db_path=path,
        dim=config.embedding_dim,
        recovery_policy=config.memory_recovery_policy,
        corrupt_backup_keep=config.memory_corrupt_backup_keep,
        rebuild_from_cold=config.memory_recovery_rebuild_from_cold,
    )


def _append_review_log(config: MemoryConfig, learning_id: str, decision: str, *, reviewer_id: str) -> None:
    with _open_quarantine_backend(config) as backend:
        conn = getattr(backend, "_conn", None)
        if conn is None:
            raise QuarantineUnreachableError("quarantine DB connection unavailable")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quarantine_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learning_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                reviewed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO quarantine_reviews (learning_id, decision, reviewer_id, reviewed_at) VALUES (?, ?, ?, ?)",
            (learning_id, decision, reviewer_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_status_history(config: MemoryConfig, learning_id: str) -> list[dict[str, str]]:
    with _open_quarantine_backend(config) as backend:
        conn = getattr(backend, "_conn", None)
        if conn is None:
            return []
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quarantine_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learning_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                reviewed_at TEXT NOT NULL
            )
            """
        )
        rows = conn.execute(
            "SELECT decision, reviewer_id, reviewed_at FROM quarantine_reviews WHERE learning_id = ? ORDER BY id ASC",
            (learning_id,),
        ).fetchall()
    return [
        {"status": str(decision), "reviewer_id": str(reviewer_id), "ts": str(reviewed_at)}
        for decision, reviewer_id, reviewed_at in rows
    ]


def review_quarantined_entry(
    config: MemoryConfig,
    *,
    active_backend: StorageBackend,
    learning_id: str,
    decision: str,
    reviewer_id: str,
) -> dict[str, str]:
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    existing_history = get_status_history(config, learning_id)
    resolved_status = next(
        (item["status"] for item in existing_history if item.get("status") in {"active", "obsolete_poisoned"}),
        "",
    )
    if resolved_status:
        return {"learning_id": learning_id, "status": "already_resolved", "resolved_status": resolved_status}
    with _open_quarantine_backend(config) as quarantine_backend:
        entry = quarantine_backend.get(learning_id)
        if entry is None:
            return {"learning_id": learning_id, "status": "not_found"}
        _append_review_log(
            config, learning_id, "active" if decision == "approve" else "obsolete_poisoned", reviewer_id=reviewer_id
        )
        if decision == "approve":
            approved = entry.model_copy(
                update={
                    "metadata": {
                        **entry.metadata,
                        "quarantined": "false",
                        "reviewed_by": reviewer_id,
                        "review_decision": "approve",
                    }
                }
            )
            active_backend.store(approved)
            quarantine_backend.delete(learning_id)
            return {"learning_id": learning_id, "status": "approved"}
        rejected = entry.model_copy(
            update={
                "metadata": {
                    **entry.metadata,
                    "reviewed_by": reviewer_id,
                    "review_decision": "reject",
                    "security_status": "obsolete_poisoned",
                }
            }
        )
        quarantine_backend.store(rejected)
        return {"learning_id": learning_id, "status": "rejected"}


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

    if config.provenance_required:
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
        entry = entry.model_copy(update={"metadata": {**entry.metadata, **provenance_metadata}})
    return entry


def initialize_canaries(config: MemoryConfig, *, backend: StorageBackend) -> None:
    state_key = str(resolve_security_path(config, "quarantine_db_path", create_parent=True))
    if _CANARY_STATE.get(state_key, {}).get("seeded"):
        return
    seeded = 0
    fixture_map = dict(_CANARY_FIXTURES)
    for canary_id, expected_hash in list(PINNED_HASHES.items())[: config.canary_injection_rate]:
        if backend.get(canary_id) is not None:
            seeded += 1
            continue
        content = fixture_map[canary_id]
        backend.store(
            MemoryEntry(
                id=canary_id,
                content=content,
                namespace="default",
                metadata={
                    "system_canary": "true",
                    "provenance_content_hash": expected_hash,
                },
            )
        )
        seeded += 1
    _CANARY_STATE[state_key] = {"seeded": True, "recall_count": 0, "failed": False}
    telemetry_session_id, telemetry_run_id = _resolve_security_trace_context(session_id="canary-bootstrap")
    emit_security_event(
        config,
        emitter="canary",
        session_id=telemetry_session_id,
        run_id=telemetry_run_id,
        payload={
            "event_name": "canary_seeded",
            "seeded_count": seeded,
            "canary_injection_rate": config.canary_injection_rate,
            "traceability": build_security_traceability(
                live_path="security.runtime.initialize_canaries",
                requirement_ids=["FR-007", "NFR-010", "NFR-011"],
            ),
        },
    )


def probe_canaries(config: MemoryConfig, *, backend: StorageBackend) -> None:
    state_key = str(resolve_security_path(config, "quarantine_db_path", create_parent=True))
    state = _CANARY_STATE.setdefault(state_key, {"seeded": False, "recall_count": 0, "failed": False})
    if not state["seeded"]:
        initialize_canaries(config, backend=backend)
    raw_recall_count = state.get("recall_count", 0)
    recall_count = int(raw_recall_count if isinstance(raw_recall_count, (int, float, str)) else 0)
    recall_count += 1
    state["recall_count"] = recall_count
    if recall_count % config.canary_probe_interval != 0:
        return
    for canary_id, expected_hash in list(PINNED_HASHES.items())[: config.canary_injection_rate]:
        entry = backend.get(canary_id)
        if entry is None:
            state["failed"] = True
            telemetry_session_id, telemetry_run_id = _resolve_security_trace_context(session_id="canary-probe")
            emit_security_event(
                config,
                emitter="canary",
                session_id=telemetry_session_id,
                run_id=telemetry_run_id,
                payload={
                    "event_name": "canary_missing",
                    "canary_id": canary_id,
                    "fail_mode": config.canary_fail_mode,
                    "traceability": build_security_traceability(
                        live_path="security.runtime.probe_canaries",
                        requirement_ids=["FR-007", "NFR-010", "NFR-011"],
                    ),
                },
            )
            raise CanaryTamperError(f"missing canary {canary_id}")
        current_hash = hashlib.sha256(entry.content.encode("utf-8")).hexdigest()
        if current_hash != expected_hash:
            state["failed"] = True
            entry.metadata["quarantined"] = "true"
            backend.store(entry)
            telemetry_session_id, telemetry_run_id = _resolve_security_trace_context(session_id="canary-probe")
            emit_security_event(
                config,
                emitter="canary",
                session_id=telemetry_session_id,
                run_id=telemetry_run_id,
                payload={
                    "event_name": "canary_hash_drift",
                    "canary_id": canary_id,
                    "expected_hash": expected_hash,
                    "observed_hash": current_hash,
                    "fail_mode": config.canary_fail_mode,
                    "traceability": build_security_traceability(
                        live_path="security.runtime.probe_canaries",
                        requirement_ids=["FR-007", "FR-009", "NFR-010", "NFR-011"],
                    ),
                },
            )
            raise CanaryTamperError(f"canary drift detected for {canary_id}")


def should_halt_recalls(config: MemoryConfig) -> bool:
    state_key = str(resolve_security_path(config, "quarantine_db_path", create_parent=True))
    return bool(_CANARY_STATE.get(state_key, {}).get("failed")) and config.canary_fail_mode == "halt"
