"""Shared runtime security helpers for store/search/forget paths."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import time

import structlog

from trw_memory.exceptions import PIIBlockError, RateLimitError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.audit import AuditLog
from trw_memory.security.pii import PIIMatch, PIIType, detect_pii
from trw_memory.security.poisoning import quarantine_entry, score_entry_anomaly, validate_entry_payload
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import lock_for_rmw, read_yaml, write_yaml
from trw_memory.storage.yaml_backend import YAMLBackend

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
) -> PreparedStoreEntry:
    """Apply rate limits, PII handling, and anomaly scoring before a write."""
    ensure_security_maintenance(config)
    actor = _actor_for_entry(entry)
    existing = backend.get(entry.id)
    op = "update" if existing is not None and existing.namespace == entry.namespace else "store"
    flagged_entry = _flag_code_snippet(entry)

    try:
        enforce_write_rate_limit(config, session_id=session_id, actor=actor, namespace=entry.namespace, entry_id=entry.id)
        validate_entry_payload(flagged_entry, max_chars=config.max_entry_chars)
        secured_entry, pii_matches = _apply_runtime_pii_policy(flagged_entry, config)
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
    """Persist a quarantined entry outside the main memory store."""
    namespace_dir = _quarantine_namespace_dir(config, entry.namespace)
    namespace_dir.mkdir(parents=True, exist_ok=True)
    (namespace_dir / _NAMESPACE_METADATA_FILE).write_text(entry.namespace, encoding="utf-8")
    with YAMLBackend(namespace_dir / "entries") as backend:
        backend.store(entry)


def list_quarantined_entries(
    config: MemoryConfig,
    *,
    namespace: str | None = None,
    actor: str | None = None,
    limit: int = 100,
) -> list[MemoryEntry]:
    """Return quarantined entries filtered by namespace and actor."""
    ensure_security_maintenance(config)
    root = Path(config.quarantine_path)
    if not root.exists():
        return []

    entries: list[MemoryEntry] = []
    for namespace_dir in sorted(root.iterdir()):
        if not namespace_dir.is_dir():
            continue
        actual_namespace = _read_namespace_metadata(namespace_dir)
        if actual_namespace is None:
            continue
        if namespace is not None and actual_namespace != namespace:
            continue
        entries_dir = namespace_dir / "entries"
        if not entries_dir.exists():
            continue
        with YAMLBackend(entries_dir) as backend:
            for entry in backend.list_entries(namespace=actual_namespace, limit=limit * 5):
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
    namespace_dir = _quarantine_namespace_dir(config, namespace)
    entries_dir = namespace_dir / "entries"
    if not entries_dir.exists():
        return 0
    deleted = 0
    with YAMLBackend(entries_dir) as backend:
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
        logger.warning("memory_store_pii_blocked", detected_type=detected_type, namespace=entry.namespace, entry_id=entry.id)
        raise PIIBlockError(f"memory entry blocked by PII policy: {detected_type}", detected_type=detected_type)

    new_content = _replace_pii(entry.content, content_matches)
    new_detail = _replace_pii(entry.detail, detail_matches)
    metadata = dict(entry.metadata)
    metadata["pii_types"] = ",".join(sorted({match.pii_type for match in all_matches}))
    if any(match.pii_type == PIIType.HIGH_ENTROPY for match in all_matches):
        metadata["contains_high_entropy_token"] = "true"
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
    combined = f"{entry.content}\n{entry.detail}"
    if any(pattern.search(combined) for pattern in _CODE_SNIPPET_PATTERNS):
        if "code_snippet_flagged" not in entry.tags:
            return entry.model_copy(update={"tags": [*entry.tags, "code_snippet_flagged"]})
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
    clean_reference = [candidate for candidate in reference_entries if candidate.metadata.get("quarantined") != "true"]
    clean_reference.sort(key=lambda candidate: candidate.updated_at, reverse=True)
    rolling = clean_reference[:100]
    stats = _build_anomaly_stats(rolling)
    anomaly = score_entry_anomaly(entry, rolling, z_threshold=config.poisoning_z_threshold)
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
