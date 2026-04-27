"""SEC-001 security-event stream emission helpers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from trw_memory.exceptions import SecurityTelemetryUnavailableError
from trw_memory.models.config import MemoryConfig
from trw_memory.storage.persistence import append_jsonl, read_yaml

logger = structlog.get_logger(__name__)
_RUN_SURFACE_SNAPSHOT_FILENAME = "run_surface_snapshot.yaml"
_SURFACE_SNAPSHOT_ENV = "TRW_SURFACE_SNAPSHOT_ID"


def _event_id() -> str:
    return f"evt_{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _events_filename(now: datetime | None = None) -> str:
    ts = now or _utc_now()
    return f"events-{ts.strftime('%Y-%m-%d')}.jsonl"


class MemorySecurityEvent(BaseModel):
    """H1-compatible security-event envelope for trw-memory live protection paths."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event_id: str = Field(default_factory=_event_id)
    session_id: str
    run_id: str | None = None
    ts: datetime = Field(default_factory=_utc_now)
    emitter: str
    event_type: str = "memory_security"
    surface_snapshot_id: str = ""
    parent_event_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def resolve_security_events_path(config: MemoryConfig, *, now: datetime | None = None) -> Path:
    return Path(config.audit_log_path).expanduser().resolve().parent / _events_filename(now)


def _discover_trw_dirs(config: MemoryConfig) -> tuple[Path, ...]:
    candidates: list[Path] = []
    env_trw_dir = os.environ.get("TRW_DIR", "").strip()
    if env_trw_dir:
        candidates.append(Path(env_trw_dir).expanduser().resolve())

    audit_path = Path(config.audit_log_path).expanduser().resolve()
    audit_trw_dir = next((parent for parent in audit_path.parents if parent.name == ".trw"), None)
    if audit_trw_dir is not None:
        candidates.append(audit_trw_dir)

    current = Path.cwd().resolve()
    cwd_trw_dir = next(
        (
            (candidate / ".trw").resolve()
            for candidate in (current, *current.parents)
            if (candidate / ".trw").exists()
        ),
        None,
    )
    if cwd_trw_dir is not None:
        candidates.append(cwd_trw_dir)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return tuple(unique)


def _load_surface_snapshot_id_from_manifest(config: MemoryConfig, *, run_id: str | None) -> str:
    if not run_id:
        return ""
    for trw_dir in _discover_trw_dirs(config):
        runs_root = trw_dir / "runs"
        if not runs_root.exists():
            continue
        pattern = f"*/{run_id}/meta/{_RUN_SURFACE_SNAPSHOT_FILENAME}"
        for manifest_path in sorted(runs_root.glob(pattern)):
            try:
                manifest = read_yaml(manifest_path)
            except Exception:
                logger.warning(
                    "memory_security_surface_snapshot_unreadable",
                    component="memory_security",
                    op="resolve_surface_snapshot_id",
                    outcome="failed",
                    run_id=run_id,
                    path=str(manifest_path),
                    exc_info=True,
                )
                continue
            snapshot_id = manifest.get("snapshot_id")
            if isinstance(snapshot_id, str) and snapshot_id.strip():
                return snapshot_id.strip()
    return ""


def resolve_surface_snapshot_id(
    config: MemoryConfig,
    *,
    run_id: str | None = None,
    explicit_snapshot_id: str | None = None,
) -> str:
    if explicit_snapshot_id is not None and explicit_snapshot_id.strip():
        return explicit_snapshot_id.strip()
    env_snapshot_id = os.environ.get(_SURFACE_SNAPSHOT_ENV, "").strip()
    if env_snapshot_id:
        return env_snapshot_id
    resolved_run_id = run_id or os.environ.get("TRW_RUN_ID", "").strip() or None
    return _load_surface_snapshot_id_from_manifest(config, run_id=resolved_run_id)


def build_security_traceability(
    *,
    live_path: str,
    requirement_ids: list[str],
    dependency_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "prd_id": "PRD-SEC-001",
        "requirement_ids": list(requirement_ids),
        "dependency_ids": list(dependency_ids or ["PRD-HPO-MEAS-001"]),
        "live_path": live_path,
    }


def emit_security_event(
    config: MemoryConfig,
    *,
    emitter: str,
    session_id: str,
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
    parent_event_id: str | None = None,
    surface_snapshot_id: str | None = None,
    required: bool = True,
) -> bool:
    event = MemorySecurityEvent(
        session_id=session_id or "memory-security",
        run_id=run_id,
        emitter=emitter,
        surface_snapshot_id=resolve_surface_snapshot_id(
            config,
            run_id=run_id,
            explicit_snapshot_id=surface_snapshot_id,
        ),
        parent_event_id=parent_event_id,
        payload=payload or {},
    )
    path = resolve_security_events_path(config)
    try:
        append_jsonl(path, event.model_dump(mode="json"))
    except Exception as exc:
        logger.exception(
            "memory_security_event_write_failed",
            component="memory_security",
            op="emit_security_event",
            outcome="failed",
            emitter=emitter,
            session_id=event.session_id,
            run_id=event.run_id,
            surface_snapshot_id=event.surface_snapshot_id,
            activation_gate_blocked_reason="security_telemetry_unavailable",
            path=str(path),
        )
        if required:
            raise SecurityTelemetryUnavailableError(
                f"security telemetry unavailable: unable to append event for {emitter}"
            ) from exc
        return False
    return True


__all__ = [
    "MemorySecurityEvent",
    "build_security_traceability",
    "emit_security_event",
    "resolve_security_events_path",
    "resolve_surface_snapshot_id",
]
