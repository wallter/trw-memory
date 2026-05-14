"""Canary seeding + probing helpers for the runtime path.

Belongs to ``security/runtime.py``. Re-exported there for back-compat.

3 helpers + 1 module-level state dict covering FR-007 canary
mechanism:

- ``initialize_canaries`` — seed N canary learnings (deterministic
  content + pinned content-hash metadata) idempotently per
  quarantine-DB path; emit ``canary_seeded`` security event.
- ``probe_canaries`` — re-read each canary, compare content hash
  against pin; emit ``canary_missing`` or ``canary_hash_drift`` and
  raise ``CanaryTamperError`` on tamper.
- ``should_halt_recalls`` — return True when a canary failure was
  observed AND ``canary_fail_mode == "halt"``.

State:

- ``CANARY_STATE`` — process-wide dict keyed by ``(quarantine-DB
  path, backend identity)`` containing ``{seeded, recall_count,
  failed}``. Survives across module reloads only within a single
  process. The composite key is required so multiple memory
  backends sharing one quarantine DB (the trw-distill lab pattern)
  each get their canaries seeded and probed independently — cycle
  121 surfaced ``CanaryTamperError`` on every recall when only the
  quarantine path keyed the state.

`_resolve_security_trace_context` is looked up lazily from the
parent ``runtime`` module to break the import cycle.

Extracted as PRD-DIST-245 Phase 3 batch 101.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog

from trw_memory.exceptions import CanaryTamperError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.canary import _CANARY_FIXTURES, PINNED_HASHES
from trw_memory.security.startup import resolve_security_path
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

CANARY_STATE: dict[str, dict[str, object]] = {}


def _trace_context(*, session_id: str | None = None) -> tuple[str, str | None]:
    from trw_memory.security import runtime as _runtime

    result: tuple[str, str | None] = _runtime._resolve_security_trace_context(session_id=session_id)
    return result


def _backend_identity(backend: StorageBackend) -> str:
    """Stable identity string for the backend's data store."""
    db_path = getattr(backend, "_db_path", None)
    if db_path is not None:
        return str(db_path)
    dir_path = getattr(backend, "_dir", None)
    if dir_path is not None:
        return str(dir_path)
    return repr(backend)


def _state_key(config: MemoryConfig, backend: StorageBackend) -> str:
    quarantine_path = str(resolve_security_path(config, "quarantine_db_path", create_parent=True))
    return f"{quarantine_path}::{_backend_identity(backend)}"


def initialize_canaries(config: MemoryConfig, *, backend: StorageBackend) -> None:
    state_key = _state_key(config, backend)
    if CANARY_STATE.get(state_key, {}).get("seeded"):
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
    CANARY_STATE[state_key] = {"seeded": True, "recall_count": 0, "failed": False}
    telemetry_session_id, telemetry_run_id = _trace_context(session_id="canary-bootstrap")
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
    state_key = _state_key(config, backend)
    state: dict[str, Any] = CANARY_STATE.setdefault(state_key, {"seeded": False, "recall_count": 0, "failed": False})
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
            telemetry_session_id, telemetry_run_id = _trace_context(session_id="canary-probe")
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
            telemetry_session_id, telemetry_run_id = _trace_context(session_id="canary-probe")
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


def should_halt_recalls(config: MemoryConfig, *, backend: StorageBackend) -> bool:
    state_key = _state_key(config, backend)
    return bool(CANARY_STATE.get(state_key, {}).get("failed")) and config.canary_fail_mode == "halt"
