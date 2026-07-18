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
  backends sharing one quarantine DB each get their canaries seeded
  and probed independently — a concurrent multi-backend audit
  surfaced ``CanaryTamperError`` on every recall when only the
  quarantine path keyed the state.

`_resolve_security_trace_context` is looked up lazily from the
parent ``runtime`` module to break the import cycle.
"""

from __future__ import annotations

import hashlib
from typing import Any

from trw_memory.exceptions import CanaryTamperError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.canary import _CANARY_FIXTURES, PINNED_HASHES
from trw_memory.security.startup import resolve_security_path
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event
from trw_memory.storage.interface import StorageBackend

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


def _store_pinned_canary(
    backend: StorageBackend,
    *,
    canary_id: str,
    content: str,
    expected_hash: str,
) -> None:
    """Store one trusted canary with its security metadata invariant."""
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
        _store_pinned_canary(backend, canary_id=canary_id, content=content, expected_hash=expected_hash)
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
    fixture_map = dict(_CANARY_FIXTURES)
    for canary_id, expected_hash in list(PINNED_HASHES.items())[: config.canary_injection_rate]:
        entry = backend.get(canary_id)
        if entry is None:
            # PRD-FIX-102 (FR-1/FR-3): a MISSING canary is self-healed from the trusted,
            # hash-pinned in-process fixture rather than halting ALL recall. A missing canary
            # is a lost-detector (DB recovery/salvage, or a stale process-wide ``seeded`` flag
            # that defeated initialize_canaries' idempotent re-seed) — NOT content tampering.
            # The fixture content is pinned (PINNED_HASHES), so an attacker cannot inject
            # content via this path; the recovery is audit-logged via ``canary_reseeded``.
            # Drift (content present but tampered) below remains the genuine poisoning signal.
            content = fixture_map.get(canary_id)
            telemetry_session_id, telemetry_run_id = _trace_context(session_id="canary-probe")
            if content is None:
                # No fixture to restore from — fall back to the tamper signal.
                state["failed"] = True
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
            _store_pinned_canary(backend, canary_id=canary_id, content=content, expected_hash=expected_hash)
            emit_security_event(
                config,
                emitter="canary",
                session_id=telemetry_session_id,
                run_id=telemetry_run_id,
                payload={
                    "event_name": "canary_reseeded",
                    "canary_id": canary_id,
                    "fail_mode": config.canary_fail_mode,
                    "traceability": build_security_traceability(
                        live_path="security.runtime.probe_canaries",
                        requirement_ids=["FR-007", "NFR-010", "NFR-011"],
                    ),
                },
            )
            continue
        current_hash = hashlib.sha256(entry.content.encode("utf-8")).hexdigest()
        if current_hash != expected_hash:
            # PRD-FIX-102 (FR-2/FR-4): DRIFT is the genuine content-tamper signal. Always
            # quarantine + emit, but RAISE only when ``canary_fail_mode == 'halt'`` (the
            # default) — ``degrade``/``log-only`` set ``failed`` + emit without halting recall,
            # making the previously-dead config knob live.
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
            if config.canary_fail_mode == "halt":
                raise CanaryTamperError(f"canary drift detected for {canary_id}")


def _has_canary_drift(config: MemoryConfig, *, backend: StorageBackend) -> bool:
    """True iff any active canary is PRESENT but content-tampered (hash != pin) — a genuine
    poisoning signal. A MISSING canary is NOT drift: it is recoverable (probe self-heals it
    from the trusted pin, PRD-FIX-102). Read-only; does not mutate state or re-seed.
    """
    for canary_id, expected_hash in list(PINNED_HASHES.items())[: config.canary_injection_rate]:
        entry = backend.get(canary_id)
        if entry is None:
            continue  # missing => recoverable, not a tamper
        if hashlib.sha256(entry.content.encode("utf-8")).hexdigest() != expected_hash:
            return True
    return False


def should_halt_recalls(config: MemoryConfig, *, backend: StorageBackend) -> bool:
    state_key = _state_key(config, backend)
    state = CANARY_STATE.get(state_key)
    if not (state and state.get("failed") and config.canary_fail_mode == "halt"):
        return False
    # PRD-FIX-102 resilience completion (meta-harness C008): a sticky ``failed`` flag must not
    # permanently halt recall AFTER the tamper condition has cleared (e.g. canaries lost to a DB
    # salvage then self-healed/re-seeded). The flag is checked here, BEFORE probe_canaries runs,
    # so a stuck process would otherwise never reach the probe's self-heal. Re-verify: only a
    # CONFIRMED DRIFT (present + hash-mismatch) is a genuine persistent tamper that keeps halting.
    # Missing/recovered canaries un-stick the flag so recall resumes (the probe then self-heals
    # any still-missing canary from the trusted pin). Drift detection is unchanged.
    if _has_canary_drift(config, backend=backend):
        return True
    state["failed"] = False
    telemetry_session_id, telemetry_run_id = _trace_context(session_id="canary-recovered")
    emit_security_event(
        config,
        emitter="canary",
        session_id=telemetry_session_id,
        run_id=telemetry_run_id,
        payload={
            "event_name": "canary_recovered",
            "fail_mode": config.canary_fail_mode,
            "traceability": build_security_traceability(
                live_path="security.runtime.should_halt_recalls",
                requirement_ids=["FR-007", "NFR-010", "NFR-011"],
            ),
        },
    )
    return False
