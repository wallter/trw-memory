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

import threading
from collections.abc import Mapping
from typing import Any, Literal

from trw_memory.exceptions import CanaryTamperError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.validation import DEFAULT_NAMESPACE
from trw_memory.security.canary import _CANARY_FIXTURES, PINNED_HASHES, _sha
from trw_memory.security.poisoning import RESERVED_SYSTEM_METADATA_KEYS
from trw_memory.security.provenance import INTAKE_METADATA_KEYS
from trw_memory.security.startup import resolve_security_path
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event
from trw_memory.storage._shared import _BOOKKEEPING_FIELDS
from trw_memory.storage.interface import StorageBackend

#: Canaries are seeded into the store's unnamed namespace. Under schema 5 a row
#: is identified by ``(namespace, id)`` (PRD-CORE-245 FR01), so the seed, the
#: probe and the drift check must all name the SAME namespace or the probe
#: reads "missing" for a canary that is present and self-heals it forever.
CANARY_NAMESPACE = DEFAULT_NAMESPACE


CANARY_STATE: dict[str, dict[str, object]] = {}

# PRD-CORE-279 FR04: recall now runs in a worker pool, so seeding and probing
# can be entered concurrently. This is a security control -- an unlocked
# check-then-seed lets two callers both initialise, and a late initialiser that
# REPLACES the state dict erases a tamper failure another caller just recorded.
# Reentrant because probe_canaries calls initialize_canaries.
_CANARY_STATE_LOCK = threading.RLock()


def _trace_context(*, session_id: str | None = None) -> tuple[str, str | None]:
    from trw_memory.security import runtime as _runtime

    result: tuple[str, str | None] = _runtime._resolve_security_trace_context(session_id=session_id)
    return result


def _emit_canary_event(
    config: MemoryConfig,
    trace_session: str,
    event_name: str,
    live_path: str,
    requirement_ids: tuple[str, ...] = ("FR-007", "NFR-010", "NFR-011"),
    **fields: object,
) -> None:
    """Emit one ``canary`` security event: *fields* between its name and its traceability block."""
    session_id, run_id = _trace_context(session_id=trace_session)
    emit_security_event(
        config,
        emitter="canary",
        session_id=session_id,
        run_id=run_id,
        payload={
            "event_name": event_name,
            **fields,
            "traceability": build_security_traceability(
                live_path=f"security.runtime.{live_path}", requirement_ids=list(requirement_ids)
            ),
        },
    )


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


#: What may differ from the seeded row and still be the store's canary: the store's own bookkeeping
#: (a recalled canary's counters), the namespace the row sits in, the clock it was seeded at, and
#: (in :func:`_compared`) the system metadata keys: the reserved ones 4.0.0's import intake stripped and
#: the ones every intake writes (a canary that went through an import carries them).
_SYSTEM_METADATA = INTAKE_METADATA_KEYS.union(RESERVED_SYSTEM_METADATA_KEYS)
_NOT_COMPARED = {*_BOOKKEEPING_FIELDS, "namespace", "created_at", "updated_at", "valid_from"}


def _seeded_canary(canary_id: str) -> MemoryEntry:
    """The pinned canary *canary_id* exactly as :func:`_store_pinned_canary` writes it."""
    metadata = {"system_canary": "true", "provenance_content_hash": PINNED_HASHES[canary_id]}
    content = dict(_CANARY_FIXTURES)[canary_id]
    return MemoryEntry(id=canary_id, content=content, namespace=CANARY_NAMESPACE, metadata=metadata)


def classify_canary(row: Mapping[str, object] | MemoryEntry) -> Literal["canary", "user-data"] | None:
    """The one predicate every import path skips a source store's system canary by (PRD-CORE-309).

    ``None``: not a pinned identity (the id and its content's hash), so an ordinary row, whatever its
    ``system_canary`` flag says. ``"canary"``: every other field is what :func:`_seeded_canary` writes,
    apart from ``_NOT_COMPARED``; skip it, the destination seeds its own. ``"user-data"``: a pinned
    identity carrying anything else (fail-closed), which the caller rejects rather than drop.
    """
    entry_id, content = (row.get("id"), row.get("content")) if isinstance(row, Mapping) else (row.id, row.content)
    expected = PINNED_HASHES.get(str(entry_id))
    if expected is None or expected != _sha(str(content or "")):
        return None
    if isinstance(row, Mapping) and not set(row) <= set(MemoryEntry.model_fields):
        return "user-data"
    try:
        entry = row if isinstance(row, MemoryEntry) else MemoryEntry.model_validate(row)
    except (TypeError, ValueError):  # a row MemoryEntry cannot hold is not the seeded one (sol r2 P2)
        return "user-data"
    return "canary" if _compared(entry) == _compared(_seeded_canary(entry.id)) else "user-data"


def _compared(entry: MemoryEntry) -> dict[str, Any]:
    shape = entry.model_dump(exclude=_NOT_COMPARED)
    shape["metadata"] = {k: v for k, v in entry.metadata.items() if k not in _SYSTEM_METADATA}
    return shape


def _store_pinned_canary(backend: StorageBackend, canary_id: str) -> None:
    """Store one trusted canary with its security metadata invariant."""
    backend.store(_seeded_canary(canary_id))


def initialize_canaries(config: MemoryConfig, *, backend: StorageBackend) -> None:
    with _CANARY_STATE_LOCK:
        _initialize_canaries_locked(config, backend=backend)


def _initialize_canaries_locked(config: MemoryConfig, *, backend: StorageBackend) -> None:
    state_key = _state_key(config, backend)
    if CANARY_STATE.get(state_key, {}).get("seeded"):
        return
    seeded = 0
    for canary_id in list(PINNED_HASHES)[: config.canary_injection_rate]:
        if backend.get(canary_id, namespace=CANARY_NAMESPACE) is not None:
            seeded += 1
            continue
        _store_pinned_canary(backend, canary_id)
        seeded += 1
    # Update in place rather than replacing: a concurrent probe may already have
    # recorded a tamper failure against this key, and a fresh dict would drop it.
    state = CANARY_STATE.setdefault(state_key, {"seeded": False, "recall_count": 0, "failed": False})
    state["seeded"] = True
    _emit_canary_event(
        config,
        "canary-bootstrap",
        "canary_seeded",
        "initialize_canaries",
        seeded_count=seeded,
        canary_injection_rate=config.canary_injection_rate,
    )


def probe_canaries(config: MemoryConfig, *, backend: StorageBackend) -> None:
    with _CANARY_STATE_LOCK:
        _probe_canaries_locked(config, backend=backend)


def _probe_canaries_locked(config: MemoryConfig, *, backend: StorageBackend) -> None:
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
        entry = backend.get(canary_id, namespace=CANARY_NAMESPACE)
        if entry is None:
            # PRD-FIX-102 (FR-1/FR-3): a MISSING canary is self-healed from the trusted,
            # hash-pinned in-process fixture rather than halting ALL recall. A missing canary
            # is a lost-detector (DB recovery/salvage, or a stale process-wide ``seeded`` flag
            # that defeated initialize_canaries' idempotent re-seed) — NOT content tampering.
            # The fixture content is pinned (PINNED_HASHES), so an attacker cannot inject
            # content via this path; the recovery is audit-logged via ``canary_reseeded``.
            # Drift (content present but tampered) below remains the genuine poisoning signal.
            content = fixture_map.get(canary_id)
            if content is None:
                # No fixture to restore from — fall back to the tamper signal.
                state["failed"] = True
                _emit_canary_event(
                    config,
                    "canary-probe",
                    "canary_missing",
                    "probe_canaries",
                    canary_id=canary_id,
                    fail_mode=config.canary_fail_mode,
                )
                raise CanaryTamperError(f"missing canary {canary_id}")
            _store_pinned_canary(backend, canary_id)
            _emit_canary_event(
                config,
                "canary-probe",
                "canary_reseeded",
                "probe_canaries",
                canary_id=canary_id,
                fail_mode=config.canary_fail_mode,
            )
            continue
        current_hash = _sha(entry.content)
        if current_hash != expected_hash:
            # PRD-FIX-102 (FR-2/FR-4): DRIFT is the genuine content-tamper signal. Always
            # quarantine + emit, but RAISE only when ``canary_fail_mode == 'halt'`` (the
            # default) — ``degrade``/``log-only`` set ``failed`` + emit without halting recall,
            # making the previously-dead config knob live.
            state["failed"] = True
            entry.metadata["quarantined"] = "true"
            backend.store(entry)
            _emit_canary_event(
                config,
                "canary-probe",
                "canary_hash_drift",
                "probe_canaries",
                ("FR-007", "FR-009", "NFR-010", "NFR-011"),
                canary_id=canary_id,
                expected_hash=expected_hash,
                observed_hash=current_hash,
                fail_mode=config.canary_fail_mode,
            )
            if config.canary_fail_mode == "halt":
                raise CanaryTamperError(f"canary drift detected for {canary_id}")


def _has_canary_drift(config: MemoryConfig, *, backend: StorageBackend) -> bool:
    """True iff any active canary is PRESENT but content-tampered (hash != pin) — a genuine
    poisoning signal. A MISSING canary is NOT drift: it is recoverable (probe self-heals it
    from the trusted pin, PRD-FIX-102). Read-only; does not mutate state or re-seed.
    """
    for canary_id, expected_hash in list(PINNED_HASHES.items())[: config.canary_injection_rate]:
        entry = backend.get(canary_id, namespace=CANARY_NAMESPACE)
        if entry is None:
            continue  # missing => recoverable, not a tamper
        if _sha(entry.content) != expected_hash:
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
    _emit_canary_event(
        config, "canary-recovered", "canary_recovered", "should_halt_recalls", fail_mode=config.canary_fail_mode
    )
    return False
