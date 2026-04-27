"""FR-004 — Canary learnings with in-memory hash pinning.

Sprint 96 W1-E scaffolding. Seeds known-good canary learnings into the
memory store and later verifies their content against a frozen dict of
``{canary_id: expected_hash}`` loaded at boot time. Mismatches trigger
``CanaryVerificationResult.tampered`` (observe-mode only — no fail-closed
until Phase 3).

The ``memory_store`` parameter is typed as a ``CanaryStore`` Protocol so
this module remains decoupled from :class:`trw_memory.client.MemoryClient`
and is trivially testable with a minimal fake.
"""

from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Protocol

import structlog
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CanaryLearning",
    "CanaryStore",
    "CanaryVerificationResult",
    "seed_canaries",
    "verify_canaries",
]

_LOG = structlog.get_logger(__name__)


# The fixture corpus. In-code constants (NOT DB-loaded) per FR-009.
_CANARY_FIXTURES: tuple[tuple[str, str], ...] = (
    ("canary-001", "TRW canary: architectural truthfulness > velocity."),
    ("canary-002", "TRW canary: verify before recommend."),
    ("canary-003", "TRW canary: no secrets in logs."),
    ("canary-004", "TRW canary: idempotent writes are load-bearing."),
    ("canary-005", "TRW canary: learnings compound across sessions."),
    ("canary-006", "TRW canary: fail-closed beats silent drop."),
    ("canary-007", "TRW canary: session pins isolate clients."),
    ("canary-008", "TRW canary: structlog has no event kwarg."),
    ("canary-009", "TRW canary: hash-pinned integrity check."),
    ("canary-010", "TRW canary: observe mode before enforce mode."),
)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# Frozen at import time (boot) — this is the in-memory pin set.
PINNED_HASHES: MappingProxyType[str, str] = MappingProxyType({cid: _sha(content) for cid, content in _CANARY_FIXTURES})


class CanaryLearning(BaseModel):
    """A canary learning record."""

    model_config = ConfigDict(strict=True)

    canary_id: str
    content: str
    expected_hash: str


class CanaryVerificationResult(BaseModel):
    """Outcome of a canary probe."""

    model_config = ConfigDict(strict=True)

    total: int
    ok: int
    tampered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class CanaryStore(Protocol):
    """Minimal store interface required for seeding / verification.

    :class:`trw_memory.client.MemoryClient` will satisfy this incidentally
    once Sprint 96 Week 2 wires canaries into the write path. For now,
    a dict-backed fake is sufficient for unit testing.
    """

    def seed(self, canary_id: str, content: str) -> None: ...

    def read(self, canary_id: str) -> str | None: ...


def seed_canaries(memory_store: CanaryStore, count: int = 10) -> list[CanaryLearning]:
    """Insert up to *count* canary learnings into *memory_store*.

    Returns the list of canaries seeded. The pinned hash table is NOT
    written to the store — it lives only in process memory at
    :data:`PINNED_HASHES`.
    """
    limit = max(0, min(count, len(_CANARY_FIXTURES)))
    seeded: list[CanaryLearning] = []
    for canary_id, content in _CANARY_FIXTURES[:limit]:
        memory_store.seed(canary_id, content)
        seeded.append(
            CanaryLearning(
                canary_id=canary_id,
                content=content,
                expected_hash=PINNED_HASHES[canary_id],
            )
        )
    _LOG.info("canary.seed", count=len(seeded))
    return seeded


def verify_canaries(memory_store: CanaryStore) -> CanaryVerificationResult:
    """Read every pinned canary from *memory_store* and compare hashes.

    Tampered canaries (hash mismatch) and missing canaries (not found)
    are both reported. Emits a ``canary.verify`` structlog event.
    """
    tampered: list[str] = []
    missing: list[str] = []
    ok = 0
    for canary_id, expected in PINNED_HASHES.items():
        got_content = memory_store.read(canary_id)
        if got_content is None:
            missing.append(canary_id)
            continue
        if _sha(got_content) != expected:
            tampered.append(canary_id)
            continue
        ok += 1
    result = CanaryVerificationResult(
        total=len(PINNED_HASHES),
        ok=ok,
        tampered=tampered,
        missing=missing,
    )
    _LOG.info(
        "canary.verify",
        total=result.total,
        ok=result.ok,
        tampered_count=len(tampered),
        missing_count=len(missing),
    )
    return result
