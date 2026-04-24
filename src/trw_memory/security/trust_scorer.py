"""FR-001 — Intake trust-scoring for ``trw_learn`` write path.

Sprint 96 W1-E scaffolding: runs in ``observe_mode=True`` by default.
Emits ``trust_scorer.observe`` structlog events with would-be decisions.

Heuristics:
- Injection-pattern match against the regex corpus in
  :mod:`trw_memory.security.poisoning`.
- Size vs a caller-supplied rolling baseline (``metadata["size_baseline"]``).
- Presence of ``source_identity`` in metadata.

Target: p95 latency <=50ms per learning (PRD-SEC-001 NFR-001).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from trw_memory.security.poisoning import _INJECTION_PATTERNS

__all__ = ["TrustScore", "score_intake"]

# PRD-SEC-001 Item 5: 14-day observe clock. Start once per anchor dir.
_clock_started_for: set[str] = set()

_LOG = structlog.get_logger(__name__)

# p95 latency budget (ms) — logged if exceeded. PRD-SEC-001 NFR-001.
_LATENCY_BUDGET_MS = 50.0

# Default absolute size ceiling used when the caller does not supply a
# rolling baseline in ``metadata``. A single learning larger than this is
# anomalous on its own.
_DEFAULT_SIZE_CEILING = 100_000

Decision = Literal["allow", "quarantine", "reject"]


class TrustScore(BaseModel):
    """Result of scoring a single intake payload."""

    model_config = ConfigDict(strict=True)

    score: float = Field(ge=0.0, le=1.0)
    decision: Decision
    reasons: list[str] = Field(default_factory=list)


def _classify(content: str, metadata: dict[str, str]) -> tuple[float, Decision, list[str]]:
    """Apply heuristics; return ``(score, decision, reasons)``."""
    reasons: list[str] = []
    score = 1.0

    # 1) Injection pattern match → reject.
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(content):
            reasons.append(f"injection_pattern:{pattern.pattern}")
            score -= 0.6
            break

    # 2) Size heuristic — caller baseline or default ceiling.
    baseline_raw = metadata.get("size_baseline")
    size_ceiling: float
    try:
        size_ceiling = float(baseline_raw) * 3.0 if baseline_raw else float(_DEFAULT_SIZE_CEILING)
    except (TypeError, ValueError):
        size_ceiling = float(_DEFAULT_SIZE_CEILING)
    size = len(content)
    if size > size_ceiling:
        reasons.append(f"size_anomaly:{size}>{int(size_ceiling)}")
        score -= 0.3

    # 3) Source identity presence.
    if not metadata.get("source_identity"):
        reasons.append("missing_source_identity")
        score -= 0.15

    score = max(0.0, min(1.0, score))
    if score >= 0.7:
        decision: Decision = "allow"
    elif score >= 0.4:
        decision = "quarantine"
    else:
        decision = "reject"
    return score, decision, reasons


def score_intake(
    content: str,
    metadata: dict[str, str],
    *,
    observe_mode: bool = True,
    trw_dir: Path | None = None,
) -> TrustScore:
    """Score an intake payload.

    In ``observe_mode=True`` (Sprint 96 default) the returned decision is
    always ``"allow"``; the would-be decision is recorded in ``reasons``
    prefixed with ``"WOULD-BE:"`` and a ``trust_scorer.observe`` event
    is emitted.

    When *trw_dir* is provided, the 14-day observe calibration clock is
    started once per process (PRD-SEC-001 Item 5). Fail-open: any clock
    error is logged and swallowed.
    """
    global _clock_started_for
    if trw_dir is not None and str(trw_dir) not in _clock_started_for:
        _clock_started_for.add(str(trw_dir))
        try:
            from trw_memory.security.observe_clock import start_observe_clock

            start_observe_clock(trw_dir)
        except Exception:  # justified: fail-open, clock is advisory
            _LOG.warning("trust_scorer.observe_clock_start_failed", exc_info=True)

    t0 = time.monotonic_ns()
    score, would_be, reasons = _classify(content, metadata)

    if observe_mode:
        reasons = [f"WOULD-BE:{would_be}", *reasons]
        decision: Decision = "allow"
    else:
        decision = would_be

    elapsed_ms = (time.monotonic_ns() - t0) / 1_000_000.0

    if observe_mode:
        _LOG.info(
            "trust_scorer.observe",
            score=score,
            would_be=would_be,
            reasons=reasons,
            latency_ms=round(elapsed_ms, 3),
            observe_mode=True,
        )

    if elapsed_ms > _LATENCY_BUDGET_MS:
        _LOG.warning(
            "trust_scorer.latency_budget_exceeded",
            latency_ms=round(elapsed_ms, 3),
            budget_ms=_LATENCY_BUDGET_MS,
        )

    return TrustScore(score=score, decision=decision, reasons=reasons)
