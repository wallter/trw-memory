"""FR-003 — Recall-time pattern filter.

Sprint 96 W1-E scaffolding in ``observe_mode=True``. Accepts a window of
``MemoryEntry`` (the trw-memory equivalent of the spec's ``LearnIn``)
and returns both the passthrough ``accepted`` list and a ``would_reject``
list for telemetry. In observe mode, ``accepted`` always contains the
full original input.

Target: p95 latency <=20ms for a 25-learning window (PRD-SEC-001 NFR-002).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from trw_memory.models.memory import MemoryEntry
from trw_memory.security.pii import strip_pii
from trw_memory.security.poisoning import _INJECTION_PATTERNS
from trw_memory.storage.persistence import lock_for_rmw

__all__ = ["RecallDecision", "RecallFilterResult", "filter_recall_window"]

_LOG = structlog.get_logger(__name__)
_LATENCY_BUDGET_MS = 20.0


Action = Literal["allow", "redact", "block"]


class RecallDecision(BaseModel):
    """Decision for a single recalled entry."""

    model_config = ConfigDict(strict=True)

    action: Action
    reasons: list[str] = Field(default_factory=list)


class RecallFilterResult(BaseModel):
    """Outcome of filtering a recall window.

    In observe mode ``accepted`` equals the full input; ``would_reject``
    collects entries that the enforcing filter would have blocked,
    keyed in ``reasons`` by entry id.
    """

    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    accepted: list[MemoryEntry]
    would_reject: list[MemoryEntry] = Field(default_factory=list)
    reasons: dict[str, list[str]] = Field(default_factory=dict)
    actions: dict[str, Action] = Field(default_factory=dict)


def _inspect(entry: MemoryEntry) -> list[str]:
    """Return a list of recall-filter reasons; empty list = pass."""
    reasons: list[str] = []
    combined = f"{entry.content}{entry.detail}"
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(combined):
            reasons.append(f"injection_pattern:{pattern.pattern}")
            break

    # Hash-pin drift: if metadata carries ``content_hash`` (provenance
    # record), recompute and compare.
    pinned = entry.metadata.get("provenance_content_hash") or entry.metadata.get("content_hash")
    if pinned:
        current = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        if current != pinned:
            reasons.append("hash_pin_drift")
    return reasons


def _redact_entry(entry: MemoryEntry) -> MemoryEntry:
    redacted_content = entry.content
    redacted_detail = entry.detail
    for pattern in _INJECTION_PATTERNS:
        redacted_content = pattern.sub("[redacted]", redacted_content)
        redacted_detail = pattern.sub("[redacted]", redacted_detail)
    return entry.model_copy(update={"content": redacted_content, "detail": redacted_detail})


def _decide(entry: MemoryEntry, *, mode: Literal["strict", "redact", "observe"]) -> RecallDecision:
    reasons = _inspect(entry)
    if not reasons:
        return RecallDecision(action="allow", reasons=[])
    if mode == "observe":
        return RecallDecision(action="allow", reasons=reasons)
    if mode == "redact" and all(reason != "hash_pin_drift" for reason in reasons):
        return RecallDecision(action="redact", reasons=reasons)
    return RecallDecision(action="block", reasons=reasons)


def _shadow_quarantine(
    entry: MemoryEntry,
    reasons: list[str],
    quarantine_dir: Path,
) -> None:
    """Append a shadow quarantine record (FR-004 observe partition).

    Fail-open: any I/O error is logged and swallowed. The record captures
    what WOULD have been routed to a quarantine store in enforce mode.
    """
    try:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        shadow_path = quarantine_dir / "quarantined_entries.jsonl"
        record = {
            "id": entry.id,
            "reasons": reasons,
            # Redact PII before persisting: this entry was flagged (often FOR a
            # PII/injection pattern), so writing its raw content to the shadow
            # partition would leak the very secret/email that triggered the flag.
            # strip_pii preserves injection-pattern structure for forensics while
            # masking emails/API keys/PATs.
            "content_preview": strip_pii(entry.content[:120]),
            "shadowed_at": datetime.now(timezone.utc).isoformat(),
            "mode": "observe",
        }
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with lock_for_rmw(shadow_path):
            if shadow_path.is_symlink():
                raise OSError("refusing symlink quarantine shadow path")
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(shadow_path, flags, stat.S_IRUSR | stat.S_IWUSR)
            original_size = os.fstat(fd).st_size
            try:
                if shadow_path.is_symlink():
                    raise OSError("quarantine shadow path became a symlink")
                fchmod = getattr(os, "fchmod", None)
                if fchmod is not None:
                    fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("failed to append quarantine shadow record")
                    remaining = remaining[written:]
            except BaseException:
                with contextlib.suppress(OSError):
                    os.ftruncate(fd, original_size)
                raise
            finally:
                os.close(fd)
        _LOG.info(
            "recall_filter.quarantine_shadow",
            action="quarantine_shadow",
            entry_id=entry.id,
            shadow_path=str(shadow_path),
        )
    except OSError:  # justified: fail-open on I/O
        _LOG.warning(
            "recall_filter.quarantine_shadow_failed",
            entry_id=entry.id,
            exc_info=True,
        )


def filter_recall_window(
    learnings: list[MemoryEntry],
    *,
    mode: Literal["strict", "redact", "observe"] | None = None,
    observe_mode: bool | None = None,
    quarantine_dir: Path | None = None,
) -> RecallFilterResult:
    """Filter a recall window for injection patterns and hash drift.

    ``mode`` controls live behavior:
    - ``strict``: block matched entries
    - ``redact``: redact injection content unless the reason is hash drift
    - ``observe``: return originals while recording would-block entries

    When *quarantine_dir* is provided, every ``would_reject`` entry is
    ALSO appended to a JSONL shadow partition at
    ``<quarantine_dir>/quarantined_entries.jsonl`` (FR-004 observe-mode
    scaffold). The entry is still dropped from recall results in the
    same manner as before; the shadow partition simply records what
    enforce-mode would have done.

    ``observe_mode`` is retained as a backwards-compatible alias for the
    pre-SEC call surface: ``True`` maps to ``mode="observe"`` and
    ``False`` maps to ``mode="strict"``.
    """
    legacy_mode: Literal["observe", "strict"] = "observe" if observe_mode is not None and observe_mode else "strict"
    if mode is None:
        resolved_mode: Literal["strict", "redact", "observe"] = legacy_mode if observe_mode is not None else "observe"
    else:
        resolved_mode = mode
        if observe_mode is not None and resolved_mode != legacy_mode:
            raise ValueError("mode and observe_mode disagree")

    t0 = time.monotonic_ns()
    would_reject: list[MemoryEntry] = []
    reasons: dict[str, list[str]] = {}
    actions: dict[str, Action] = {}
    accepted_enforce: list[MemoryEntry] = []

    for entry in learnings:
        decision = _decide(entry, mode=resolved_mode)
        actions[entry.id] = decision.action
        if decision.reasons:
            would_reject.append(entry)
            reasons[entry.id] = decision.reasons
            if quarantine_dir is not None:
                _shadow_quarantine(entry, decision.reasons, quarantine_dir)
        if decision.action == "allow":
            accepted_enforce.append(entry)
        elif decision.action == "redact":
            accepted_enforce.append(_redact_entry(entry))

    accepted = list(learnings) if resolved_mode == "observe" else accepted_enforce

    elapsed_ms = (time.monotonic_ns() - t0) / 1_000_000.0

    _LOG.info(
        "recall_filter.observe" if resolved_mode == "observe" else "recall_filter.enforce",
        window_size=len(learnings),
        would_reject_count=len(would_reject),
        latency_ms=round(elapsed_ms, 3),
        mode=resolved_mode,
        shadow_partition=str(quarantine_dir) if quarantine_dir else None,
    )

    if elapsed_ms > _LATENCY_BUDGET_MS and len(learnings) <= 25:
        _LOG.warning(
            "recall_filter.latency_budget_exceeded",
            latency_ms=round(elapsed_ms, 3),
            budget_ms=_LATENCY_BUDGET_MS,
            window_size=len(learnings),
        )

    return RecallFilterResult(accepted=accepted, would_reject=would_reject, reasons=reasons, actions=actions)
