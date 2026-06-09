"""Bounded open-time recovery classification + advisory state sidecar.

Split from ``_recovery.py`` (PRD-DIST-245 effective-LOC ratchet) so the
corrupt-DB salvage orchestration and the startup preflight/state-persistence
concern live in separate, single-responsibility modules. ``_recovery.py``
re-exports the public names below for back-compat, so importers that resolve
``classify_recovery_preflight`` / ``write_recovery_state`` /
``recovery_state_path`` / ``RecoveryPreflight`` from ``_recovery`` keep working.

The sidecar (``<db>.recovery.json``) is *advisory*: an absent, unreadable,
non-UTF-8, malformed, non-object, or non-string-status file must never break
bounded startup classification — every read fails closed to ``""``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)

_RECOVERY_STATE_SUFFIX = ".recovery.json"


@dataclass(frozen=True)
class RecoveryPreflight:
    """Bounded open-time recovery classification."""

    classification: Literal["fast_open", "degraded_open_with_background_recovery", "hard_fail"]
    reason: str
    db_size_bytes: int
    state_path: str
    persisted_status: str = ""


def recovery_state_path(db_path: Path) -> Path:
    """Return the sidecar path used for persisted recovery state."""
    return db_path.with_name(f"{db_path.name}{_RECOVERY_STATE_SUFFIX}")


def write_recovery_state(db_path: Path, *, status: str, reason: str, db_size_bytes: int) -> None:
    """Persist additive recovery state for future bounded-open decisions."""
    state_path = recovery_state_path(db_path)
    payload = {
        "status": status,
        "reason": reason,
        "db_size_bytes": db_size_bytes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with contextlib.suppress(OSError):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(state_path.parent),
            prefix=f".{state_path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(state_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise


def _read_persisted_recovery_status(state_path: Path) -> str:
    """Read the advisory recovery-state sidecar status, fail-closed to ``""``.

    The sidecar is *advisory*: an absent, unreadable, non-UTF-8, malformed,
    non-object, or non-string-status file must never break bounded startup
    classification. Returns the ``status`` string only when the sidecar holds a
    JSON object whose ``status`` field is itself a string; otherwise ``""``.

    Reads raw bytes and decodes explicitly so non-UTF-8 content surfaces as a
    caught ``UnicodeDecodeError`` (a ``ValueError`` subclass that escapes
    ``suppress(OSError, JSONDecodeError)``) rather than crashing the caller.

    Diagnostics are content-free — ``reason``/``error_type`` only. The filesystem
    path, raw bytes, and decoded payload are never logged, so a poisoned or
    secret-bearing sidecar cannot leak through startup logs.
    """
    try:
        raw_bytes = state_path.read_bytes()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        logger.debug("recovery_state_unreadable", reason="read_failed", error_type=type(exc).__name__)
        return ""

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("recovery_state_unreadable", reason="non_utf8")
        return ""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("recovery_state_unreadable", reason="malformed_json")
        return ""

    if not isinstance(parsed, dict):
        logger.debug("recovery_state_unreadable", reason="non_object_json")
        return ""

    status = parsed.get("status", "")
    if not isinstance(status, str):
        logger.debug("recovery_state_unreadable", reason="non_string_status")
        return ""
    return status


def classify_recovery_preflight(db_path: Path, *, inline_max_bytes: int) -> RecoveryPreflight:
    """Classify whether startup can recover inline or should degrade/fail early."""
    state_path = recovery_state_path(db_path)
    db_size_bytes = 0
    with contextlib.suppress(OSError):
        db_size_bytes = db_path.stat().st_size

    persisted_status = _read_persisted_recovery_status(state_path)

    if persisted_status == "hard_fail":
        return RecoveryPreflight(
            classification="hard_fail",
            reason="previous_recovery_hard_fail",
            db_size_bytes=db_size_bytes,
            state_path=str(state_path),
            persisted_status=persisted_status,
        )

    if inline_max_bytes > 0 and db_size_bytes > inline_max_bytes:
        return RecoveryPreflight(
            classification="degraded_open_with_background_recovery",
            reason="db_exceeds_inline_recovery_budget",
            db_size_bytes=db_size_bytes,
            state_path=str(state_path),
            persisted_status=persisted_status,
        )

    if persisted_status in {"pending", "running", "degraded_open_with_background_recovery"}:
        return RecoveryPreflight(
            classification="degraded_open_with_background_recovery",
            reason="recovery_already_pending",
            db_size_bytes=db_size_bytes,
            state_path=str(state_path),
            persisted_status=persisted_status,
        )

    return RecoveryPreflight(
        classification="fast_open",
        reason="within_inline_recovery_budget",
        db_size_bytes=db_size_bytes,
        state_path=str(state_path),
        persisted_status=persisted_status,
    )
