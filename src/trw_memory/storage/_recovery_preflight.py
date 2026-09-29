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
_RECOVERING_SUFFIX = ".recovering"


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


def _db_identity(db_path: Path) -> list[int] | None:
    """Return ``[st_dev, st_ino]`` for *db_path*, or ``None`` if it cannot be stat'd."""
    with contextlib.suppress(OSError):
        st = db_path.stat()
        return [st.st_dev, st.st_ino]
    return None


def _sidecar_identity(db_path: Path) -> list[int] | None:
    """``[st_dev, st_ino, st_ctime_ns]`` for *db_path*, or ``None`` if it cannot be stat'd.

    The inode alone does not tell a replaced file apart on Linux, which reuses a freed inode
    number at once, so a store unlinked and recreated at the same path came back with the same
    ``(st_dev, st_ino)``. The change time moves on every write, rename or re-create.
    """
    with contextlib.suppress(OSError):
        st = db_path.stat()
        return [st.st_dev, st.st_ino, st.st_ctime_ns]
    return None


def write_recovery_state(db_path: Path, *, status: str, reason: str, db_size_bytes: int) -> None:
    """Persist additive recovery state for future bounded-open decisions.

    Binds the sidecar to the db file's identity (``(st_dev, st_ino)``) so a
    later read can tell a replaced/restored file apart from the one the
    verdict was recorded against (B71-03) — a sidecar is keyed by path, so
    without this a swapped-in file at the same path silently inherits the
    prior file's verdict.
    """
    state_path = recovery_state_path(db_path)
    payload: dict[str, object] = {
        "status": status,
        "reason": reason,
        "db_size_bytes": db_size_bytes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    identity = _sidecar_identity(db_path)
    if identity is not None:
        payload["db_identity"] = identity
    with contextlib.suppress(OSError):
        _write_json_atomic(state_path, payload)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write *payload* to *path* durably: a synced temp file renamed over it, so a reader sees all of it or none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def recovery_marker_path(db_path: Path) -> Path:
    """``<db>.recovering``: present from just before a recovery rotates the store until it has restored the rows."""
    return db_path.with_name(f"{db_path.name}{_RECOVERING_SUFFIX}")


def write_recovery_marker(db_path: Path) -> None:
    """Record, before the rotation, the identity of the store a recovery is about to move aside (B71-133 (a)).

    A rename keeps the inode, so the rotated backup is found by it: a recovery killed anywhere between the
    rotation and the restored rows is resumed by the next open instead of leaving it an empty store."""
    identity = _db_identity(db_path)
    if identity is None:
        return
    try:
        _write_json_atomic(recovery_marker_path(db_path), {"db_identity": identity})
    except OSError:
        logger.warning("recovery_marker_write_failed", reason="a recovery killed now cannot be resumed", exc_info=True)


def read_recovery_marker(db_path: Path) -> list[int] | None:
    """The identity an unfinished recovery recorded, or None (no marker, or one no recovery could have written)."""
    try:
        identity = json.loads(recovery_marker_path(db_path).read_bytes()).get("db_identity")
    except (OSError, ValueError, AttributeError):  # trw-fail-silent-allow: no marker, or not ours: none to resume
        return None
    valid = isinstance(identity, list) and len(identity) == 2 and all(isinstance(n, int) for n in identity)
    return identity if valid else None


def clear_recovery_marker(db_path: Path) -> None:
    recovery_marker_path(db_path).unlink(missing_ok=True)


def _read_persisted_recovery_status(state_path: Path, *, db_path: Path) -> str:
    """Read the advisory recovery-state sidecar status, fail-closed to ``""``.

    The sidecar is *advisory*: an absent, unreadable, non-UTF-8, malformed,
    non-object, or non-string-status file must never break bounded startup
    classification. Returns the ``status`` string only when the sidecar holds a
    JSON object whose ``status`` field is itself a string; otherwise ``""``.

    B71-03: a sidecar is keyed by *path*, not by the db file's identity. A
    replaced or restored store (same path, new inode) must not inherit the
    old verdict, so the recorded ``db_identity`` — ``[st_dev, st_ino]`` — is
    compared against the CURRENT ``db_path``, and a mismatch is treated as an
    absent sidecar (``""``). A sidecar with no recorded identity keeps its
    verdict: it was written after a failed strict recovery removed the db, or
    before this field existed, and dropping it would let the next open start an
    empty store (sol review of the B71-03 fix).

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

    # Only a RECORDED identity that no longer matches clears the verdict: a sidecar written after a
    # failed recovery removed the db (no identity), or before this field existed, keeps it (sol r1).
    recorded_identity = parsed.get("db_identity")
    if recorded_identity is not None and recorded_identity != _sidecar_identity(db_path):
        logger.debug("recovery_state_unreadable", reason="identity_mismatch")
        return ""
    return status


def classify_recovery_preflight(db_path: Path, *, inline_max_bytes: int) -> RecoveryPreflight:
    """Classify whether startup can recover inline or should degrade/fail early."""
    state_path = recovery_state_path(db_path)
    db_size_bytes = 0
    with contextlib.suppress(OSError):
        db_size_bytes = db_path.stat().st_size

    persisted_status = _read_persisted_recovery_status(state_path, db_path=db_path)

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
