"""WAL-checkpoint types + the single-connection checkpoint primitive.

Belongs to the ``sqlite_backend.py`` facade. The public types
(:data:`CheckpointMode`, :class:`CheckpointResult`) are re-exported from
``trw_memory.storage`` so downstream consumers (e.g. ``trw-mcp``'s
``maybe_checkpoint_wal``) import a precise, shared contract instead of an
opaque ``dict[str, object]``.

Background
----------
SQLite < 3.51.3 carries the WAL-reset corruption bug
(sqlite.org/wal.html §walresetbug): a *resetting* checkpoint
(``TRUNCATE``/``RESTART``) that races a second connection's checkpoint/write
can leave the WAL-index header inconsistent, so a later checkpoint skips a
committed transaction. The fix that does not depend on the engine version is
to (a) run every checkpoint on the backend's single owning connection under
its lock — there is then never a second checkpointer to race — and (b) on an
unsafe engine, downgrade resetting modes to ``PASSIVE`` (which never resets
the WAL, so it cannot trigger the bug regardless of connection/process count).
This module implements (b)'s mode coercion and the ``busy``-aware fallback;
the caller supplies the locked, owning connection for (a).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Literal, cast, get_args

import structlog
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = structlog.get_logger(__name__)

# The four real SQLite checkpoint modes that ``run_checkpoint`` may execute.
RunMode = Literal["PASSIVE", "FULL", "RESTART", "TRUNCATE"]
# What the result reports: a run mode, or the synthetic ``"error"`` sentinel
# returned when the PRAGMA itself raises (fail-open).
CheckpointMode = Literal["PASSIVE", "FULL", "RESTART", "TRUNCATE", "error"]

_VALID_MODES: frozenset[str] = frozenset(get_args(RunMode))
# The two modes that reset the WAL and therefore must be gated on unsafe
# engines (and are the only modes that fall back to PASSIVE when busy).
RESETTING_MODES: frozenset[RunMode] = frozenset({"TRUNCATE", "RESTART"})


class CheckpointResult(TypedDict):
    """Structured outcome of a WAL checkpoint.

    Fields:
        busy: ``1`` when readers held pages and the checkpoint could not run
            to completion (or the checkpoint errored), else ``0``.
        checkpointed: Frames written back to the main DB (column 2 of the
            ``wal_checkpoint`` result row).
        mode: The mode that actually ran — may differ from the requested mode
            when an unsafe engine downgraded a resetting checkpoint to
            ``PASSIVE`` or when a busy resetting checkpoint fell back.
            ``"error"`` signals the PRAGMA raised.
    """

    busy: int
    checkpointed: int
    mode: CheckpointMode


def normalize_mode(mode: str, *, wal_reset_safe: bool) -> RunMode:
    """Coerce a requested mode to a safe, valid checkpoint mode.

    - Unknown modes (incl. injection attempts) collapse to ``PASSIVE``.
    - On an engine WITHOUT the WAL-reset fix, resetting modes
      (``TRUNCATE``/``RESTART``) downgrade to ``PASSIVE`` — the gate that
      neutralizes the corruption trigger and self-resolves once a fixed
      engine (>=3.51.3) is installed.
    """
    normalized = mode.upper()
    if normalized not in _VALID_MODES:
        return "PASSIVE"
    run_mode = cast("RunMode", normalized)
    if not wal_reset_safe and run_mode in RESETTING_MODES:
        return "PASSIVE"
    return run_mode


def _read_checkpoint_row(row: object) -> tuple[int, int]:
    """Map a ``wal_checkpoint`` result row to ``(busy, checkpointed)``.

    The PRAGMA returns a ``(busy, log_frames, checkpointed_frames)`` sequence
    (a ``sqlite3.Row`` / ``pysqlite3`` ``Row`` / tuple). A missing or empty row
    is treated as ``busy`` so callers never read an uninitialised result.
    """
    if not row:
        return 1, 0
    seq = cast("Sequence[object]", row)
    busy = int(cast("int", seq[0]))
    checkpointed_raw = seq[2] if len(seq) > 2 else None
    checkpointed = int(cast("int", checkpointed_raw)) if checkpointed_raw is not None else 0
    return busy, checkpointed


def run_checkpoint(
    execute_pragma: Callable[[str], object],
    requested_mode: str,
    *,
    wal_reset_safe: bool,
    db_path: str,
    db_error: type[Exception] = sqlite3.Error,
) -> CheckpointResult:
    """Run a WAL checkpoint via *execute_pragma*, returning a :class:`CheckpointResult`.

    *execute_pragma* runs the given ``PRAGMA wal_checkpoint(...)`` on the single
    owning connection (the caller holds that connection's lock) and returns the
    result row. This function never opens a connection — that would be the
    two-connection race the fix exists to prevent.

    A resetting checkpoint that returns ``busy=1`` (readers held pages) falls
    back to a non-blocking ``PASSIVE`` checkpoint on the SAME connection.
    Fail-open: any active DB-API ``db_error`` yields ``mode="error"`` and is
    logged, never raised.
    """
    used: RunMode = normalize_mode(requested_mode, wal_reset_safe=wal_reset_safe)
    try:
        busy, checkpointed = _read_checkpoint_row(execute_pragma(f"PRAGMA wal_checkpoint({used})"))
        if busy == 1 and used in RESETTING_MODES:
            # Readers held pages; retry the non-blocking PASSIVE checkpoint on
            # this same connection (no second connection -> no race).
            busy, checkpointed = _read_checkpoint_row(execute_pragma("PRAGMA wal_checkpoint(PASSIVE)"))
            used = "PASSIVE"
    except db_error as exc:
        logger.warning("wal_checkpoint_failed", error=str(exc), db=db_path)
        return CheckpointResult(busy=1, checkpointed=0, mode="error")
    logger.debug(
        "wal_checkpoint_done",
        db=db_path,
        requested=requested_mode.upper(),
        mode=used,
        busy=busy,
        checkpointed=checkpointed,
    )
    return CheckpointResult(busy=busy, checkpointed=checkpointed, mode=used)
