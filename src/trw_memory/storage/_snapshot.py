"""Snapshot rotation for :class:`SQLiteBackend` (PRD-INFRA-065 / B4).

VACUUM INTO-based hot backup with daily + weekly rotation. Opt-in: invoked
only when ``config.memory_snapshot_enabled=True``. Snapshots live under
``<base_dir>/memory/snapshots/daily/YYYY-MM-DD.db`` and
``<base_dir>/memory/snapshots/weekly/YYYY-Www.db`` (ISO week). Rotation
follows the PRD-CORE-139 oldest-by-filename-timestamp pattern — never
consults ``st_mtime``.

Key invariants (do not relax without updating PRD-INFRA-065):

- Snapshots are WAL-safe because ``VACUUM INTO`` is a transactional
  operation — the destination is an atomic copy at a consistent read point.
- Daily snapshots overwrite same-day files intentionally — one per day is
  enough. Weekly snapshots are only taken on Sunday UTC (collapse frequency).
- Pruning uses filename-parsed dates exclusively. Unparseable names are
  skipped (never deleted, never counted).
- All public functions are safe to call under `config.local_only=True`;
  snapshots are strictly local filesystem artifacts.
- Destination paths are validated against a base directory
  (``_assert_within_snapshots_dir``) to prevent path traversal on restore.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import structlog

__all__ = [
    "SnapshotError",
    "SnapshotRotationResult",
    "create_snapshot",
    "list_snapshots",
    "restore_from_snapshot",
    "rotate_snapshots",
    "snapshots_base_dir",
    "take_daily_snapshot",
    "take_weekly_snapshot",
]

logger = structlog.get_logger(__name__)

# Daily snapshot filename: YYYY-MM-DD.db
_DAILY_RE: re.Pattern[str] = re.compile(r"^(\d{4}-\d{2}-\d{2})\.db$")
# Weekly snapshot filename: YYYY-Www.db (ISO year/week)
_WEEKLY_RE: re.Pattern[str] = re.compile(r"^(\d{4}-W\d{2})\.db$")


class SnapshotError(Exception):
    """Raised when snapshot create/restore fails in a way the caller must surface.

    Rotation pruning errors are NEVER raised — they are logged at DEBUG and
    swallowed. This exception is reserved for create/restore operations
    that the caller explicitly requested and has no fallback for.
    """


class SnapshotRotationResult:
    """Return shape for :func:`rotate_snapshots` — counts for observability."""

    __slots__ = ("daily_kept", "daily_pruned", "weekly_kept", "weekly_pruned")

    def __init__(
        self,
        daily_kept: int = 0,
        daily_pruned: int = 0,
        weekly_kept: int = 0,
        weekly_pruned: int = 0,
    ) -> None:
        self.daily_kept = daily_kept
        self.daily_pruned = daily_pruned
        self.weekly_kept = weekly_kept
        self.weekly_pruned = weekly_pruned

    def as_dict(self) -> dict[str, int]:
        return {
            "daily_kept": self.daily_kept,
            "daily_pruned": self.daily_pruned,
            "weekly_kept": self.weekly_kept,
            "weekly_pruned": self.weekly_pruned,
        }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def snapshots_base_dir(base_dir: Path) -> Path:
    """Return the snapshots root directory for a given base dir.

    The base dir mirrors the cold-tier layout:
    ``<base_dir>/memory/snapshots/``.
    """
    return base_dir / "memory" / "snapshots"


def _daily_dir(base_dir: Path) -> Path:
    return snapshots_base_dir(base_dir) / "daily"


def _weekly_dir(base_dir: Path) -> Path:
    return snapshots_base_dir(base_dir) / "weekly"


def _assert_within_snapshots_dir(base_dir: Path, candidate: Path) -> None:
    """Reject ``candidate`` if it escapes the snapshots root after resolution.

    Used during restore to block path-traversal attacks (``../../etc/passwd``).
    Resolves both paths so symlinks cannot slip out of the snapshots root.
    """
    resolved_base = snapshots_base_dir(base_dir).resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_base):
        raise SnapshotError(f"Path traversal guard: {candidate} is not under snapshots dir {resolved_base}")


# ---------------------------------------------------------------------------
# Creation via VACUUM INTO
# ---------------------------------------------------------------------------


def create_snapshot(db_path: Path, dest: Path) -> Path:
    """Create a hot snapshot at ``dest`` via ``VACUUM INTO``.

    Args:
        db_path: Source database file. Must exist.
        dest: Destination path. Parent directory is created if missing.
            If ``dest`` already exists, it is overwritten atomically via
            ``os.replace`` after the VACUUM INTO lands in a sibling tempfile.

    Returns:
        The path to the newly-created snapshot.

    Raises:
        SnapshotError: if ``db_path`` is missing, ``dest`` is invalid, or
            ``VACUUM INTO`` fails.
    """
    if not db_path.exists():
        raise SnapshotError(f"source database does not exist: {db_path}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # VACUUM INTO refuses to overwrite an existing file. Stage to a sibling
    # tempfile and atomic-replace on success.
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with contextlib.suppress(OSError):
        tmp.unlink(missing_ok=True)

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        # Use parameterized-literal quoting — VACUUM INTO does not support
        # bind parameters, so escape single quotes defensively.
        escaped = str(tmp).replace("'", "''")
        conn.execute(f"VACUUM INTO '{escaped}'")
    except sqlite3.Error as exc:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise SnapshotError(f"VACUUM INTO failed: {exc}") from exc
    finally:
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    # Atomic replace.
    try:
        tmp.replace(dest)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise SnapshotError(f"snapshot rename failed: {exc}") from exc

    size = dest.stat().st_size
    logger.info(
        "snapshot_created",
        source=str(db_path),
        dest=str(dest),
        size_bytes=size,
    )
    return dest


# ---------------------------------------------------------------------------
# Public rotation entry points
# ---------------------------------------------------------------------------


def take_daily_snapshot(
    base_dir: Path,
    db_path: Path,
    keep_daily: int = 7,
    *,
    now: datetime | None = None,
) -> Path:
    """Write ``daily/YYYY-MM-DD.db`` and prune to ``keep_daily``.

    Same-day invocations overwrite the existing daily file — the last
    write of the day wins. Pruning runs after creation.

    Args:
        base_dir: Root memory directory (e.g. ``<trw_dir>``).
        db_path: Source DB file.
        keep_daily: Number of daily snapshots retained.
        now: UTC datetime used for filename (defaults to :func:`datetime.now`).

    Returns:
        Path to the created snapshot.
    """
    stamp = _date_stamp(now)
    dest = _daily_dir(base_dir) / f"{stamp}.db"
    result = create_snapshot(db_path, dest)
    rotate_snapshots(base_dir, keep_daily=keep_daily, keep_weekly=-1)
    return result


def take_weekly_snapshot(
    base_dir: Path,
    db_path: Path,
    keep_weekly: int = 4,
    *,
    now: datetime | None = None,
    force: bool = False,
) -> Path | None:
    """Write ``weekly/YYYY-Www.db`` if today is Sunday UTC (or ``force=True``).

    Returns ``None`` when called on a non-Sunday without ``force``. Pruning
    runs after creation.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    # ISO weekday: Monday=1 ... Sunday=7. Take weekly snapshots on Sundays.
    if not force and now.isoweekday() != 7:
        return None
    iso_year, iso_week, _ = now.isocalendar()
    stamp = f"{iso_year:04d}-W{iso_week:02d}"
    dest = _weekly_dir(base_dir) / f"{stamp}.db"
    result = create_snapshot(db_path, dest)
    rotate_snapshots(base_dir, keep_daily=-1, keep_weekly=keep_weekly)
    return result


def rotate_snapshots(
    base_dir: Path,
    keep_daily: int,
    keep_weekly: int,
) -> SnapshotRotationResult:
    """Prune daily and weekly snapshots to the configured budgets.

    Negative ``keep_*`` values skip that tier entirely — useful for the
    per-tier rotation calls from :func:`take_daily_snapshot` and
    :func:`take_weekly_snapshot`.

    Pruning is always by oldest-filename-first. Files whose names do not
    match the tier regex are skipped (never deleted, never counted).
    """
    result = SnapshotRotationResult()
    if keep_daily >= 0:
        kept, pruned = _prune_tier(_daily_dir(base_dir), _DAILY_RE, keep_daily)
        result.daily_kept = kept
        result.daily_pruned = pruned
    if keep_weekly >= 0:
        kept, pruned = _prune_tier(_weekly_dir(base_dir), _WEEKLY_RE, keep_weekly)
        result.weekly_kept = kept
        result.weekly_pruned = pruned
    if result.daily_pruned or result.weekly_pruned:
        logger.info(
            "snapshots_rotated",
            base_dir=str(base_dir),
            **result.as_dict(),
        )
    return result


# ---------------------------------------------------------------------------
# Listing + restore
# ---------------------------------------------------------------------------


def list_snapshots(base_dir: Path) -> dict[str, list[Path]]:
    """Return parseable snapshots sorted newest-first per tier.

    Unparseable files are silently skipped.
    """
    return {
        "daily": _sorted_parseable(_daily_dir(base_dir), _DAILY_RE, reverse=True),
        "weekly": _sorted_parseable(_weekly_dir(base_dir), _WEEKLY_RE, reverse=True),
    }


def restore_from_snapshot(base_dir: Path, snapshot: Path, db_path: Path) -> None:
    """Copy ``snapshot`` over ``db_path`` after validating path safety.

    Args:
        base_dir: Root memory directory — used to enforce path boundary.
        snapshot: Snapshot source (daily or weekly). MUST be under
            ``base_dir/memory/snapshots/``.
        db_path: Target DB path to overwrite.

    Raises:
        SnapshotError: if ``snapshot`` is outside the snapshots directory,
            the source does not exist, or the copy fails.
    """
    _assert_within_snapshots_dir(base_dir, snapshot)
    if not snapshot.exists():
        raise SnapshotError(f"snapshot not found: {snapshot}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(snapshot), str(db_path))
    except OSError as exc:
        raise SnapshotError(f"snapshot restore failed: {exc}") from exc
    logger.info(
        "snapshot_restored",
        snapshot=str(snapshot),
        dest=str(db_path),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _prune_tier(
    tier_dir: Path,
    pattern: re.Pattern[str],
    keep: int,
) -> tuple[int, int]:
    """Prune ``tier_dir`` to at most ``keep`` files matching ``pattern``.

    Returns ``(kept, pruned)`` counts. Files that don't match the regex are
    ignored entirely. ``keep=0`` deletes all parseable files.
    """
    if not tier_dir.exists():
        return 0, 0
    # Sorted ascending by filename → oldest first.
    files = _sorted_parseable(tier_dir, pattern, reverse=False)
    total = len(files)
    if total <= keep:
        return total, 0
    victims = files[: total - keep]
    pruned = 0
    for victim in victims:
        try:
            victim.unlink()
            pruned += 1
        except OSError as exc:
            logger.debug(
                "snapshot_prune_unlink_failed",
                path=str(victim),
                error=str(exc),
            )
    return total - pruned, pruned


def _sorted_parseable(
    tier_dir: Path,
    pattern: re.Pattern[str],
    reverse: bool = False,
) -> list[Path]:
    if not tier_dir.exists():
        return []
    try:
        files = [f for f in tier_dir.iterdir() if f.is_file() and pattern.fullmatch(f.name)]
    except OSError:
        return []
    # Lexicographic sort over ``YYYY-MM-DD`` / ``YYYY-Www`` == chronological.
    files.sort(key=lambda p: p.name, reverse=reverse)
    return files


def _date_stamp(now: datetime | None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    d: date = now.date() if isinstance(now, datetime) else now
    return d.isoformat()
