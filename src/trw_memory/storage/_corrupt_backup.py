"""SQLite corrupt-backup rotation + CLI salvage helpers.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — the public API surface (``SQLiteBackend._salvage_via_recover_cli``
etc) is preserved by the parent re-exporting these as staticmethods.

PRD-CORE-138 FR04 (CLI salvage) + PRD-CORE-139 FR01-FR04 (timestamped
backup rotation) live here, isolated from the rest of SQLiteBackend.

Extracted as PRD-DIST-245 Phase 1 (cycle 85) — first sub-batch toward
sqlite_backend.py 1133→≤350 effective LOC.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Matches memory.db.corrupt.<ISO-UTC>.bak with optional -N collision suffix.
# Group 1 is the timestamp used for lexicographic (== chronological) sort.
_TIMESTAMPED_BACKUP_RE: re.Pattern[str] = re.compile(
    r"^memory\.db\.corrupt\.(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)(?:-\d+)?\.bak$"
)
# Pre-PRD-CORE-139 filenames. These are sacred: count against the keep budget
# but are never pruning victims, so forensic evidence captured before the
# rotation change is never silently destroyed on upgrade.
_LEGACY_CORRUPT_NAMES: frozenset[str] = frozenset({"memory.db.corrupt.bak", "memory.db.corrupt.bak.1"})


def salvage_via_recover_cli(backup_path: Path, dbapi: Any = sqlite3) -> list[Any]:
    """Attempt ``sqlite3 <backup> .recover`` CLI salvage.

    PRD-CORE-138 FR04 — second salvage path for cases where the in-process
    ``SELECT * FROM memories`` raises :class:`sqlite3.DatabaseError` because
    ``sqlite_master`` is damaged but B-tree pages still hold rows.

    Returns an empty list on any failure: CLI absent, non-zero exit,
    ``subprocess.TimeoutExpired``, ``FileNotFoundError``, ``sqlite3.Error``
    during dump load, or empty output.
    """
    try:
        completed = subprocess.run(  # noqa: S603 — args passed as list, shell=False
            ["sqlite3", str(backup_path), ".recover"],  # noqa: S607 — sqlite3 is on PATH
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

    if completed.returncode != 0 or not completed.stdout:
        return []

    dump_sql = completed.stdout.decode("utf-8", errors="replace")
    if not dump_sql.strip():
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "recover.db"
        try:
            tmp_conn = dbapi.connect(str(tmp_db))
            tmp_conn.row_factory = sqlite3.Row
            try:
                tmp_conn.executescript(dump_sql)
                rows = tmp_conn.execute("SELECT * FROM memories").fetchall()
                return list(rows)
            except sqlite3.Error:
                return []
            finally:
                tmp_conn.close()
        except sqlite3.Error:
            return []


def rotate_corrupt_backup(db_path: Path) -> Path:
    """Move ``db_path`` to ``memory.db.corrupt.<UTC-ISO>.bak`` (PRD-CORE-139 FR01).

    Uses ``datetime.now(timezone.utc)`` at call time. Colons in the time
    portion are replaced with hyphens for Windows/shell safety. On a
    same-second collision with an existing backup (NFR02), appends a
    ``-1``, ``-2``, ... suffix — names remain parseable by
    :data:`_TIMESTAMPED_BACKUP_RE`.

    Looks up ``datetime`` via the parent sqlite_backend module so test
    monkeypatches on ``_sqlite_backend_module.datetime`` propagate
    correctly (test-monkeypatch indirection pattern).
    """
    from trw_memory.storage import sqlite_backend as _sqlite_backend_module

    ts = _sqlite_backend_module.datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")  # type: ignore[attr-defined]
    candidate = db_path.with_name(f"memory.db.corrupt.{ts}.bak")
    i = 1
    while candidate.exists():
        candidate = db_path.with_name(f"memory.db.corrupt.{ts}-{i}.bak")
        i += 1
    shutil.move(str(db_path), str(candidate))
    return candidate


def prune_corrupt_backups(parent: Path, keep_n: int) -> None:
    """Keep at most ``keep_n`` corruption backup files in ``parent`` (FR03/FR04).

    Pruning candidates are restricted to files matching
    :data:`_TIMESTAMPED_BACKUP_RE`. Order is determined by parsing the
    timestamp out of the filename — never the filesystem mtime, which is
    trampled by ``shutil.move``.

    Legacy ``memory.db.corrupt.bak`` and ``memory.db.corrupt.bak.1`` files
    count against ``keep_n`` (so total disk usage stays bounded across an
    upgrade) but are NEVER selected for deletion.
    """
    timestamped: list[tuple[str, Path]] = []
    legacy_count = 0
    try:
        children = list(parent.iterdir())
    except OSError:
        return
    for child in children:
        name = child.name
        if name in _LEGACY_CORRUPT_NAMES:
            legacy_count += 1
            continue
        match = _TIMESTAMPED_BACKUP_RE.fullmatch(name)
        if match is not None:
            timestamped.append((match.group(1), child))
    timestamped.sort(key=lambda pair: pair[0])
    excess = len(timestamped) + legacy_count - keep_n
    pruned = 0
    while excess > 0 and timestamped:
        _, victim = timestamped.pop(0)
        with contextlib.suppress(OSError):
            victim.unlink()
            pruned += 1
        excess -= 1
    if excess > 0:
        logger.warning(
            "corrupt_backup_budget_exceeded_legacy_only",
            keep=keep_n,
            legacy_count=legacy_count,
        )
    logger.info(
        "corrupt_backup_rotated",
        parent=str(parent),
        keep=keep_n,
        total_timestamped=len(timestamped),
        total_legacy=legacy_count,
        pruned=pruned,
    )
