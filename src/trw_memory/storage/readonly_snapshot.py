"""Private diagnostic snapshots of existing plain SQLite memory stores.

The source is opened read-only with query_only enabled: no schema initialization,
recovery or permission hardening is performed there. Consumers may initialize
the disposable copy. SQLite's backup API includes committed WAL contents.
This is not an encryption adapter or a retained backup/rotation facility.
SQLite may create/update WAL shared-memory sidecars while reading; the guarantee
excludes zero source-directory writes. It protects logical database contents and
does not initialize or chmod the source main file. Never use immutable=1 to hide
live WAL state. timeout_seconds is cooperative, not a hard wall-clock deadline.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic


@contextmanager
def readonly_memory_snapshot(
    source_path: Path, *, temporary_root: Path | None = None, timeout_seconds: float = 30.0
) -> Iterator[Path]:
    """Yield a private consistent copy, deleting it on success or failure.

    Missing, unreadable and uninitialized sources raise rather than becoming
    empty stores. SQLite/OS errors propagate; callers must not report success.
    Directory/file permissions protect the temporary copy's learning content.
    The busy timeout and backup progress checks cannot preempt an individual
    SQLite operation; callers needing a hard deadline must isolate execution.
    """
    if not isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Snapshot timeout_seconds must be positive")
    deadline = monotonic() + timeout_seconds

    def check_deadline(status: int, remaining: int, total: int) -> None:
        if monotonic() >= deadline:
            raise TimeoutError("Read-only snapshot exceeded its time budget")

    uri = source_path.resolve().as_uri() + "?mode=ro"
    with TemporaryDirectory(prefix="trw-memory-audit-", dir=temporary_root) as directory:
        snapshot = Path(directory) / "memory.db"
        with closing(sqlite3.connect(uri, uri=True, timeout=timeout_seconds)) as source:
            source.execute("PRAGMA query_only=ON")
            source.execute("BEGIN")
            table = source.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'").fetchone()
            if table is None:
                raise ValueError("Source is not an initialized memory database; audit did not initialize it")
            snapshot.touch(mode=0o600, exist_ok=False)
            with closing(sqlite3.connect(snapshot)) as destination:
                source.backup(destination, pages=256, progress=check_deadline)
        # Release the live read transaction before any potentially slow analysis.
        yield snapshot
