"""PRD-QUAL-110-FR02: SQLiteBackend hardens the on-disk db to 0600.

A file-backed ``memory.db`` is secret-bearing (learning content, provenance).
The backend chmods it to 0600 on creation, mirroring the trw-mcp pins.json
0600 hardening. In-memory (``:memory:``) backends are unaffected, and a chmod
failure on a non-POSIX platform degrades gracefully (no raise).
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from trw_memory.storage.sqlite_backend import SQLiteBackend

_POSIX_ONLY = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits")


@_POSIX_ONLY
def test_file_backed_db_is_0600(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)
    try:
        assert db_path.exists()
        assert stat.S_IMODE(os.stat(db_path).st_mode) == 0o600
    finally:
        backend.close()


def test_in_memory_db_does_not_raise() -> None:
    """The :memory: path has no on-disk file to chmod and must not raise."""
    backend = SQLiteBackend(Path(":memory:"))
    backend.close()


@_POSIX_ONLY
def test_chmod_failure_is_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A chmod OSError (non-POSIX) does not block backend construction."""
    import trw_memory.storage.sqlite_backend as mod

    real_chmod = os.chmod

    def _selective(path: object, mode: int, *a: object, **k: object) -> None:
        if str(path).endswith("memory.db"):
            raise OSError("not supported")
        real_chmod(path, mode, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(mod.os, "chmod", _selective)
    db_path = tmp_path / "memory.db"
    backend = SQLiteBackend(db_path)  # must not raise
    backend.close()
