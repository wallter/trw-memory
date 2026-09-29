"""PRD-CORE-306 B71-11 (worker-1): the ``sqlite_wal_reset_unsafe`` boot warning
must fire once per process, not on every ``SQLiteBackend`` open of an unsafe
engine — a long-lived daemon that reopens stores repeatedly logged it 121
times in one daemon test run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog

from trw_memory.storage.sqlite_backend import SQLiteBackend


def test_wal_reset_warning_logs_once_across_two_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two opens of an unsafe engine must emit the boot warning exactly once."""
    from trw_memory.storage import _dbapi

    monkeypatch.setattr(_dbapi, "wal_reset_safe_version", lambda _version: False)  # type: ignore[attr-defined]

    with structlog.testing.capture_logs() as logs:
        first = SQLiteBackend(tmp_path / "a.db")
        second = SQLiteBackend(tmp_path / "b.db")
    try:
        events = [log for log in logs if log.get("event") == "sqlite_wal_reset_unsafe"]
        assert len(events) == 1, f"expected exactly one boot warning across two opens, got {len(events)}"
    finally:
        first.close()
        second.close()
