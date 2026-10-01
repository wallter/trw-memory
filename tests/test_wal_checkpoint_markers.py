"""A checkpoint stamps the three marker files the doctor's `memory_wal` row reads (UF-MEM-26).

The markers (`<db>.checkpoint-ts`, `.checkpoint-effective-ts`, `.checkpoint-reset-ts`) were written only by trw-mcp helpers
that nothing calls since the daemon migration, so any WAL over the threshold warned "nothing has checkpointed" forever.
The store's own `checkpoint_wal` is now the writer, from the `CheckpointResult` it just produced.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from trw_memory.storage._wal_checkpoint import CheckpointResult, stamp_checkpoint_markers
from trw_memory.storage.sqlite_backend import SQLiteBackend

_ATTEMPT, _EFFECTIVE, _RESET = ".checkpoint-ts", ".checkpoint-effective-ts", ".checkpoint-reset-ts"


def _markers(db: Path) -> set[str]:
    return {suffix for suffix in (_ATTEMPT, _EFFECTIVE, _RESET) if Path(f"{db}{suffix}").is_file()}


def _result(**kw: object) -> CheckpointResult:
    base: dict[str, object] = {"busy": 0, "checkpointed": 5, "log_frames": 5, "mode": "TRUNCATE"}
    base.update(kw)
    return base  # type: ignore[return-value]


def test_a_real_checkpoint_through_the_backend_stamps_the_attempt_marker(tmp_path: Path) -> None:
    from tests.conftest import make_entry

    db = tmp_path / "memory.db"
    backend = SQLiteBackend(db)
    backend.store(make_entry(content="row so the WAL has frames"))
    backend.checkpoint_wal()
    backend.close()

    assert _ATTEMPT in _markers(db)
    assert abs(float(Path(f"{db}{_ATTEMPT}").read_text().strip()) - time.time()) < 60


def test_a_resetting_checkpoint_that_cleared_the_backlog_stamps_all_three(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    stamp_checkpoint_markers(db, _result())
    assert _markers(db) == {_ATTEMPT, _EFFECTIVE, _RESET}


def test_a_passive_checkpoint_never_stamps_the_reset_marker(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    stamp_checkpoint_markers(db, _result(mode="PASSIVE"))
    assert _markers(db) == {_ATTEMPT, _EFFECTIVE}  # reclamation is only ever observed from a reset


def test_a_checkpoint_that_left_a_backlog_stamps_only_the_attempt(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    stamp_checkpoint_markers(db, _result(mode="PASSIVE", checkpointed=2, log_frames=9))
    assert _markers(db) == {_ATTEMPT}


def test_a_busy_or_errored_checkpoint_stamps_nothing(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    stamp_checkpoint_markers(db, _result(busy=1, mode="PASSIVE"))
    stamp_checkpoint_markers(db, _result(busy=1, checkpointed=0, log_frames=0, mode="error"))
    assert _markers(db) == set()  # a failed checkpoint must leave the clocks alone so the next evaluation retries


def test_a_memory_database_has_no_markers(tmp_path: Path) -> None:
    stamp_checkpoint_markers(Path(":memory:"), _result())
    assert not list(Path(".").glob(":memory:*"))


def test_an_unwritable_marker_directory_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "missing" / "memory.db"
    stamp_checkpoint_markers(db, _result())  # the directory does not exist: fail open, never break the checkpoint
    assert _markers(db) == set()
