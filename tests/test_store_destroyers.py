"""The live-store destroyers run under an EXCLUSIVE store op (PRD-CORE-306 slice 2).

Two properties per destroyer: another process's open refuses it with nothing
changed, and a process killed between any two of its steps loses no committed
row -- the store opens with every row, or the original is intact (for a
recovery: at its rotated name, WAL included).
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory._store_lock import store_access
from trw_memory.cli_storage import handle_restore
from trw_memory.exceptions import CorruptDatabaseUnsalvageableError, StoreBusyError
from trw_memory.models.config import MemoryConfig
from trw_memory.storage._snapshot import restore_from_snapshot, take_daily_snapshot
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_cold_rebuild_support import _corrupt_sqlite_master, _make_yaml, _populate_real_db
from .test_db_recovery import _make_entry
from .test_store_lock import Probe, short_waits  # noqa: F401 - a fixture

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX advisory locks")

_ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

#: Leaves committed rows in the store's WAL only: the writer dies without the checkpoint a close would run.
_WAL_WRITER = r"""
import os, sys
from pathlib import Path
from datetime import datetime, timezone
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
backend = SQLiteBackend(Path(sys.argv[1]))
now = datetime.now(timezone.utc)
for i in sys.argv[2:]:
    backend.store(MemoryEntry(id=i, content=f"wal row {i}", namespace="default", created_at=now, updated_at=now))
os._exit(0)
"""

#: Runs one destroyer and SIGKILLs itself right after the named step. A restore's steps: staged (the
#: copy is closed when the checkpoint is called), checkpointed, replaced, sidecars cleared (the closing log line).
_KILLER = r"""
import os, pathlib, signal, sys
from pathlib import Path
from trw_memory.storage import _recovery, _snapshot
from trw_memory.storage.sqlite_backend import SQLiteBackend
op, step, db, base, snap = sys.argv[1:6]

def after(owner, name, wrap=lambda f: f):
    original = getattr(owner, name)
    def killed(*args, **kwargs):
        original(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGKILL)
    setattr(owner, name, wrap(killed))

if op == "recover":
    owner = SQLiteBackend if step.startswith("_") and hasattr(SQLiteBackend, step) else _recovery
    after(owner, step, staticmethod if owner is SQLiteBackend else (lambda f: f))
    SQLiteBackend.recover_db(Path(db))
elif step == "staged":
    _snapshot._checkpoint = lambda *a: os.kill(os.getpid(), signal.SIGKILL)
elif step == "cleared":
    class Dead:
        def info(self, *a, **k):
            os.kill(os.getpid(), signal.SIGKILL)
    _snapshot.logger = Dead()
else:
    after(pathlib.Path if step == "replace" else _snapshot, step)
if op == "restore":
    _snapshot.restore_from_snapshot(Path(base), Path(snap), Path(db))
os._exit(3)  # the step never ran
"""


def _run(script: str, *args: str) -> int:
    return subprocess.run([sys.executable, "-c", script, *args], env=_ENV, check=False, timeout=120).returncode


def _ids(db: Path) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
        return {row[0] for row in conn.execute("SELECT id FROM memories")}
    finally:
        conn.close()


def _seed(db: Path, rows: list[str], wal_rows: list[str]) -> None:
    backend = SQLiteBackend(db)
    for entry_id in rows:
        backend.store(_make_entry(entry_id))
    backend.close()
    if not wal_rows:
        return
    assert _run(_WAL_WRITER, str(db), *wal_rows) == 0
    assert Path(f"{db}-wal").stat().st_size > 0, "the WAL rows must still be in the WAL"


def _snapshot_of(base: Path, db: Path) -> Path:
    return take_daily_snapshot(base, db, keep_daily=7, now=datetime(2026, 9, 26, tzinfo=timezone.utc))


def _files(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file() and not p.name.endswith(".oplock")}


# --- another process's open refuses every destroyer, nothing changed ------------------------


@pytest.mark.usefixtures("short_waits")
@pytest.mark.parametrize("destroyer", ["recover", "restore_snapshot", "cli_snapshot", "cli_cold"])
def test_an_open_in_another_process_refuses_the_destroyer(tmp_path: Path, destroyer: str) -> None:
    db = tmp_path / "memory.db"
    _seed(db, ["a", "b"], ["c"])
    snap = _snapshot_of(tmp_path, db)
    probe = Probe(db)
    try:
        assert probe.ask("take serve" if destroyer.startswith("cli") else "open") == ["ok"]
        before = _files(tmp_path)
        if destroyer.startswith("cli"):
            args = Namespace(
                db=str(db), namespace="default", from_snapshot=snap.name if destroyer == "cli_snapshot" else None
            )
            args.from_cold = destroyer == "cli_cold"
            assert handle_restore(args, config_cls=MemoryConfig) == 1
        else:
            with pytest.raises(StoreBusyError):
                if destroyer == "recover":
                    SQLiteBackend.recover_db(db)
                else:
                    restore_from_snapshot(tmp_path, snap, db)
        assert _files(tmp_path) == before
    finally:
        probe.close()
    assert _ids(db) == {"a", "b", "c"}


# --- a kill between any two steps loses no committed row ------------------------------------


@pytest.mark.parametrize(
    "step",
    ["_rotate_corrupt_backup", "_prune_corrupt_backups", "write_sentinel", "_attempt_primary_salvage", "_restore_rows"],
)
def test_a_recovery_killed_after_any_step_keeps_every_row(tmp_path: Path, step: str) -> None:
    db = tmp_path / "memory.db"
    _seed(db, ["a", "b"], ["w1", "w2"])

    assert _run(_KILLER, "recover", step, str(db), "", "") == -signal.SIGKILL

    rotated = list(tmp_path.glob("memory.db.corrupt.*.bak"))
    assert len(rotated) == 1
    committed = {"a", "b", "w1", "w2"}
    assert _ids(rotated[0]) == committed, "the original, WAL included, is intact at its rotated name"
    if step == "_restore_rows":
        assert _ids(db) == committed
    with store_access(db, "recover"):  # the dead holder's lock went with it
        pass


@pytest.mark.parametrize("step", ["staged", "_checkpoint", "replace", "cleared"])
def test_a_snapshot_restore_killed_after_any_step_keeps_every_row(tmp_path: Path, step: str) -> None:
    db = tmp_path / "memory.db"
    _seed(db, ["a", "b"], [])
    snap = _snapshot_of(tmp_path, db)
    _seed(db, [], ["w1", "w2"])  # the live store moved on: two rows only in its WAL

    assert _run(_KILLER, "restore", step, str(db), str(tmp_path), str(snap)) == -signal.SIGKILL

    replaced = step in ("replace", "cleared")
    assert _ids(db) == ({"a", "b"} if replaced else {"a", "b", "w1", "w2"})
    with store_access(db, "restore"):
        pass


# --- B71-133 (a): the next open resumes a recovery killed after any step ---------------------


@pytest.mark.parametrize(
    "step",
    [
        "write_recovery_marker",  # before the rotation: nothing moved, the marker is dropped
        "_rotate_corrupt_backup",
        "_prune_corrupt_backups",
        "write_sentinel",
        "_attempt_primary_salvage",
        "_restore_rows",
    ],
)
def test_the_next_open_after_a_recovery_killed_after_any_step_holds_every_row(tmp_path: Path, step: str) -> None:
    """Killed between the rotation and the restored rows, recovery used to leave the next open an EMPTY store,
    every row stranded in the rotated backup. The next open now resumes the salvage from it."""
    db = tmp_path / "memory.db"
    _seed(db, ["a", "b"], ["w1", "w2"])

    assert _run(_KILLER, "recover", step, str(db), "", "") == -signal.SIGKILL

    SQLiteBackend(db).close()
    assert _ids(db) == {"a", "b", "w1", "w2"}
    assert not Path(f"{db}.recovering").exists()


#: SIGKILLs a strict recovery right after its cold rebuild committed, before the recovery marker is cleared.
_COLD_REBUILD_KILLER = r"""
import os, signal, sys
from pathlib import Path
from trw_memory.storage import _cold_rebuild
from trw_memory.storage.sqlite_backend import SQLiteBackend
rebuild = _cold_rebuild.rebuild_from_cold

def killed(*args, **kwargs):
    rebuild(*args, **kwargs)
    os.kill(os.getpid(), signal.SIGKILL)

_cold_rebuild.rebuild_from_cold = killed
SQLiteBackend.recover_db(Path(sys.argv[1]), recovery_policy="strict", rebuild_from_cold=True)
os._exit(3)  # the rebuild never ran
"""


@pytest.mark.parametrize("rows_after_the_kill", [False, True])
def test_a_resumed_recovery_never_deletes_the_store_it_resumes_into(tmp_path: Path, rows_after_the_kill: bool) -> None:
    """sol r1 P1-a: the resume salvaged the unsalvageable backup again (0 rows), re-ran the cold rebuild (0 new:
    every id was already there) and, as a strict refusal, unlinked the store at the path: the rebuilt rows and
    any row written after the kill with it."""
    db = tmp_path / "memory.db"
    _populate_real_db(db, entries=1)
    _corrupt_sqlite_master(db)
    committed = {f"L-COLD{i}" for i in range(3)}
    for entry_id in sorted(committed):
        _make_yaml(tmp_path, entry_id)

    assert _run(_COLD_REBUILD_KILLER, str(db)) == -signal.SIGKILL
    assert _ids(db) == committed, "the rebuild committed before the kill"
    if rows_after_the_kill:
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TEMP TABLE one AS SELECT * FROM memories LIMIT 1; UPDATE one SET id = 'L-AFTER';"
            "INSERT INTO memories SELECT * FROM one;"
        )
        conn.close()
        committed.add("L-AFTER")

    SQLiteBackend(db, recovery_policy="strict", rebuild_from_cold=True).close()
    assert _ids(db) == committed


def test_a_resumed_recovery_refused_strictly_leaves_the_store_the_backup_and_the_marker(tmp_path: Path) -> None:
    """Nothing to salvage and nothing in the cold tier: the resume still refuses, but removes nothing it did not
    create -- the (empty) store at the path, the rotated backup and the marker all stay, and the verdict is hard_fail."""
    db = tmp_path / "memory.db"
    _populate_real_db(db, entries=1)
    _corrupt_sqlite_master(db)

    assert _run(_COLD_REBUILD_KILLER, str(db)) == -signal.SIGKILL  # the rebuild found no cold rows
    assert db.exists()

    with pytest.raises(CorruptDatabaseUnsalvageableError):
        SQLiteBackend(db, recovery_policy="strict", rebuild_from_cold=True)
    assert db.exists()
    assert len(list(tmp_path.glob("memory.db.corrupt.*.bak"))) == 1
    assert Path(f"{db}.recovering").exists()
    assert json.loads(Path(f"{db}.recovery.json").read_text())["status"] == "hard_fail"
