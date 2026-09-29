"""The store-operation protocol (PRD-CORE-306 slice 1, ``docs/sprint-mcp7/DESIGN-B71-00-store-protocol.md`` §6 tests 1-7).

A lock is only visible from another process, so each test drives one
long-lived probe child over a pipe. Every step waits for the child's reply;
nothing is synchronized by sleeping.
"""

from __future__ import annotations

import asyncio
import errno
import gc
import os
import select
import signal
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from trw_memory import _live_stores, _store_lock
from trw_memory._dir_trust import open_component_fd
from trw_memory._live_stores import connect_registered
from trw_memory._store_lock import StoreOp, store_access
from trw_memory.exceptions import StoreBusyError, UnsupportedStorageError, UntrustedDirectoryError

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX advisory locks")

_PROBE = r"""
import fcntl, gc, os, sqlite3, sys, time
from pathlib import Path
from trw_memory import _store_lock as sl
from trw_memory._live_stores import connect_registered
from trw_memory.exceptions import StorageError
db = sys.argv[1]
holds, conns, keep, kid = [], [], [], None
# Only for test_a_slow_release_after_a_locked_check_is_not_raced_by_the_next_exclusive:
# how long this probe sits on its own "locked" check's lock before releasing it, to simulate
# scheduling delay under host load without depending on real load to land the race.
_locked_release_delay = float(os.environ.get("TRW_TEST_LOCKED_RELEASE_DELAY", "0"))

def reply(*words):
    print(*words, flush=True)

def can_write():
    conn = sqlite3.connect(db, timeout=0, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
        return "granted"
    except sqlite3.OperationalError:
        return "blocked"

for line in sys.stdin:
    cmd, *args = line.split()
    try:
        if cmd == "open":
            conns.append(connect_registered(db, sqlite3, db))
        elif cmd == "take":
            holds.append(sl.acquire(db, args[0]))
        elif cmd == "backend":
            from trw_memory.storage.sqlite_backend import SQLiteBackend
            keep.append(SQLiteBackend(Path(db)))
        elif cmd == "client":
            from trw_memory.client import MemoryClient
            keep.append(MemoryClient("project:probe-00000000", mode="local", db_path=db))
        elif cmd == "release":
            for hold in holds:
                sl.release(hold)
            for conn in conns:
                conn.close()
            holds.clear(); conns.clear()
        elif cmd == "can_write":
            reply(can_write()); continue
        elif cmd == "locked":  # only while this probe holds nothing: its own close would drop its locks
            fd = os.open(db + ".oplock", os.O_RDWR)
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Release BEFORE replying "free": a reply sent first would tell the parent nothing
                # holds the lock while this probe's own fd (about to be closed) still does, which
                # races a caller that immediately takes an exclusive op with no wait budget.
                if _locked_release_delay:
                    time.sleep(_locked_release_delay)
                os.close(fd)
                reply("free")
            except OSError:
                os.close(fd)
                reply("held")
            continue
        elif cmd == "fork_open":
            ready_r, ready_w = os.pipe()
            done_r, done_w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(done_w)  # so the parent's close is this read's end of file
                conn = connect_registered(db, sqlite3, db)
                os.write(ready_w, b"x")
                os.read(done_r, 1)
                conn.close()
                os._exit(0)
            os.read(ready_r, 1)
            os.close(done_r)
            kid = (pid, done_w)
            reply("ok", pid); continue
        elif cmd == "fork_use":  # fork_use <journal mode> <idle|txn> <use|quiet>: a child and what it inherited
            conn = connect_registered(db, sqlite3, db, isolation_level=None)
            conn.execute(f"PRAGMA journal_mode={args[0]}")
            conn.execute("CREATE TABLE IF NOT EXISTS t(x)")
            cur, ex = conn.cursor(), conn.execute  # a cursor and a bound method, both made before the fork
            if args[1] == "txn":
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany("INSERT INTO t VALUES (?)", [("parent" * 200,)] * 200)
            pid = os.fork()
            if pid == 0:
                insert = "INSERT INTO t VALUES ('child')"
                uses = (lambda: conn.execute(insert), lambda: cur.execute(insert), lambda: ex(insert), lambda: conn.commit())
                refused = 0
                for use in uses if args[2] == "use" else ():
                    try:
                        use()
                    except Exception as exc:
                        refused += isinstance(exc, (sqlite3.ProgrammingError, StorageError))
                os._exit(refused)
            code = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
            wal = "wal" if os.path.exists(db + "-wal") else "no-wal"  # still there after the child's close
            if args[1] == "txn":
                conn.execute("COMMIT")
            conn.execute("INSERT INTO t VALUES ('after')")
            check = conn.execute("PRAGMA integrity_check").fetchone()[0]
            counts = conn.execute("SELECT count(*) FILTER (WHERE x = 'child'), count(*) FROM t").fetchone()
            conn.close()
            reply(code, wal, check, *counts)
            continue
        elif cmd == "fork_end":
            os.close(kid[1]); os.waitpid(kid[0], 0)
        elif cmd == "drop_last":
            # The process's last action: a connection only the collector frees, collected while
            # the registry mutex is held, so its release is queued and must still be applied.
            conn = connect_registered(db, sqlite3, db)
            cycle = [conn]; cycle.append(cycle); del conn, cycle
            with sl._registry():
                gc.collect()
        reply("ok")
    except sl.StoreBusyError:
        reply("busy")
    except Exception as exc:
        reply("error", type(exc).__name__, str(exc).replace(chr(10), " "))
"""


class Probe:
    """A child process that holds and tests locks on one store, one command per reply."""

    def __init__(self, db: Path, *, env: dict[str, str] | None = None) -> None:
        full_env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), **(env or {})}
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _PROBE, str(db)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=full_env,
        )

    @property
    def pid(self) -> int:
        return self.proc.pid

    def send(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def answer(self) -> list[str]:
        assert self.proc.stdout is not None
        return self.proc.stdout.readline().split()

    def ask(self, command: str) -> list[str]:
        self.send(command)
        return self.answer()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


@pytest.fixture
def probe(db: Path) -> Iterator[Probe]:
    child = Probe(db)
    yield child
    child.close()


@pytest.fixture
def short_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every op waits at most a fraction of a second, so a refusal is quick to observe."""
    for op, wait in list(_store_lock.WAITS.items()):
        monkeypatch.setitem(_store_lock.WAITS, op, min(wait, 0.3))


def _holds_nothing(db: Path) -> bool:
    return not any(name.startswith(os.path.realpath(db.parent)) for name in _store_lock._FILES)


def _refusal(db: Path, op: StoreOp = "restore") -> StoreBusyError:
    with pytest.raises(StoreBusyError) as caught, store_access(db, op):
        pass
    return caught.value


# --- 1. idle holders refuse EXCLUSIVE, and a killed holder frees it -------------------------


@pytest.mark.parametrize("holder", ["backend", "client", "take serve"])
def test_an_idle_holder_refuses_exclusive_until_it_is_killed(db: Path, probe: Probe, holder: str) -> None:
    assert probe.ask(holder) == ["ok"]

    refused = _refusal(db)
    assert "in use by another process" in str(refused)
    assert "Nothing was changed" in str(refused)
    assert refused.path == os.path.realpath(db)

    os.kill(probe.pid, signal.SIGKILL)
    probe.proc.wait()
    with store_access(db, "restore"):
        pass


# --- 2. an OPEN elsewhere waits for EXCLUSIVE, then proceeds --------------------------------


def test_an_open_in_another_process_waits_for_exclusive(db: Path, probe: Probe) -> None:
    with store_access(db, "restore"):
        probe.send("open")
        # The one timed check: the probe's reply must not come while the store is held.
        assert select.select([probe.proc.stdout], [], [], 0.5)[0] == []
    assert probe.answer() == ["ok"]
    assert "another process" in str(_refusal(db))


# --- 3. C15 survival: nothing this protocol does drops SQLite's locks or its own ------------


@pytest.fixture
def writing(db: Path) -> Iterator[sqlite3.Connection]:
    conn = connect_registered(db, sqlite3, str(db), isolation_level=None)
    conn.execute("CREATE TABLE IF NOT EXISTS t(x)")
    conn.execute("BEGIN IMMEDIATE")
    yield conn
    conn.execute("ROLLBACK")
    conn.close()


def _still_held(probe: Probe) -> None:
    assert probe.ask("can_write") == ["blocked"], "SQLite's write lock was dropped"
    assert probe.ask("locked") == ["held"], "the store's lock file lock was dropped"


@pytest.mark.usefixtures("writing")
def test_shared_cycles_on_other_threads_keep_every_lock(db: Path, probe: Probe) -> None:
    def cycle() -> None:
        for _ in range(20):
            connect_registered(db, sqlite3, str(db)).close()

    threads = [threading.Thread(target=cycle) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _still_held(probe)


@pytest.mark.usefixtures("writing")
def test_a_non_quiescing_exclusive_and_its_downgrade_keep_every_lock(db: Path, probe: Probe) -> None:
    during: list[list[str]] = []

    def import_op() -> None:
        with store_access(db, "import"):
            during.append(probe.ask("locked"))

    helper = threading.Thread(target=import_op)
    helper.start()
    helper.join()
    assert during == [["held"]]
    _still_held(probe)


def test_a_reader_landing_on_the_lock_file_is_refused_with_only_serve_held(
    db: Path, probe: Probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No connection anywhere in this process (another test's may still await collection): the lock
    # file's registration alone must protect it.
    monkeypatch.setattr(_live_stores, "_OPEN", {})
    monkeypatch.setattr(_live_stores, "_LIVE_CACHE", None)
    hold = _store_lock.acquire(db, "serve")
    try:
        dir_fd = os.open(db.parent, os.O_RDONLY)
        try:
            with pytest.raises(UntrustedDirectoryError):
                open_component_fd(dir_fd, f"{db.name}.oplock", directory=False)
            fd = os.open(f"{db}.oplock", os.O_RDONLY)
            assert _live_stores.admit_reader_fd(fd) is False  # parked, never closed
        finally:
            os.close(dir_fd)
        assert probe.ask("locked") == ["held"]
    finally:
        _store_lock.release(hold)


# --- 4. fork: the child's inherited token gives it no lock; its own open takes one ---------


def test_a_forked_child_takes_its_own_lock(db: Path, probe: Probe) -> None:
    assert probe.ask("open") == ["ok"]
    reply = probe.ask("fork_open")
    assert reply[0] == "ok"
    assert probe.ask("release") == ["ok"]

    assert probe.ask("take restore") == ["busy"]  # the child's own open holds it
    assert probe.ask("fork_end") == ["ok"]
    assert probe.ask("take restore") == ["ok"]


def test_a_lock_file_replaced_while_a_first_pin_waited_is_refused(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """sol B71-00 s1 r2: the waiter took the entry another thread admitted without the identity check, so a
    lock file replaced in between left it pinned to an inode that no longer guards the store's name."""
    lock_file = f"{db}.oplock"
    db.touch()
    reader = os.open(lock_file, os.O_RDONLY | os.O_CREAT, 0o600)
    assert _live_stores.admit_reader_fd(reader)
    admitted: list[object] = []

    def _another_thread_admits_then_the_file_is_replaced(_timeout: float) -> bool:
        _live_stores.close_reader_fd(reader)
        admitted.append(_store_lock._pin(os.path.realpath(db), read_only=False))
        replacement = f"{lock_file}.new"
        os.close(os.open(replacement, os.O_WRONLY | os.O_CREAT, 0o600))
        os.replace(replacement, lock_file)
        return True

    monkeypatch.setattr(_live_stores._LEASE_RELEASED, "wait", _another_thread_admits_then_the_file_is_replaced)
    with pytest.raises(UnsupportedStorageError, match="was replaced while this process held it"):
        _store_lock._pin(os.path.realpath(db), read_only=False)
    _store_lock._unpin(admitted[0])  # type: ignore[arg-type]


_FORK_WHILE_HELD = """
import os, sys, threading, time
from trw_memory import _live_stores
held = threading.Event()
def hold():
    with _live_stores.FD_LOCK:
        held.set()
        time.sleep(0.5)
threading.Thread(target=hold).start()
held.wait()
pid = os.fork()
if pid == 0:
    with _live_stores._locked():
        os._exit(0)
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    if os.waitpid(pid, os.WNOHANG)[0]:
        print("ok")
        sys.exit(0)
    time.sleep(0.05)
os.kill(pid, 9)
print("hung")
"""


def test_a_child_forked_while_another_thread_holds_the_descriptor_lock_can_take_it() -> None:
    """sol B71-00 s1: the child inherited FD_LOCK held by a thread that does not exist in it, and hung."""
    run = subprocess.run([sys.executable, "-c", _FORK_WHILE_HELD], capture_output=True, text=True, timeout=30)
    assert run.stdout.strip() == "ok", run.stderr


# --- 5. helpers of an EXCLUSIVE owner: copied context re-enters, a plain pool does not ------


@pytest.mark.usefixtures("short_waits")
def test_a_helper_thread_under_exclusive(db: Path) -> None:
    def open_close() -> None:
        connect_registered(db, sqlite3, str(db)).close()

    async def owner() -> None:
        with store_access(db, "restore"):
            await asyncio.to_thread(open_close)  # the copied context owns the hold: re-entry
            with ThreadPoolExecutor(1) as pool, pytest.raises(StoreBusyError, match="restore"):
                pool.submit(open_close).result()  # a plain pool is another requester: refused

    asyncio.run(owner())
    open_close()  # released: anyone may open again


def test_a_plain_pool_open_waits_for_a_quiescing_exclusive(db: Path) -> None:
    entered, finish = threading.Event(), threading.Event()

    def restore() -> None:
        with store_access(db, "restore"):
            entered.set()
            finish.wait()

    owner = threading.Thread(target=restore)
    owner.start()
    entered.wait()
    with ThreadPoolExecutor(1) as pool:
        opened = pool.submit(lambda: connect_registered(db, sqlite3, str(db)).close())
        assert not opened.done()
        finish.set()
        opened.result()
    owner.join()


def test_a_context_holding_shared_cannot_upgrade(db: Path) -> None:
    conn = connect_registered(db, sqlite3, str(db))
    try:
        with pytest.raises(RuntimeError, match="already holds"), store_access(db, "restore"):
            pass
    finally:
        conn.close()


def test_serve_is_the_process_hold_not_the_context_one(db: Path, probe: Probe) -> None:
    """The daemon copies its context into every request; SERVE there must not read as the request's own."""
    serving = _store_lock.acquire(db, "serve")
    try:
        with store_access(db, "restore"):  # no "already holds": SERVE confers no ownership
            assert probe.ask("locked") == ["held"]
        assert probe.ask("take restore") == ["busy"]  # still the process's SHARED hold
    finally:
        _store_lock.release(serving)


def test_an_exclusive_op_is_one_scope_and_does_not_nest(db: Path, probe: Probe) -> None:
    with store_access(db, "snapshot"), pytest.raises(RuntimeError, match="already holds"), store_access(db, "snapshot"):
        pass
    assert probe.ask("locked") == ["free"]
    assert _holds_nothing(db)


# --- 6. finalizer releases: queued under the mutex, and never stranded ---------------------


def test_a_finalizer_under_the_registry_mutex_is_queued_then_applied(db: Path, probe: Probe) -> None:
    conn = connect_registered(db, sqlite3, str(db))
    cycle: list[object] = [conn]
    cycle.append(cycle)
    del conn, cycle
    with _store_lock._registry():
        gc.collect()  # the finalizer cannot take the mutex this thread holds
        assert _store_lock._QUEUE
    assert not _store_lock._QUEUE
    assert probe.ask("locked") == ["free"]
    with store_access(db, "restore"):  # the next operation closes the idle descriptor
        pass
    assert _holds_nothing(db)


def test_a_slow_release_after_a_locked_check_is_not_raced_by_the_next_exclusive(db: Path) -> None:
    """The probe's own "locked" check takes an exclusive fcntl lock to answer "free"; if it reported
    "free" before actually releasing that lock, a scheduling delay in the probe process (real under
    host contention) would race the very next exclusive op here, which has no wait budget for
    "restore". Reproduced deterministically (no host load needed) pre-fix by making the probe sleep
    between taking the lock and releasing it, while still replying only after release."""
    connect_registered(db, sqlite3, str(db)).close()  # creates the lock file the probe's "locked" opens
    slow_probe = Probe(db, env={"TRW_TEST_LOCKED_RELEASE_DELAY": "0.3"})
    try:
        assert slow_probe.ask("locked") == ["free"]  # only sent once release has already happened
        with store_access(db, "restore"):  # must not raise StoreBusyError: nothing else holds the lock
            pass
    finally:
        slow_probe.close()


def test_a_finalizer_inside_a_descriptor_scan_frees_the_lock_but_leaves_the_table(db: Path, probe: Probe) -> None:
    """int rc7 C2's rule, for the store hold too: a finalizer may run inside a scan holding FD_LOCK (an RLock),
    so it releases the kernel lock but leaves the descriptor table to the next store operation."""
    conn = connect_registered(db, sqlite3, str(db))
    cycle: list[object] = [conn]
    cycle.append(cycle)
    del conn, cycle
    with _live_stores._locked():
        files, lock_files = dict(_store_lock._FILES), set(_live_stores._LOCK_FILES)
        gc.collect()
        assert _store_lock._FILES == files and _live_stores._LOCK_FILES == lock_files, "mutated mid-scan"
        assert probe.ask("locked") == ["free"]  # yet no other process is kept out
    with store_access(db, "restore"):
        pass
    assert _holds_nothing(db)


def test_a_collection_as_the_last_action_frees_the_store(db: Path, probe: Probe) -> None:
    assert probe.ask("drop_last") == ["ok"]
    with store_access(db, "restore"):  # no further call in the probe
        pass


# --- 7. lock-file admission --------------------------------------------------------------------


def test_two_first_openers_across_a_lease_wait_share_one_descriptor(db: Path) -> None:
    """sol B71-00 s1: the lease wait releases FD_LOCK, so two first openers both opened the lock file and the
    second overwrote the first's entry -- closing either descriptor then dropped the other's lock."""
    lock_file = f"{db}.oplock"
    db.touch()
    reader = os.open(lock_file, os.O_RDONLY | os.O_CREAT, 0o600)
    assert _live_stores.admit_reader_fd(reader)  # a reader's lease holds both openers in the wait
    with ThreadPoolExecutor(2) as pool:
        pins = [pool.submit(_store_lock._pin, os.path.realpath(db), read_only=False) for _ in range(2)]
        threading.Event().wait(0.3)
        _live_stores.close_reader_fd(reader)
        first, second = (pin.result(timeout=10) for pin in pins)
    assert first is second and first is not None and first.users == 2
    _store_lock._unpin(first)
    _store_lock._unpin(first)
    assert _holds_nothing(db)


def _held_by_sqlite(db: Path) -> sqlite3.Connection:
    conn = connect_registered(db, sqlite3, str(db), isolation_level=None, store_lock=False)
    conn.execute("CREATE TABLE IF NOT EXISTS t(x)")
    conn.execute("BEGIN IMMEDIATE")
    return conn


def test_a_hard_link_to_the_store_is_refused_without_dropping_its_locks(db: Path, probe: Probe) -> None:
    conn = _held_by_sqlite(db)
    try:
        os.link(db, f"{db}.oplock")
        with pytest.raises(UnsupportedStorageError, match="more than one name"):
            connect_registered(db, sqlite3, str(db))
        assert probe.ask("can_write") == ["blocked"]
    finally:
        conn.execute("ROLLBACK")
        conn.close()


@pytest.mark.parametrize(
    ("plant", "message"),
    [
        (lambda lock: lock.symlink_to(lock.with_name("elsewhere")), "cannot be used"),
        (lambda lock: os.mkfifo(lock), "not a regular file"),
        pytest.param(
            lambda lock: (lock.touch(), lock.chmod(0)),
            "left by sudo",
            marks=pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the mode bits that force the refusal"),
        ),
    ],
    ids=["symlink", "fifo", "not-yours"],
)
def test_a_planted_lock_file_is_refused(db: Path, plant: object, message: str) -> None:
    plant(Path(f"{db}.oplock"))  # type: ignore[operator]
    with pytest.raises(UnsupportedStorageError, match=message):
        connect_registered(db, sqlite3, str(db))
    assert _holds_nothing(db)


def test_a_lock_file_owned_by_another_user_is_refused(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Path(f"{db}.oplock").touch()
    monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)
    with pytest.raises(UnsupportedStorageError, match="owned by uid"):
        connect_registered(db, sqlite3, str(db))


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the read-only mode that forces the refusal")
def test_a_read_only_directory_is_refused_by_name(db: Path) -> None:
    db.parent.chmod(0o500)
    try:
        with pytest.raises(UnsupportedStorageError, match="not writable"):
            _store_lock.acquire(db, "open")
    finally:
        db.parent.chmod(0o700)


def test_a_lock_file_replaced_while_held_refuses_further_holds(db: Path) -> None:
    conn = connect_registered(db, sqlite3, str(db))
    try:
        replacement = db.with_name("swap")
        replacement.touch()
        os.replace(replacement, f"{db}.oplock")
        with pytest.raises(UnsupportedStorageError, match="replaced while this process held it"):
            connect_registered(db, sqlite3, str(db))
    finally:
        conn.close()


def test_a_filesystem_without_locks_fails_closed(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_locks(*_args: object) -> None:
        raise OSError(errno.ENOLCK, "no locks")

    monkeypatch.setattr(_store_lock.fcntl, "lockf", no_locks)
    with pytest.raises(UnsupportedStorageError, match="advisory locks"):
        connect_registered(db, sqlite3, str(db))
    monkeypatch.undo()
    assert _holds_nothing(db)


# --- 8. one store, one lock (sol PRD-CORE-306 review P1s) ----------------------------------------


@pytest.mark.usefixtures("short_waits")
def test_an_exclusive_through_one_name_keeps_out_an_open_through_another(db: Path, probe: Probe) -> None:
    """P1 #1: the lock was keyed by name, so a second name for the store got its own lock file and walked past."""
    sqlite3.connect(db).close()
    symlink, hard_link = db.with_name("symlink.db"), db.with_name("hardlink.db")
    symlink.symlink_to(db)
    assert probe.ask("take restore") == ["ok"]
    with pytest.raises(StoreBusyError, match="another process"):
        connect_registered(symlink, sqlite3, str(symlink))  # resolved: the same lock file
    os.link(db, hard_link)
    for name in (db, hard_link):
        with pytest.raises(UnsupportedStorageError, match="more than one name"):
            connect_registered(name, sqlite3, str(name))
    assert probe.ask("release") == ["ok"]
    with pytest.raises(UnsupportedStorageError, match="more than one name"):
        _refusal(db)  # an exclusive op is refused on a linked store too
    assert _holds_nothing(db)


def test_a_hard_link_made_between_the_lock_and_the_connect_is_refused(tmp_path: Path) -> None:
    """P1 #1: the name check before the lock cannot see a link made before the connect; the one after it does."""
    existing, fresh = tmp_path / "existing.db", tmp_path / "fresh.db"
    sqlite3.connect(existing).close()

    class _LinkingDriver:
        Connection = sqlite3.Connection

        @staticmethod
        def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            os.link(existing, fresh)
            return sqlite3.connect(*args, **kwargs)  # type: ignore[arg-type]

    with pytest.raises(UnsupportedStorageError, match="more than one name"):
        connect_registered(fresh, _LinkingDriver, str(fresh))
    assert _holds_nothing(fresh)
    assert all(identity[1] != os.stat(existing).st_ino for identity in _live_stores._OPEN)


@pytest.mark.parametrize(
    ("mode", "state", "expected"),
    [
        ("wal", "idle", ["4", "wal", "ok", "0", "1"]),
        ("wal", "txn", ["4", "wal", "ok", "0", "201"]),
        ("delete", "idle", ["4", "no-wal", "ok", "0", "1"]),
    ],
)
def test_a_forked_child_cannot_use_an_inherited_connection(
    probe: Probe, mode: str, state: str, expected: list[str]
) -> None:
    """P1 #2 (sol rounds 1-2): the child keeps the parent's connection but not its lock. It closes it, so the
    connection, a cursor and a bound method made before the fork all fail closed, and the parent's store (its WAL,
    and a transaction open across the fork) comes through intact."""
    assert probe.ask(f"fork_use {mode} {state} use") == expected
    assert probe.ask("fork_open")[0] == "ok"  # a connection of its own still works
    assert probe.ask("fork_end") == ["ok"]


def test_a_rollback_journal_write_open_across_a_fork_is_not_rolled_back_by_the_child(probe: Probe) -> None:
    """Closing that connection in the child would play the journal back over the parent's write and delete it."""
    assert probe.ask("fork_use delete txn quiet") == ["0", "no-wal", "ok", "0", "201"]
