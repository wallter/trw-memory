"""Permission hardening must never release a live connection's SQLite locks (C15).

POSIX fcntl locks belong to a (process, inode) pair: ``close()`` on ANY
descriptor for the file drops every lock the process holds on it, SQLite's
included. The 0600 hardening opened and closed the store (and its ``-wal`` /
``-shm``) on every ``SQLiteBackend`` construction and reconnect, so a daemon
that builds a backend per write silently stripped its live connections' locks,
and a second process then wrote the store concurrently (b-tree and index tears).
Each probe below runs in a separate process, the only place the lock is visible.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX advisory-lock semantics")

_WRITE_PROBE = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)
try:
    if sys.argv[2] == "exclusive":
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("CREATE TABLE IF NOT EXISTS intruder(x)")
    conn.execute("COMMIT")
    print("granted")
except sqlite3.OperationalError:
    print("blocked")
"""


def _other_process_can_write(db: Path, mode: str = "immediate") -> bool:
    out = subprocess.run(
        [sys.executable, "-c", _WRITE_PROBE, str(db), mode], capture_output=True, text=True, timeout=30, check=True
    )
    return out.stdout.strip() == "granted"


@pytest.fixture
def _short_busy_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend built while another holds the write lock waits out busy_timeout; keep that short."""
    monkeypatch.setattr("trw_memory.storage._connection._BUSY_TIMEOUT_MS", 200)


@pytest.mark.usefixtures("_short_busy_timeout")
def test_a_second_backend_keeps_the_first_ones_write_lock(tmp_path: Path) -> None:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db = tmp_path / "memory.db"
    holder = SQLiteBackend(db)
    holder._conn.execute("BEGIN IMMEDIATE")
    try:
        assert _other_process_can_write(db) is False
        second = SQLiteBackend(db)  # the daemon's per-request construction
        assert _other_process_can_write(db) is False
        second.close()
    finally:
        holder._conn.execute("ROLLBACK")
        holder.close()


@pytest.mark.usefixtures("_short_busy_timeout")
def test_a_reconnect_keeps_another_connections_write_lock(tmp_path: Path) -> None:
    from trw_memory.storage._stale_handle import reconnect
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db = tmp_path / "memory.db"
    holder = SQLiteBackend(db)
    other = SQLiteBackend(db)
    holder._conn.execute("BEGIN IMMEDIATE")
    try:
        reconnect(other)
        assert other.reconnect_count == 1
        assert _other_process_can_write(db) is False
    finally:
        holder._conn.execute("ROLLBACK")
        holder.close()
        other.close()


def test_an_idle_backend_keeps_its_wal_liveness_lock(tmp_path: Path) -> None:
    """An idle WAL connection holds a shared lock on ``-shm``; while it is held no
    other process may enter exclusive mode and write under it."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db = tmp_path / "memory.db"
    idle = SQLiteBackend(db)
    try:
        idle.count()
        SQLiteBackend(db).close()
        assert _other_process_can_write(db, "exclusive") is False
    finally:
        idle.close()


def test_hardening_a_connected_loose_store_tightens_it_and_keeps_its_lock(tmp_path: Path) -> None:
    from trw_memory.storage._permissions import harden_db_file_mode
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db = tmp_path / "memory.db"
    holder = SQLiteBackend(db)
    holder._conn.execute("BEGIN IMMEDIATE")
    try:
        for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
            path.chmod(0o644)  # loosened after the connection opened
        harden_db_file_mode(db)
        assert _other_process_can_write(db) is False
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in (db, Path(f"{db}-wal"), Path(f"{db}-shm")))
    finally:
        holder._conn.execute("ROLLBACK")
        holder.close()


def test_sqlite_created_sidecars_inherit_the_store_mode_under_a_loose_umask(tmp_path: Path) -> None:
    """SQLite creates ``-wal``/``-shm`` with the store file's permission bits, so a
    0600 store never has world-readable sidecars, whatever the umask."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db = tmp_path / "memory.db"
    previous = os.umask(0o022)
    try:
        backend = SQLiteBackend(db)
        backend.count()
        modes = {p.name: p.stat().st_mode & 0o777 for p in (db, Path(f"{db}-wal"), Path(f"{db}-shm"))}
        backend.close()
    finally:
        os.umask(previous)
    assert modes == {"memory.db": 0o600, "memory.db-wal": 0o600, "memory.db-shm": 0o600}


def test_a_loose_store_is_tightened_before_the_first_connect(tmp_path: Path) -> None:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    db = tmp_path / "memory.db"
    wal, shm = Path(f"{db}-wal"), Path(f"{db}-shm")
    seed = sqlite3.connect(db)  # rollback journal: leaves no sidecars behind
    seed.execute("CREATE TABLE seed(x)")
    seed.commit()
    seed.close()
    for path in (db, wal, shm):  # a store and leftover sidecars an older version left world-readable
        path.touch()
        path.chmod(0o644)

    backend = SQLiteBackend(db)
    try:
        assert {p.name: p.stat().st_mode & 0o777 for p in (db, wal, shm)} == {
            "memory.db": 0o600,
            "memory.db-wal": 0o600,
            "memory.db-shm": 0o600,
        }
    finally:
        backend.close()


def test_the_checkout_opener_refuses_a_live_store_and_its_sidecars(tmp_path: Path) -> None:
    """Every caller-named read goes through ``open_checkout_file_fd``; its close would
    release the live store's locks, so it refuses before opening."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.entry import open_checkout_file_fd

    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o700)
    db = checkout / "memory.db"
    holder = SQLiteBackend(db)
    holder._conn.execute("BEGIN IMMEDIATE")
    try:
        for name in ("memory.db", "memory.db-wal", "memory.db-shm"):
            refused = open_checkout_file_fd(str(checkout.resolve()), str((checkout / name).resolve()), "verify")
            assert isinstance(refused, dict) and "has open" in str(refused["error"]), name
        assert _other_process_can_write(db) is False
        plain = checkout / "notes.txt"
        plain.write_text("x")
        fd = open_checkout_file_fd(str(checkout.resolve()), str(plain.resolve()), "verify")
        assert isinstance(fd, int)
        from trw_memory._live_stores import close_reader_fd

        close_reader_fd(fd)
    finally:
        holder._conn.execute("ROLLBACK")
        holder.close()


def test_the_import_sidecar_probe_never_opens_a_live_sidecar(tmp_path: Path) -> None:
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.checkout_import import _sidecar_present_no_follow

    db = tmp_path / "memory.db"
    holder = SQLiteBackend(db)
    holder._conn.execute("BEGIN IMMEDIATE")
    parent_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        assert _sidecar_present_no_follow(parent_fd, "memory.db-shm") is True
        assert _sidecar_present_no_follow(parent_fd, "memory.db-journal") is False
        assert _other_process_can_write(db) is False
    finally:
        os.close(parent_fd)
        holder._conn.execute("ROLLBACK")
        holder.close()


def test_the_verification_walk_never_opens_a_live_store(tmp_path: Path) -> None:
    """``verify`` reads through its own descriptor walk (SEC-016 round 10); the guard
    sits in the shared ``open_component_fd`` primitive, so it covers that walk too."""
    from trw_memory.lifecycle._checkout_walk import _read_bytes_through_checkout
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    checkout = tmp_path / "checkout"
    (checkout / "store").mkdir(parents=True, mode=0o700)
    db = checkout / "store" / "memory.db"
    holder = SQLiteBackend(db)
    holder._conn.execute("BEGIN IMMEDIATE")
    anchor = os.open(checkout, os.O_RDONLY)
    try:
        for name in ("memory.db", "memory.db-wal", "memory.db-shm"):
            assert _read_bytes_through_checkout(anchor, Path("store") / name) is None, name
        assert _other_process_can_write(db) is False
    finally:
        os.close(anchor)
        holder._conn.execute("ROLLBACK")
        holder.close()


def test_a_hard_link_to_a_live_store_or_its_sidecar_is_refused(tmp_path: Path) -> None:
    """Liveness is by inode: an alias with no sidecars beside it is still the live store."""
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.entry import open_checkout_file_fd

    db = tmp_path / "store" / "memory.db"
    db.parent.mkdir(mode=0o700)
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o700)
    holder = SQLiteBackend(db)
    holder._conn.execute("BEGIN IMMEDIATE")
    try:
        os.link(db, checkout / "alias.bin")
        os.link(f"{db}-shm", checkout / "shm-alias.bin")
        for name in ("alias.bin", "shm-alias.bin"):
            refused = open_checkout_file_fd(str(checkout.resolve()), str((checkout / name).resolve()), "verify")
            assert isinstance(refused, dict) and "has open" in str(refused["error"]), name
        assert _other_process_can_write(db) is False
    finally:
        holder._conn.execute("ROLLBACK")
        holder.close()


def test_every_file_backed_sqlite_connect_registers_its_store() -> None:
    """A connection the registry never counted holds locks the reader guard cannot see:
    no SQLite driver ``.connect(...)`` runs outside ``connect_registered`` itself, apart
    from ``_connection.connect``'s in-memory branch."""
    import ast

    import trw_memory

    src = Path(trw_memory.__file__).parent
    drivers = {"sqlite3", "dbapi", "pysqlite3"}
    unregistered: list[str] = []
    for path in sorted(src.rglob("*.py")):
        if path == src / "_live_stores.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "connect"
            ):
                continue
            receiver = ast.unparse(node.func.value)
            if receiver not in drivers and not receiver.endswith("_dbapi"):
                continue
            in_memory_branch = path.name == "_connection.py" and "file_backed" in ast.unparse(parents.get(node, node))
            if not in_memory_branch:
                unregistered.append(f"{path.relative_to(src)}:{node.lineno}")
    assert unregistered == []


def test_a_rollback_mode_lock_with_no_journal_yet_is_protected(tmp_path: Path) -> None:
    """``BEGIN IMMEDIATE`` locks before any ``-journal`` exists: liveness is the open
    connection, not a sidecar on disk."""
    from trw_memory._live_stores import connect_registered
    from trw_memory.tools.entry import open_checkout_file_fd

    db = tmp_path / "rollback.db"
    conn = connect_registered(db, sqlite3, str(db), isolation_level=None)
    conn.execute("CREATE TABLE t(x)")
    conn.execute("BEGIN IMMEDIATE")  # RESERVED lock, no write, so no -journal
    try:
        assert not Path(f"{db}-journal").exists()
        refused = open_checkout_file_fd(str(tmp_path.resolve()), str(db.resolve()), "verify")
        assert isinstance(refused, dict) and "has open" in str(refused["error"])
        assert _other_process_can_write(db) is False
    finally:
        conn.execute("ROLLBACK")
        conn.close()


def test_a_closed_store_leaves_the_registry(tmp_path: Path) -> None:
    from trw_memory._live_stores import _OPEN
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.entry import open_checkout_file_fd

    db = tmp_path / "memory.db"
    backend = SQLiteBackend(db)
    identity = (db.stat().st_dev, db.stat().st_ino)
    assert identity in _OPEN
    backend.close()
    assert identity not in _OPEN
    fd = open_checkout_file_fd(str(tmp_path.resolve()), str(db.resolve()), "verify")
    assert isinstance(fd, int)
    from trw_memory._live_stores import close_reader_fd

    close_reader_fd(fd)


def test_a_sidecar_alias_stays_refused_after_the_store_path_is_replaced(tmp_path: Path) -> None:
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.entry import open_checkout_file_fd

    store_dir = tmp_path / "store"
    store_dir.mkdir(mode=0o700)
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o700)
    db = store_dir / "memory.db"
    holder = SQLiteBackend(db)
    holder._conn.execute("BEGIN IMMEDIATE")
    try:
        os.link(f"{db}-shm", checkout / "old-shm.bin")
        # No check runs before the replacement: the -shm inode was recorded when the store opened.
        replacement = store_dir / "replacement.db"
        replacement.write_bytes(b"")
        os.replace(replacement, db)
        Path(f"{db}-shm").unlink()
        refused = open_checkout_file_fd(str(checkout.resolve()), str((checkout / "old-shm.bin").resolve()), "verify")
        assert isinstance(refused, dict) and "has open" in str(refused["error"])
    finally:
        holder._conn.execute("ROLLBACK")
        holder.close()


def test_a_parked_descriptor_outlives_the_refusal_and_closes_with_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name swapped onto a live store AFTER the name check is opened; that descriptor is never
    closed while the store is open (its close would drop the locks), and is closed once the
    store's last connection closes."""
    from trw_memory import _dir_trust, _live_stores
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.entry import open_checkout_file_fd

    monkeypatch.setattr(_dir_trust, "is_known_live", lambda *_a: False)  # the swap lands after the check

    db = tmp_path / "memory.db"
    holder = SQLiteBackend(db)
    holder._conn.execute("BEGIN IMMEDIATE")
    refused = open_checkout_file_fd(str(tmp_path.resolve()), str(db.resolve()), "verify")
    assert isinstance(refused, dict)
    parked = [fd for identity, fd in _live_stores._PARKED if identity == (db.stat().st_dev, db.stat().st_ino)]
    assert len(parked) == 1
    os.fstat(parked[0])  # still open
    assert _other_process_can_write(db) is False
    holder._conn.execute("ROLLBACK")
    holder.close()
    assert not any(fd == parked[0] for _, fd in _live_stores._PARKED)
    with pytest.raises(OSError):
        os.fstat(parked[0])


def test_a_read_lease_holds_off_a_connect_to_that_file_only(tmp_path: Path) -> None:
    import threading

    from trw_memory._live_stores import admit_reader_fd, close_reader_fd, connect_registered

    leased, other = tmp_path / "leased.db", tmp_path / "other.db"
    for path in (leased, other):
        sqlite3.connect(path).close()
    fd = os.open(leased, os.O_RDONLY)
    assert admit_reader_fd(fd) is True
    connected = threading.Event()

    def connect_leased() -> None:
        connect_registered(leased, sqlite3, str(leased)).close()
        connected.set()

    worker = threading.Thread(target=connect_leased)
    worker.start()
    try:
        connect_registered(other, sqlite3, str(other)).close()  # another file is not held up
        assert not connected.wait(0.3), "a connect ran while a reader held the file open"
    finally:
        close_reader_fd(fd)
    assert connected.wait(5), "the connect never ran after the lease was released"
    worker.join(5)


def test_a_connect_whose_file_changes_underneath_is_refused(tmp_path: Path) -> None:
    from trw_memory._live_stores import _OPEN, connect_registered
    from trw_memory.exceptions import StorageError

    db = tmp_path / "memory.db"
    sqlite3.connect(db).close()
    swapped = tmp_path / "swapped.db"
    sqlite3.connect(swapped).close()

    class _SwappingDriver:
        Connection = sqlite3.Connection

        @staticmethod
        def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            conn = sqlite3.connect(*args, **kwargs)  # type: ignore[arg-type]
            os.replace(swapped, db)  # the path now names another file
            return conn

    before = set(_OPEN)
    with pytest.raises(StorageError, match="identity changed during the SQLite connect"):
        connect_registered(db, _SwappingDriver, str(db))
    assert set(_OPEN) == before


def test_connections_through_two_hard_link_names_protect_both(tmp_path: Path) -> None:
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.entry import open_checkout_file_fd

    store_dir = tmp_path / "store"
    store_dir.mkdir(mode=0o700)
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o700)
    first = store_dir / "memory.db"
    SQLiteBackend(first).close()
    second = store_dir / "alias.db"
    os.link(first, second)
    a, b = SQLiteBackend(first), SQLiteBackend(second)
    b._conn.execute("BEGIN IMMEDIATE")
    try:
        for name in ("memory.db", "alias.db", *(n for n in os.listdir(store_dir) if n.endswith(("-wal", "-shm")))):
            os.link(store_dir / name, checkout / f"{name}.bin")
            refused = open_checkout_file_fd(
                str(checkout.resolve()), str((checkout / f"{name}.bin").resolve()), "verify"
            )
            assert isinstance(refused, dict), name
        assert _other_process_can_write(first) is False
    finally:
        b._conn.execute("ROLLBACK")
        a.close()
        b.close()


def test_store_creation_never_opens_an_existing_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A creator that loses the O_EXCL race chmods the winner's file by name; it never opens it,
    because the winner's entry could be a hard link to a live store."""
    from trw_memory.storage import _permissions

    db = tmp_path / "memory.db"
    real_open = os.open
    opens: list[int] = []

    def racing_open(name: object, flags: int, *args: object, **kwargs: object) -> int:
        if str(name) == db.name and flags & os.O_CREAT:
            opens.append(flags)
            if len(opens) == 1:  # another process creates the store first, loosely
                db.touch()
                db.chmod(0o644)
        return real_open(name, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_permissions.os, "open", racing_open)
    _permissions.prepare_db_file_mode(db)
    assert opens and all(flags & os.O_EXCL for flags in opens)
    assert db.stat().st_mode & 0o777 == 0o600


def test_a_rejected_close_keeps_the_store_live(tmp_path: Path) -> None:
    """``check_same_thread`` refuses a close from another thread; the connection stays open,
    so its store must stay registered."""
    import threading

    from trw_memory._live_stores import _OPEN, connect_registered

    db = tmp_path / "memory.db"
    conn = connect_registered(db, sqlite3, str(db))  # check_same_thread=True by default
    identity = (db.stat().st_dev, db.stat().st_ino)
    rejected: list[BaseException] = []

    def close_elsewhere() -> None:
        try:
            conn.close()
        except sqlite3.ProgrammingError as exc:
            rejected.append(exc)

    worker = threading.Thread(target=close_elsewhere)
    worker.start()
    worker.join(5)
    assert rejected, "the cross-thread close was expected to be refused"
    assert identity in _OPEN
    conn.close()
    assert identity not in _OPEN


def test_a_sidecar_appearing_beside_a_new_name_is_caught_at_admission(tmp_path: Path) -> None:
    """A second name's -shm can appear after that name registered but before its open sequence
    records it; the admission that sees it must probe for it itself."""
    from trw_memory._live_stores import admit_reader_fd, connect_registered
    from trw_memory.storage.sqlite_backend import SQLiteBackend

    first = tmp_path / "memory.db"
    holder = SQLiteBackend(first)
    alias = tmp_path / "alias.db"
    os.link(first, alias)
    second = connect_registered(alias, sqlite3, str(alias))  # registered; its open sequence not yet run
    try:
        late_shm = Path(f"{alias}-shm")
        late_shm.write_bytes(b"")  # the new name's sidecar appears mid-open
        fd = os.open(late_shm, os.O_RDONLY)
        assert admit_reader_fd(fd) is False  # parked, not admitted
    finally:
        second.close()
        holder.close()


def test_parked_descriptors_are_capped_and_then_every_read_is_refused_unopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory import _live_stores
    from trw_memory.tools.entry import open_checkout_file_fd

    plain = tmp_path / "notes.txt"
    plain.write_text("x")
    monkeypatch.setattr(_live_stores, "_PARKED", [((0, i), -1) for i in range(_live_stores.MAX_PARKED)])
    opened: list[object] = []
    real_open = os.open
    monkeypatch.setattr(os, "open", lambda *a, **k: opened.append(a[0]) or real_open(*a, **k))
    refused = open_checkout_file_fd(str(tmp_path.resolve()), str(plain.resolve()), "verify")
    assert isinstance(refused, dict)
    assert "notes.txt" not in [str(name) for name in opened]


def test_a_lease_closed_by_hand_does_not_hang_a_connect(tmp_path: Path) -> None:
    from trw_memory._live_stores import admit_reader_fd, connect_registered

    db = tmp_path / "memory.db"
    sqlite3.connect(db).close()
    fd = os.open(db, os.O_RDONLY)
    assert admit_reader_fd(fd) is True
    os.close(fd)  # a caller that forgot close_reader_fd
    connect_registered(db, sqlite3, str(db)).close()  # must not wait forever


def test_a_callers_own_factory_is_still_tracked(tmp_path: Path) -> None:
    from trw_memory._live_stores import _OPEN, connect_registered

    class _Mine(sqlite3.Connection):
        pass

    db = tmp_path / "memory.db"
    conn = connect_registered(db, sqlite3, str(db), factory=_Mine)
    identity = (db.stat().st_dev, db.stat().st_ino)
    assert isinstance(conn, _Mine) and identity in _OPEN
    conn.close()
    assert identity not in _OPEN


def test_the_parked_cap_holds_under_concurrent_swapped_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from trw_memory import _dir_trust, _live_stores
    from trw_memory.storage.sqlite_backend import SQLiteBackend
    from trw_memory.tools.entry import open_checkout_file_fd

    db = tmp_path / "memory.db"
    holder = SQLiteBackend(db)
    alias = tmp_path / "alias.bin"
    os.link(db, alias)
    filler = [((0, i), -1) for i in range(_live_stores.MAX_PARKED - 1)]
    monkeypatch.setattr(_live_stores, "_PARKED", list(filler))
    # Every name check "misses" the live store (a swap after it), but the cap check stays real.
    monkeypatch.setattr(_dir_trust, "is_known_live", lambda *_a: len(_live_stores._PARKED) >= _live_stores.MAX_PARKED)
    barrier = threading.Barrier(8)

    def reader() -> None:
        barrier.wait()
        open_checkout_file_fd(str(tmp_path.resolve()), str(alias.resolve()), "verify")

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    try:
        assert len(_live_stores._PARKED) <= _live_stores.MAX_PARKED
        assert len(_live_stores._PARKED) > len(filler)  # the race did park, up to the cap
    finally:
        _live_stores._PARKED[:] = [entry for entry in _live_stores._PARKED if entry[1] != -1]  # drop the filler
        holder.close()  # closes the parked descriptors once the store is no longer live


def test_the_legacy_no_dir_fd_path_also_creates_exclusively(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.storage import _permissions

    monkeypatch.setattr(_permissions, "DIR_FD_SUPPORTED", False)
    db = tmp_path / "memory.db"
    tmp_path.chmod(0o700)
    real_open = os.open
    opens: list[int] = []

    def racing_open(name: object, flags: int, *args: object, **kwargs: object) -> int:
        if Path(str(name)).name == db.name and flags & os.O_CREAT and flags & os.O_RDWR:
            opens.append(flags)
            if len(opens) == 1:
                db.touch()
                db.chmod(0o644)
        return real_open(name, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_permissions.os, "open", racing_open)
    _permissions.prepare_db_file_mode(db)
    assert opens and all(flags & os.O_EXCL for flags in opens)
    assert db.stat().st_mode & 0o777 == 0o600


def test_a_first_connect_to_a_name_that_appears_as_a_leased_file_waits(tmp_path: Path) -> None:
    """The path is absent at the pre-connect check, then becomes a hard link to a file a reader
    holds open: the connect must still wait for that reader."""
    import threading

    from trw_memory._live_stores import admit_reader_fd, close_reader_fd, connect_registered

    leased = tmp_path / "leased.db"
    sqlite3.connect(leased).close()
    fd = os.open(leased, os.O_RDONLY)
    assert admit_reader_fd(fd) is True
    late = tmp_path / "late.db"
    connected = threading.Event()

    class _LinkingDriver:
        Connection = sqlite3.Connection

        @staticmethod
        def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            if not late.exists():
                os.link(leased, late)  # appears between the check and the open
            return sqlite3.connect(*args, **kwargs)  # type: ignore[arg-type]

    def connect_late() -> None:
        connect_registered(late, _LinkingDriver, str(late)).close()
        connected.set()

    worker = threading.Thread(target=connect_late)
    worker.start()
    try:
        assert not connected.wait(0.8), "the connect went ahead while a reader held the file open"
    finally:
        close_reader_fd(fd)
    assert connected.wait(5)
    worker.join(5)
