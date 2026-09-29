"""SQLite store files this process holds open, and how every other descriptor coexists with them.

POSIX fcntl locks belong to a (process, inode) pair: ``close()`` on ANY
descriptor for a file releases every lock this process holds on it, SQLite's
included ("How To Corrupt An SQLite Database" §2.2). SQLite coordinates its own
connections, but a descriptor anything else in the process closes on a live
store, its ``-wal`` or its ``-shm`` silently strips the live connections' locks,
and another process then writes under them (C15, 2026-09-24).

The rules, all enforced through this module:

- Every file-backed SQLite connection in the package opens through
  :func:`connect_registered`. It counts the connection against the store's inode
  until the connection closes or is collected, and refuses a connection whose
  inode it cannot pin down. A collected connection's finalizer only queues its
  release; the next registry access applies it, so a collection can never change
  the registry under a scan that already holds :data:`FD_LOCK`.
- Nothing opens an existing store file by descriptor. New store files are
  created (the one descriptor use left) under :data:`FD_LOCK`, which every
  connect holds too.
- A reader of a caller-named path admits the descriptor it ACTUALLY opened
  (:func:`admit_reader_fd`, by ``fstat``, so no path swap between a check and
  the open can fool it). A live store's descriptor is parked: never closed
  while a connection to it is open, and so harmless. Any other descriptor holds
  a read lease on its inode until :func:`close_reader_fd`. A connect to a
  leased inode waits for the lease, so no store can go live under an open read,
  and reads of other files never block it.
- A reader that needs a consistent snapshot of a SQLite file holds SQLite's own
  SHARED lock through :func:`sqlite_read_lock`. While any holder has it, reader
  descriptors on that inode are closed only after the last holder releases, since
  closing one would drop the lock for every holder.

Liveness is by inode, never by name, so a hard link or any other alias of a
store or of its sidecars counts too. A connection holds locks from open to close
whatever its journal mode, so a store is live exactly while a connection to it
is open.
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import fcntl
import os
import sqlite3
import threading
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trw_memory._inode_pin import Identity, current_identity, pinned_identity
from trw_memory.exceptions import StorageError

#: Held while a store file is created by descriptor, and while a connection is
#: opened, registered or released, or a reader descriptor admitted or closed.
FD_LOCK = threading.RLock()
_LEASE_RELEASED = threading.Condition(FD_LOCK)
#: Every registered connection -> (its path, its inode, its release-once flag, the thread that opened it if it is
#: confined to that thread (``check_same_thread``), else None), for :func:`_after_fork_in_child`.
_CONNECTIONS: weakref.WeakKeyDictionary[Any, tuple[Path, Identity, list[bool], int | None]] = (
    weakref.WeakKeyDictionary()
)
#: Inherited connections the child quarantined: referenced (and made immortal) so no collection ever closes them.
_KEPT_AFTER_FORK: list[Any] = []
#: This process's pid (re-read in a forked child): a connection is usable only in the process that opened it.
_PID = os.getpid()


def _after_fork_in_child() -> None:
    """FD_LOCK is held across fork(): a child forked while another thread held it would inherit it locked, by
    nobody. A child holds no store lock, so no connection it inherited is usable: every method of one, and of
    a cursor made from it, refuses in a process other than the one that opened it (:func:`_refuse_inherited`).

    Only a connection confined to the forking thread is closed, with the driver's close: no other thread can
    have been inside SQLite on it at the fork. That close touches no file of the parent's: a WAL store's close
    sees the parent's SHARED lock and skips its checkpoint, and an idle rollback-journal connection has nothing
    to roll back. Every other connection is quarantined without one call into SQLite (B71-133): another
    thread's may have been mid-SQL, its mutex inherited locked by a thread that does not exist here, and even
    its close would wait forever; a rollback-journal write in flight would play the journal back over the
    parent's transaction. A quarantined connection leaves the registry and is never closed or collected."""
    global _PID
    _PID = os.getpid()
    FD_LOCK.release()
    forking_thread = threading.get_ident()
    for conn, (path, identity, done, confined_to) in list(_CONNECTIONS.items()):
        if done[0]:
            continue
        closable = confined_to == forking_thread
        try:
            closable = closable and (not conn.in_transaction or os.path.exists(f"{os.path.realpath(path)}-wal"))
            if closable:  # the driver's own close: the child has no store hold to release
                getattr(type(conn).__mro__[1], "close")(conn)  # noqa: B009 - the driver class is typed as a bare type
        except Exception:  # trw-fail-silent-allow: a close the driver refused leaves it quarantined, still refused
            closable = False
        if not closable:
            _KEPT_AFTER_FORK.append(conn)
            ctypes.pythonapi.Py_IncRef(ctypes.py_object(conn))  # immortal: interpreter teardown would close it
        _apply_release(identity, done)


def _refuse_inherited(conn: Any) -> None:
    if getattr(conn, "_trw_pid", _PID) != _PID:
        raise StorageError("this SQLite connection was inherited across fork(); open a connection of your own")


def _guarded(method: Any, owner: Any = lambda self: self) -> Any:
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        _refuse_inherited(owner(self))
        return method(self, *args, **kwargs)

    return guarded


def _guarded_cursor_class(factory: Any) -> type:
    """*factory*'s subclass whose methods refuse in a process other than the one that opened the connection. A
    factory that is not a ``sqlite3.Cursor`` subclass could return a cursor nothing guards: it is refused."""
    if not (isinstance(factory, type) and issubclass(factory, sqlite3.Cursor)):
        raise StorageError(f"a registered connection's cursor factory must subclass sqlite3.Cursor, not {factory!r}")
    cached = _FACTORIES.get(factory)
    if cached is None:
        guards = {
            name: _guarded(getattr(factory, name), lambda cursor: cursor.connection)
            for name in ("execute", "executemany", "executescript", "fetchone", "fetchmany", "fetchall")
        }
        cached = _FACTORIES[factory] = type(f"Guarded{factory.__name__}", (factory,), guards)
    return cached


os.register_at_fork(before=FD_LOCK.acquire, after_in_parent=FD_LOCK.release, after_in_child=_after_fork_in_child)

_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


@dataclass
class _Store:
    #: Every name a connection opened this inode by (hard links included).
    paths: set[Path] = field(default_factory=set)
    open_connections: int = 0
    #: Sidecar inodes seen while open; kept so an alias still matches after the path is replaced.
    sidecars: set[Identity] = field(default_factory=set)
    #: Names whose WAL index (-shm) is recorded; admissions keep probing every other name.
    shm_recorded: set[Path] = field(default_factory=set)


_OPEN: dict[Identity, _Store] = {}
#: Read leases: inode -> the reader descriptors holding it.
_LEASES: dict[Identity, set[int]] = {}
#: Reader descriptors on a live inode: closing one would drop the store's locks.
_PARKED: list[tuple[Identity, int]] = []
#: Parked descriptors only come from a path swapped onto a live store between the
#: name check and the open. Past this many, every reader open is refused instead,
#: so a swap loop cannot exhaust the process's descriptors.
MAX_PARKED = 64
_LEASE_POLL_SECONDS = 0.5
#: Driver class (a connection's, or a cursor factory) -> its registered or guarded subclass.
_FACTORIES: dict[type, type] = {}
#: Inodes this process holds SQLite's SHARED lock on, and how many holders share it.
_READ_LOCKS: dict[Identity, int] = {}
#: Reader descriptors whose close waits for their inode's last read-lock holder.
_DEFERRED_CLOSES: dict[Identity, list[int]] = {}
#: SQLite's unix-VFS lock bytes (``os_unix.c``): a reader holds a read lock on the SHARED
#: range, taken under a read lock on PENDING, for the length of a read transaction.
_PENDING_BYTE = 0x40000000
_SHARED_FIRST = _PENDING_BYTE + 2
_SHARED_SIZE = 510
#: Store lock files this process holds a descriptor on (``_store_lock``): live even with no connection open.
_LOCK_FILES: set[Identity] = set()
#: Union of open store, sidecar and lock-file inodes; None when any of them changed.
_LIVE_CACHE: set[Identity] | None = None
#: Releases of connections the garbage collector finalized. A finalizer runs on whatever thread
#: triggered the collection, possibly inside a loop over ``_OPEN`` that already holds FD_LOCK
#: (an RLock, so it would re-enter), so it only appends here: ``deque.append`` is atomic and
#: takes no lock. :func:`_locked` applies them before anything reads the registry.
_FINALIZED: collections.deque[tuple[Identity, list[bool]]] = collections.deque()


@contextlib.contextmanager
def _locked() -> Iterator[None]:
    """Hold FD_LOCK with every finalized connection's release applied: every entry point that
    reads or changes the registry goes through here, never through FD_LOCK alone."""
    with FD_LOCK:
        _drain_finalized()
        yield


def _drain_finalized() -> None:
    while _FINALIZED:
        _apply_release(*_FINALIZED.popleft())


def _record_sidecars(identity: Identity, store: _Store) -> None:
    """Record the sidecars beside every name that still names *identity*."""
    global _LIVE_CACHE
    for path in store.paths:
        if current_identity(path) != identity:
            continue
        for suffix in _SIDECAR_SUFFIXES:
            sidecar = current_identity(Path(f"{path}{suffix}"))
            if sidecar is not None and sidecar not in store.sidecars:
                store.sidecars.add(sidecar)
                _LIVE_CACHE = None
                if suffix == "-shm":
                    store.shm_recorded.add(path)


def _live_identities() -> set[Identity]:
    """Open store and sidecar inodes. Probes only names whose own WAL index is not recorded yet,
    so a sidecar a new name creates mid-open is caught by the admission that would see it."""
    global _LIVE_CACHE
    for identity, store in _OPEN.items():
        if store.paths - store.shm_recorded:
            _record_sidecars(identity, store)
    if _LIVE_CACHE is None:
        _LIVE_CACHE = set(_OPEN).union(_LOCK_FILES, *(store.sidecars for store in _OPEN.values()))
    return _LIVE_CACHE


def _release(identity: Identity, done: list[bool]) -> None:
    """An explicit ``close()``: release the connection's count now."""
    with _locked():
        _apply_release(identity, done)


def _apply_release(identity: Identity, done: list[bool]) -> None:
    """Drop one connection's count (once: close and finalizer share *done*). Caller holds FD_LOCK."""
    if done[0]:
        return
    done[0] = True
    store = _OPEN.get(identity)
    if store is None:
        return
    store.open_connections -= 1
    if store.open_connections > 0:
        return
    del _OPEN[identity]
    _close_unparked()


def _close_unparked() -> None:
    global _LIVE_CACHE
    _LIVE_CACHE = None
    live = _live_identities()
    for parked in [p for p in _PARKED if p[0] not in live]:
        _PARKED.remove(parked)
        os.close(parked[1])  # nothing in this process holds a lock on this inode any more


def track_lock_file(identity: Identity, *, open_: bool) -> None:
    """A store lock file's descriptor opened (it is live from now on) or closed; the caller holds :data:`FD_LOCK`
    (or is a just-forked child)."""
    (_LOCK_FILES.add if open_ else _LOCK_FILES.discard)(identity)
    _close_unparked()


def _tracked_factory(connection_cls: type) -> type:
    """A subclass of the driver's ``Connection`` whose ``close()`` releases its registry count. A SQLite one's
    methods, and its cursors', refuse in a forked child (B71-133), so a pre-fork bound method or cursor does too."""
    cached = _FACTORIES.get(connection_cls)
    if cached is None:

        def close(self: Any) -> None:
            if getattr(self, "_trw_pid", _PID) != _PID:
                return  # the child's fork handler released it; the driver's close could wait forever here
            # Released only once the driver has closed: a rejected close (another
            # thread under check_same_thread) leaves the connection and its locks live.
            getattr(connection_cls, "close")(self)  # noqa: B009 - the driver class is typed as a bare type
            release = getattr(self, "_trw_release", None)
            if release is not None:
                release()

        def cursor(self: Any, factory: Any = None) -> Any:
            _refuse_inherited(self)
            guarded = _guarded_cursor_class(factory or sqlite3.Cursor)  # a caller's factory is guarded too
            return getattr(connection_cls, "cursor")(self, guarded)  # noqa: B009

        def through_cursor(name: str) -> Any:  # the driver's execute* would make a cursor of its own
            return lambda self, *args: getattr(self.cursor(), name)(*args)

        namespace: dict[str, Any] = {"close": close}
        if issubclass(connection_cls, sqlite3.Connection):
            namespace["cursor"] = cursor
            namespace.update({name: through_cursor(name) for name in ("execute", "executemany", "executescript")})
            for name in ("commit", "rollback", "backup", "blobopen", "deserialize", "__exit__"):
                if hasattr(connection_cls, name):
                    namespace[name] = _guarded(getattr(connection_cls, name))
        cached = type(f"Registered{connection_cls.__name__}", (connection_cls,), namespace)
        _FACTORIES[connection_cls] = cached
    return cached


def connect_registered(db_path: Path | str, dbapi: Any, *args: Any, store_lock: bool = True, **kwargs: Any) -> Any:
    """``dbapi.connect(*args, **kwargs)``, counted as open on *db_path*'s inode until closed.

    The connection also holds the store's ``OPEN`` op (B71-00) until it closes, so no
    other process can replace the store under it. ``store_lock=False`` is for files
    that are not stores (backup targets, scratch copies); each such call is listed
    in ``test_direct_sqlite_connect_census.py``.

    Waits while a reader leases that inode. Refuses (closing it) a connection
    whose store it cannot identify: the path missing, or naming a different
    inode after the connect than before. Only a real driver connection (an
    instance of the driver's own ``Connection`` class, which every SQLite driver
    has) is registered: a test double holds no descriptor, and registering one
    would leave an entry nothing closes -- on Linux, where a freed inode number is
    reused at once, that stale entry would then refuse an unrelated file.
    """
    from trw_memory import _store_lock  # imports this module

    target = str(args[0]) if args else ""
    read_only = bool(kwargs.get("uri")) and ("mode=ro" in target or "immutable=1" in target)
    hold = _store_lock.acquire(db_path, "open", read_only=read_only) if store_lock else None
    try:
        conn, identity = _connect_counted(Path(db_path), dbapi, args, kwargs)
    except BaseException:
        _store_lock.release(hold)
        raise
    if identity is None:  # a test double: no descriptor, so no lock to keep
        _store_lock.release(hold)
        return conn
    done = [False]
    # A connection another thread could be inside at a fork is never closed by the child (B71-133).
    confined_to = threading.get_ident() if kwargs.get("check_same_thread", True) and len(args) < 5 else None

    def release() -> None:
        _release(identity, done)  # FD_LOCK first, then the store hold: never both at once
        _store_lock.release(hold)

    conn._trw_release = release
    conn._trw_pid = _PID
    weakref.finalize(conn, _finalize, identity, done, hold)
    with _locked():
        _CONNECTIONS[conn] = (Path(db_path), identity, done, confined_to)
    if hold is not None:
        try:
            _store_lock.check_one_name(hold.db)  # a hard link made between the lock and the connect
        except StorageError:
            conn.close()
            raise
    return conn


def _finalize(identity: Identity, done: list[bool], hold: Any) -> None:
    """A collected connection's release: queued, never applied here (see :data:`_FINALIZED`)."""
    from trw_memory import _store_lock

    _FINALIZED.append((identity, done))
    _store_lock.release_soon(hold)


def _connect_counted(
    path: Path, dbapi: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, Identity | None]:
    """The connect, counted against its inode; the identity is None for a test double (not counted)."""
    global _LIVE_CACHE
    # A caller's own factory is tracked too, by subclassing it: no connection escapes.
    candidate = kwargs.pop("factory", None) or getattr(dbapi, "Connection", None)
    connection_cls = candidate if isinstance(candidate, type) else None
    with _locked():
        while True:
            _drain_finalized()  # the lease wait below releases FD_LOCK, so collections may have queued more
            # Leases are checked BEFORE the pin opens: its descriptor could take the number of
            # a lease closed by hand, which would then look live forever. No lease can be
            # admitted between the check and the pin (FD_LOCK), and a leased inode stays
            # allocated, so a swap in between cannot reuse its number.
            unpinned = current_identity(path)
            if unpinned is not None and _live_leases(unpinned):
                _LEASE_RELEASED.wait(_LEASE_POLL_SECONDS)
                continue
            # The pin holds the inode across the connect, so a swapped-in file can never
            # come back with the same (st_dev, st_ino) -- Linux reuses freed inode numbers.
            with pinned_identity(path) as before:
                if before != unpinned:
                    continue  # the path changed before the pin: check the new file's leases
                if connection_cls is not None:
                    conn = dbapi.connect(*args, factory=_tracked_factory(connection_cls), **kwargs)
                else:
                    conn = dbapi.connect(*args, **kwargs)
                is_real_connection = connection_cls is not None and isinstance(conn, connection_cls)
                if not is_real_connection:
                    # Not a real driver connection (a test double): it holds no file descriptor,
                    # so no lock to protect -- and it could never report its close.
                    return conn, None
                identity = current_identity(path)
            if before is None and identity is not None and _live_leases(identity):
                # The path was absent at the check and now names a leased file (e.g. a
                # hard link made in between). The new connection has taken no lock yet:
                # close it, wait for the reader, and connect again.
                conn.close()
                _LEASE_RELEASED.wait(_LEASE_POLL_SECONDS)
                continue
            break
        if identity is None or (before is not None and identity != before):
            conn.close()
            raise StorageError(
                f"{path}'s identity changed during the SQLite connect (before={before}, after={identity}); "
                "refusing a connection whose file cannot be identified",
                path=str(path),
            )
        store = _OPEN.setdefault(identity, _Store())
        store.paths.add(path)
        store.open_connections += 1
        _LIVE_CACHE = None
        _record_sidecars(identity, store)
        return conn, identity


def note_store_sidecars(db_path: Path | str) -> None:
    """Record *db_path*'s sidecar inodes now; called once the connection has entered WAL mode."""
    with _locked():
        identity = current_identity(Path(db_path))
        store = _OPEN.get(identity) if identity is not None else None
        if identity is not None and store is not None:
            store.paths.add(Path(db_path))
            _record_sidecars(identity, store)


def _live_leases(identity: Identity) -> bool:
    """Whether a reader still holds *identity*; drops leases whose descriptor was closed without
    :func:`close_reader_fd` (or now names another file), so a leaked lease cannot hang a connect."""
    fds = _LEASES.get(identity)
    if not fds:
        return False
    for fd in list(fds):
        try:
            st = os.fstat(fd)
        except OSError:  # trw-fail-silent-allow: EBADF means the reader closed it by hand; the stale lease is dropped, which is the point
            fds.discard(fd)
            continue
        if (st.st_dev, st.st_ino) != identity:
            fds.discard(fd)
    if not fds:
        del _LEASES[identity]
        return False
    return True


def is_known_live(dir_fd: int, name: str) -> bool:
    """Name-based pre-check before a reader opens *name*: a known live store or sidecar is refused
    without opening anything, and so without parking a descriptor. Once the parked cap is
    reached every open is refused (fail closed, bounded). The descriptor check in
    :func:`admit_reader_fd` still decides for a name swapped after this call."""
    with _locked():
        if len(_PARKED) >= MAX_PARKED:
            return True
        if not (_OPEN or _LOCK_FILES):
            return False
        identity = current_identity(name, dir_fd=dir_fd)
        return identity is not None and identity in _live_identities()


def admit_reader_fd(fd: int) -> bool:
    """Admit a just-opened reader descriptor: lease its inode, or park it if a store is live on it.

    Returns False for a parked descriptor. The caller then treats the read as
    refused and must NOT close it: this module closes it once the store's last
    connection has closed.
    """
    st = os.fstat(fd)
    identity = (st.st_dev, st.st_ino)
    with _locked():
        if (_OPEN or _LOCK_FILES) and identity in _live_identities():
            _PARKED.append((identity, fd))
            return False
        _LEASES.setdefault(identity, set()).add(fd)
        return True


def close_reader_fd(fd: int) -> None:
    """Close an admitted reader descriptor and release its lease.

    Deferred while a :func:`sqlite_read_lock` is held on its inode: the close would drop it.
    """
    st = os.fstat(fd)
    identity = (st.st_dev, st.st_ino)
    with _locked():
        if identity in _READ_LOCKS:
            _DEFERRED_CLOSES.setdefault(identity, []).append(fd)
        else:
            _close_leased(identity, fd)


def _close_leased(identity: Identity, fd: int) -> None:
    try:
        os.close(fd)
    finally:
        fds = _LEASES.get(identity)
        if fds is not None:
            fds.discard(fd)
            if not fds:
                del _LEASES[identity]
        _LEASE_RELEASED.notify_all()


@contextlib.contextmanager
def sqlite_read_lock(fd: int) -> Iterator[None]:
    """Hold SQLite's own SHARED lock on admitted reader *fd*, as a reader in a read transaction does.

    While it is held no rollback-mode writer in another process can reach EXCLUSIVE, so the
    file cannot change. A writer already holding PENDING or EXCLUSIVE makes this raise
    ``OSError`` rather than wait. The lock is per (process, inode), so holders of one inode
    share it and it is released with the last one; :func:`close_reader_fd` defers every close
    on that inode until then. A live store's inode never gets here: its descriptor is parked.
    """
    st = os.fstat(fd)
    identity = (st.st_dev, st.st_ino)
    with _locked():
        if identity not in _READ_LOCKS:
            try:
                fcntl.lockf(fd, fcntl.LOCK_SH | fcntl.LOCK_NB, 1, _PENDING_BYTE)
                try:
                    fcntl.lockf(fd, fcntl.LOCK_SH | fcntl.LOCK_NB, _SHARED_SIZE, _SHARED_FIRST)
                finally:
                    fcntl.lockf(fd, fcntl.LOCK_UN, 1, _PENDING_BYTE)
            except OSError as exc:
                raise OSError(f"a writer holds the file's SQLite lock; retry once it is idle ({exc})") from exc
        _READ_LOCKS[identity] = _READ_LOCKS.get(identity, 0) + 1
    try:
        yield
    finally:
        with _locked():
            _READ_LOCKS[identity] -= 1
            if not _READ_LOCKS[identity]:
                del _READ_LOCKS[identity]
                fcntl.lockf(fd, fcntl.LOCK_UN, _SHARED_SIZE, _SHARED_FIRST)
                for deferred in _DEFERRED_CLOSES.pop(identity, []):
                    _close_leased(identity, deferred)
