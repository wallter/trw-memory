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
import fcntl
import os
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
#: Union of open store and sidecar inodes; None when a store opened, closed or gained a sidecar.
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
        _LIVE_CACHE = set(_OPEN).union(*(store.sidecars for store in _OPEN.values()))
    return _LIVE_CACHE


def _release(identity: Identity, done: list[bool]) -> None:
    """An explicit ``close()``: release the connection's count now."""
    with _locked():
        _apply_release(identity, done)


def _apply_release(identity: Identity, done: list[bool]) -> None:
    """Drop one connection's count (once: close and finalizer share *done*). Caller holds FD_LOCK."""
    global _LIVE_CACHE
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
    _LIVE_CACHE = None
    live = _live_identities()
    for parked in [p for p in _PARKED if p[0] not in live]:
        _PARKED.remove(parked)
        os.close(parked[1])  # no connection holds a lock on this inode any more


def _tracked_factory(connection_cls: type) -> type:
    """A subclass of the driver's ``Connection`` whose ``close()`` releases its registry count."""
    cached = _FACTORIES.get(connection_cls)
    if cached is None:

        def close(self: Any) -> None:
            # Released only once the driver has closed: a rejected close (another
            # thread under check_same_thread) leaves the connection and its locks live.
            getattr(connection_cls, "close")(self)  # noqa: B009 - the driver class is typed as a bare type
            release = getattr(self, "_trw_release", None)
            if release is not None:
                release()

        cached = type(f"Registered{connection_cls.__name__}", (connection_cls,), {"close": close})
        _FACTORIES[connection_cls] = cached
    return cached


def connect_registered(db_path: Path | str, dbapi: Any, *args: Any, **kwargs: Any) -> Any:
    """``dbapi.connect(*args, **kwargs)``, counted as open on *db_path*'s inode until closed.

    Waits while a reader leases that inode. Refuses (closing it) a connection
    whose store it cannot identify: the path missing, or naming a different
    inode after the connect than before. Only a real driver connection (an
    instance of the driver's own ``Connection`` class, which every SQLite driver
    has) is registered: a test double holds no descriptor, and registering one
    would leave an entry nothing closes -- on Linux, where a freed inode number is
    reused at once, that stale entry would then refuse an unrelated file.
    """
    global _LIVE_CACHE
    path = Path(db_path)
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
                    return conn
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
        done = [False]
        conn._trw_release = lambda: _release(identity, done)
        weakref.finalize(conn, _FINALIZED.append, (identity, done))
        return conn


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
        if not _OPEN:
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
        if _OPEN and identity in _live_identities():
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
