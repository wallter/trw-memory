"""Append-only quarantine ledger, keyed on an entry's identity set (PRD-CORE-333 FR01).

One row per decision -- ``quarantined``, ``approved`` or ``rejected`` -- recording the
identity set (namespace-qualified entry id, known derived local ids, source learning id,
content hash), the episode, reason, actor and timestamp. Nothing is updated or deleted:
SQLite triggers refuse both, so approve APPENDS a row rather than clearing one.

The ledger lives in its own file (``MemoryConfig.quarantine_ledger_path``, beside the
other SEC-001 artifacts), never inside an active store, so replacing or rebuilding an
active store leaves every recorded decision in place (NFR01).

Reads go through :meth:`QuarantineLedger.index`, a per-key "latest row" map rebuilt only
when the ledger file changes (one ``stat`` per call decides). An absent ledger file is
the empty ledger, and the empty ledger filters nothing: :meth:`entry_filter` hands the
caller's predicate back unchanged, so a read with nothing quarantined takes exactly the
path it took before this module existed.

Matching is exact on the identity set (NFR02, OQ-001): an entry is blocked when ANY of
its keys' latest decision is ``quarantined`` or ``rejected``. Ids are
namespace-qualified (PRD-CORE-294: an id is only unique within a namespace); the
source learning id and the content hash are global.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.provenance import entry_content_hash
from trw_memory.security.startup import resolve_security_path

__all__ = [
    "BLOCKING_DECISIONS",
    "LedgerDecision",
    "LedgerIdentity",
    "LedgerRow",
    "LedgerView",
    "QuarantineLedger",
    "entry_identity_keys",
    "ledger_for_config",
]

LedgerDecision = Literal["quarantined", "approved", "rejected"]
BLOCKING_DECISIONS: frozenset[str] = frozenset({"quarantined", "rejected"})

_Key = tuple[str, ...]

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS quarantine_ledger (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace TEXT NOT NULL,
        entry_id TEXT NOT NULL,
        source_learning_id TEXT NOT NULL DEFAULT '',
        derived_ids TEXT NOT NULL DEFAULT '',
        content_hash TEXT NOT NULL DEFAULT '',
        episode INTEGER NOT NULL,
        decision TEXT NOT NULL CHECK (decision IN ('quarantined', 'approved', 'rejected')),
        reason TEXT NOT NULL DEFAULT '',
        actor TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS quarantine_ledger_no_update BEFORE UPDATE ON quarantine_ledger
    BEGIN SELECT RAISE(ABORT, 'quarantine_ledger is append-only'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS quarantine_ledger_no_delete BEFORE DELETE ON quarantine_ledger
    BEGIN SELECT RAISE(ABORT, 'quarantine_ledger is append-only'); END
    """,
    # INSERT OR REPLACE on an existing seq deletes that row without firing the DELETE trigger
    # (recursive_triggers is off in every default connection), so an insert may not name a seq
    # already recorded (review r1 P0-2). IF NOT EXISTS: an existing ledger gains it at its next open.
    """
    CREATE TRIGGER IF NOT EXISTS quarantine_ledger_no_replace BEFORE INSERT ON quarantine_ledger
    WHEN EXISTS (SELECT 1 FROM quarantine_ledger WHERE seq = NEW.seq)
    BEGIN SELECT RAISE(ABORT, 'quarantine_ledger is append-only'); END
    """,
)
_COLUMNS = (
    "seq, namespace, entry_id, source_learning_id, derived_ids, content_hash, episode, decision, reason, actor, "
    "recorded_at"
)


@dataclass(frozen=True)
class LedgerIdentity:
    """The identity set one ledger row is keyed on."""

    namespace: str
    entry_id: str
    source_learning_id: str = ""
    derived_ids: tuple[str, ...] = ()
    content_hash: str = ""

    @classmethod
    def of(cls, entry: MemoryEntry, *, source_learning_id: str = "", derived_ids: Iterable[str] = ()) -> LedgerIdentity:
        """The identity set of *entry*, plus any ids the caller knows were derived from it."""
        return cls(
            namespace=entry.namespace,
            entry_id=entry.id,
            source_learning_id=source_learning_id or entry.metadata.get("source_learning_id", ""),
            derived_ids=tuple(derived_ids),
            content_hash=entry_content_hash(entry.content, entry.detail),
        )

    def identity_keys(self) -> list[_Key]:
        keys: list[_Key] = [("id", self.namespace, self.entry_id)]
        keys.extend(("id", self.namespace, derived) for derived in self.derived_ids if derived)
        if self.source_learning_id:
            keys.append(("src", self.source_learning_id))
        if self.content_hash:
            keys.append(("hash", self.content_hash))
        return keys


def entry_identity_keys(entry: MemoryEntry) -> list[_Key]:
    """The ledger keys a stored *entry* answers to: its id, its source learning ids, its content hash."""
    keys: list[_Key] = [("id", entry.namespace, entry.id)]
    source = entry.metadata.get("source_learning_id", "")
    if source:
        keys.append(("src", source))
    if entry.remote_id:
        keys.append(("src", entry.remote_id))
    keys.append(("hash", entry_content_hash(entry.content, entry.detail)))
    return keys


@dataclass(frozen=True)
class LedgerRow:
    """One recorded decision."""

    seq: int
    identity: LedgerIdentity
    episode: int
    decision: str
    reason: str
    actor: str
    recorded_at: str

    @property
    def blocks(self) -> bool:
        return self.decision in BLOCKING_DECISIONS


def _row(values: tuple[object, ...]) -> LedgerRow:
    seq, namespace, entry_id, source, derived, content_hash, episode, decision, reason, actor, recorded_at = values
    derived_ids = tuple(part for part in str(derived).split("\n") if part)
    identity = LedgerIdentity(str(namespace), str(entry_id), str(source), derived_ids, str(content_hash))
    return LedgerRow(
        int(str(seq)), identity, int(str(episode)), str(decision), str(reason), str(actor), str(recorded_at)
    )


_StatKey = tuple[int, int, int, bytes]
#: SQLite's "file change counter" (header offset 24, 4 bytes): bumped by every commit in
#: rollback-journal mode, so it tells two same-size writes apart even when a coarse
#: filesystem clock gives them one mtime.
_CHANGE_COUNTER_OFFSET = 24
_EMPTY_INDEX: dict[_Key, LedgerRow] = {}
# Process-wide, keyed on the ledger path: every backend reading one ledger shares one index.
_INDEX_CACHE: dict[str, tuple[_StatKey, dict[_Key, LedgerRow]]] = {}
_INDEX_LOCK = threading.Lock()


class QuarantineLedger:
    """The append-only decision ledger stored at *path*."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def __repr__(self) -> str:
        return f"QuarantineLedger({str(self._path)!r})"

    @property
    def path(self) -> Path:
        return self._path

    # -- writes -------------------------------------------------------------

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection through the package's one factory (``storage._connection.connect``).

        It registers the ledger's inode with the live-store registry and holds its store
        lock until the close, so no other descriptor this process opens and closes on the
        file can drop the connection's POSIX locks (C15; POSIX fcntl locks go with ANY close).
        """
        from trw_memory.storage._connection import connect  # storage imports this module

        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        conn = connect(self._path, dbapi=sqlite3, timeout=30.0, check_same_thread=True)
        conn.isolation_level = None  # explicit BEGIN IMMEDIATE / COMMIT below
        conn.row_factory = None  # plain tuples, as _row unpacks them
        conn.execute("PRAGMA journal_mode=DELETE")  # every commit rewrites the header counter
        try:
            with contextlib.suppress(OSError):
                os.chmod(self._path, 0o600)
            for statement in _SCHEMA:
                conn.execute(statement)
            yield conn
        finally:
            conn.close()

    def append(self, identity: LedgerIdentity, decision: LedgerDecision, *, actor: str, reason: str = "") -> LedgerRow:
        """Append one decision for *identity* and return the recorded row.

        ``quarantined`` opens a new episode unless the identity's current episode is
        already a quarantine; ``approved``/``rejected`` close the current one. The read
        of the current episode and the insert share one ``BEGIN IMMEDIATE``, so two
        processes cannot record the same episode number for different decisions.
        """
        if decision not in {"quarantined", "approved", "rejected"}:
            raise ValueError(f"unknown ledger decision: {decision!r}")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                latest = self._latest_in(conn, identity.identity_keys())
                episode = latest.episode if latest is not None else 1
                if decision == "quarantined" and latest is not None and latest.decision != "quarantined":
                    episode += 1
                recorded_at = datetime.now(timezone.utc).isoformat()
                cursor = conn.execute(
                    "INSERT INTO quarantine_ledger (namespace, entry_id, source_learning_id, derived_ids, "
                    "content_hash, episode, decision, reason, actor, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        identity.namespace,
                        identity.entry_id,
                        identity.source_learning_id,
                        "\n".join(identity.derived_ids),
                        identity.content_hash,
                        episode,
                        decision,
                        reason,
                        actor,
                        recorded_at,
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        _INDEX_CACHE.pop(str(self._path), None)
        return LedgerRow(int(cursor.lastrowid or 0), identity, episode, decision, reason, actor, recorded_at)

    @staticmethod
    def _latest_in(conn: sqlite3.Connection, keys: list[_Key]) -> LedgerRow | None:
        latest: LedgerRow | None = None
        for row in (_row(values) for values in conn.execute(f"SELECT {_COLUMNS} FROM quarantine_ledger")):  # noqa: S608
            if (latest is None or row.seq > latest.seq) and not set(keys).isdisjoint(row.identity.identity_keys()):
                latest = row
        return latest

    # -- reads --------------------------------------------------------------

    def rows(self) -> list[LedgerRow]:
        """Every recorded row, oldest first."""
        if not self._path.exists():
            return []
        with self._connect() as conn:
            return [
                _row(values)
                for values in conn.execute(f"SELECT {_COLUMNS} FROM quarantine_ledger ORDER BY seq")  # noqa: S608
            ]

    def index(self) -> dict[_Key, LedgerRow]:
        """Each identity key's latest row, rebuilt only when the ledger file changed.

        Per call: one ``open`` + ``fstat`` + 4-byte header read; an absent file is the empty
        ledger and costs the failed ``open`` alone.
        """
        try:
            stat_key = self._stat_key()
        except FileNotFoundError:
            return _EMPTY_INDEX
        if stat_key is None:  # a ledger connection is open in this process: read through one, uncached
            return self._build_index()
        cache_key = str(self._path)
        cached = _INDEX_CACHE.get(cache_key)
        if cached is not None and cached[0] == stat_key:
            return cached[1]
        with _INDEX_LOCK:
            index = self._build_index()
            _INDEX_CACHE[cache_key] = (stat_key, index)
        return index

    def _build_index(self) -> dict[_Key, LedgerRow]:
        index: dict[_Key, LedgerRow] = {}
        for row in self.rows():  # ascending seq: a later row replaces an earlier one per key
            for key in row.identity.identity_keys():
                index[key] = row
        return index

    def _stat_key(self) -> _StatKey | None:
        """The file's identity, size, mtime and change counter; ``None`` while this process has it open.

        The header read opens a descriptor, and closing one on a file a connection has open would
        drop that connection's POSIX locks (C15). So the open goes through the live-store reader
        guard: refused by name when the ledger is live here, and leased otherwise.
        """
        from trw_memory._live_stores import FD_LOCK, admit_reader_fd, close_reader_fd, is_known_live

        parent = os.open(self._path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            with FD_LOCK:
                if is_known_live(parent, self._path.name):
                    return None
                fd = os.open(self._path.name, os.O_RDONLY, dir_fd=parent)
                if not admit_reader_fd(fd):  # became live between the check and the open: parked, never closed here
                    return None
        finally:
            os.close(parent)
        try:
            st = os.fstat(fd)
            return (st.st_ino, st.st_size, st.st_mtime_ns, os.pread(fd, 4, _CHANGE_COUNTER_OFFSET))
        finally:
            close_reader_fd(fd)

    def latest_decision(self, candidate: MemoryEntry | LedgerIdentity) -> LedgerRow | None:
        """The latest row recorded for any key of *candidate*'s identity set, or ``None``."""
        keys = candidate.identity_keys() if isinstance(candidate, LedgerIdentity) else entry_identity_keys(candidate)
        index = self.index()
        matches = [index[key] for key in keys if key in index]
        return max(matches, key=lambda row: row.seq) if matches else None

    def view(self) -> LedgerView:
        """The ledger as it stands now: what one read call checks its rows against."""
        return LedgerView(self.index())


class LedgerView:
    """One read call's snapshot of the ledger index: every check below is O(1) per entry."""

    __slots__ = ("_index",)

    def __init__(self, index: dict[_Key, LedgerRow]) -> None:
        self._index = index

    def __bool__(self) -> bool:
        return bool(self._index)

    def blocks(self, entry: MemoryEntry) -> bool:
        """Whether any identity key of *entry* currently resolves to ``quarantined``/``rejected``."""
        index = self._index
        return any((row := index.get(key)) is not None and row.blocks for key in entry_identity_keys(entry))

    def filter(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """*entries* minus every blocked one (*entries* itself when nothing is ledgered)."""
        if not self._index:
            return entries
        return [entry for entry in entries if not self.blocks(entry)]

    def entry_filter(self, inner: Callable[[MemoryEntry], bool] | None) -> Callable[[MemoryEntry], bool] | None:
        """*inner* extended to refuse blocked entries (*inner* itself when nothing is ledgered)."""
        if not self._index:
            return inner
        blocks = self.blocks
        if inner is None:
            return lambda entry: not blocks(entry)
        return lambda entry: not blocks(entry) and inner(entry)


class LedgerSeedError(RuntimeError):
    """A source ledger exists but could not be copied: nothing was seeded."""


def seed_ledger(source: Path, destination: Path) -> tuple[int, int] | None:
    """Copy the ledger at *source* to *destination* as one consistent snapshot; the published ``(st_dev, st_ino)``.

    A store seeded without its ledger serves every identity the source blocks (PRD-CORE-333), so
    a seed carries it. ``None`` only when *source* does not exist; a lookup that fails any other
    way (EACCES on its directory) refuses, so an unreadable ledger is never taken for an absent one. The copy is the store
    snapshot's own ``VACUUM INTO`` over a registered connection (``storage._snapshot``, 30 s busy
    timeout) into a temp file beside *destination*, published by ``publish_no_clobber``: a ledger
    that appears at *destination* at any point (a daemon's first decision) is never replaced.

    Raises :class:`LedgerSeedError` when *source* is a symlink, cannot be looked up or read, stays locked past
    the busy timeout, or *destination* exists by the time of the publish.
    """
    from trw_memory.storage import _snapshot  # storage imports this module

    try:
        identity = _snapshot.seed_no_clobber(source, destination)
    except FileExistsError as exc:
        raise LedgerSeedError(
            f"the quarantine ledger cannot be seeded: {destination} already exists, and seeding never overwrites a "
            "ledger; no store was seeded"
        ) from exc
    except (_snapshot.SnapshotError, OSError) as exc:
        if os.path.islink(source):  # Path.is_symlink() re-raises EACCES on Python < 3.13
            raise LedgerSeedError(f"the quarantine ledger {source} is a symlink; no store was seeded") from exc
        raise LedgerSeedError(
            f"the quarantine ledger {source} could not be copied to {destination} ({type(exc).__name__}: {exc}); "
            "no store was seeded. The user can check the permissions of that file and its directories, or wait for "
            "the source env's memory daemon to finish writing it, then create the env again."
        ) from exc
    if identity is None:
        return None
    _INDEX_CACHE.pop(str(destination), None)
    return identity


def ledger_for_config(config: MemoryConfig) -> QuarantineLedger:
    """The ledger *config*'s security paths name (``quarantine_ledger_path``)."""
    return QuarantineLedger(resolve_security_path(config, "quarantine_ledger_path"))
