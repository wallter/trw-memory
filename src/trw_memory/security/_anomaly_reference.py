"""The per-namespace anomaly reference window, kept current incrementally.

The runtime intake gate scores every stored entry against a rolling window:
the ``_ROLLING_WINDOW`` most recently updated ACTIVE, non-quarantined,
non-canary entries of the entry's namespace (``_runtime_anomaly.score_anomaly``).
Re-reading that window (``_REFERENCE_FETCH_LIMIT`` rows decoded) on every
single-row store made the store cost grow with the namespace and was most of a
store's time by a few hundred rows.

This module keeps, per ``(database, namespace)`` and per process, the ranked
reference rows reduced to the few numbers the scorer and the stats file read,
and brings them up to date from the backend's change feed:

1. read the namespace's :class:`NamespaceChangeToken` (two index seeks);
2. equal to the cached token: reuse the cached view;
3. otherwise merge ``entries_changed_since`` (rows inserted or stamped since
   the cached token, any status) into the ranked rows;
4. re-read the window in full (a *reseed*) when the feed cannot account for
   the change: a delete (this process's deletes bump the token's epoch; a
   token maximum that went down reveals another process's), more changed rows
   than the feed limit, a ranked row leaving so that deeper rows would move up,
   or a snapshot older than ``_RESEED_MAX_AGE_S``.

Detection is the same as re-reading the window on every store (the same rows,
ranked the same way, and the same statistics) with one bounded exception:
the writes ``storage/_change_feed.py`` lists as invisible to the token, in
practice another process's DELETE. A row deleted that way stays in this
process's reference until the next reseed, at most ``_RESEED_MAX_AGE_S`` later.
Backends without a change token (``YAMLBackend``, test doubles) re-read the
window on every call, as before.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import NamespaceChangeToken, StorageBackend

# Rolling-window size used for per-namespace anomaly statistics.
_ROLLING_WINDOW = 100
# Over-fetch buffer: list_entries already returns updated_at DESC, so the
# rolling window is the first _ROLLING_WINDOW clean rows. We fetch 2x the
# window so the small set of in-store filtered rows (system canaries — capped
# at 5 by canary_injection_rate; legacy quarantined rows are kept in a
# SEPARATE quarantine store) can be skipped without dropping below the window.
_REFERENCE_FETCH_LIMIT = _ROLLING_WINDOW * 2
# Change-feed sizes tried before a reseed. A single-row writer changes one row
# between scores, so the small feed almost always suffices.
_FEED_LIMITS = (4, 64)
# Upper bound on how long another process's undetectable delete can linger.
_RESEED_MAX_AGE_S = 30.0
_MAX_CACHED_NAMESPACES = 128


@dataclass(frozen=True)
class AnomalyStats:
    """Rolling anomaly statistics persisted alongside the quarantine store."""

    sample_count: int
    dimensions: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ReferenceView:
    """What scoring needs from a namespace's reference window."""

    stats: AnomalyStats
    lengths: list[float]  # len(content) + len(detail), one per non-blank clean window entry
    tag_counts: list[float]


@dataclass(frozen=True, slots=True)
class _Row:
    key: tuple[str, str]  # (updated_at, id): list_entries ranks by this, descending
    clean: bool
    blank: bool
    char_len: float
    byte_len: float
    tag_count: float
    importance: float


@dataclass
class _Window:
    rows: list[_Row]  # ACTIVE rows, ranked, at most _REFERENCE_FETCH_LIMIT
    exhaustive: bool  # rows hold EVERY active row of the namespace
    token: NamespaceChangeToken
    seeded_at: float
    view: ReferenceView


@dataclass
class _Slot:
    lock: threading.Lock = field(default_factory=threading.Lock)
    window: _Window | None = None


_CACHE: OrderedDict[tuple[str, str], _Slot] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std_dev": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std_dev": math.sqrt(variance)}


def build_anomaly_stats(entries: list[MemoryEntry]) -> AnomalyStats:
    """Mean/std_dev of entry length (UTF-8 bytes), tag count, and importance."""
    return _stats([_row(entry) for entry in entries])


def fetch_reference(namespace: str, backend: StorageBackend) -> list[MemoryEntry]:
    # ACTIVE-only: retire/obsolete/archived entries no longer represent normal
    # write behaviour — including them skews mean/std and corrupts z-scores.
    return backend.list_entries(namespace=namespace, status=MemoryStatus.ACTIVE, limit=_REFERENCE_FETCH_LIMIT)


def reference_view(namespace: str, backend: StorageBackend) -> ReferenceView:
    """Return the current reference view of *namespace* (see the module docstring)."""
    token = backend.namespace_change_token(namespace)
    if not isinstance(token, NamespaceChangeToken):
        return _view(_ranked(fetch_reference(namespace, backend)))
    slot = _slot((token.store, namespace))
    with slot.lock:
        window = slot.window
        fresh = window is not None and time.monotonic() - window.seeded_at < _RESEED_MAX_AGE_S
        if window is not None and fresh and (window.token == token or _advance(window, namespace, backend, token)):
            return window.view
        # The token is read BEFORE the rows, so a write racing this read shows
        # up as a token change on the next call instead of being lost.
        rows = _ranked(fetch_reference(namespace, backend))
        slot.window = _Window(
            rows=rows,
            exhaustive=len(rows) < _REFERENCE_FETCH_LIMIT,
            token=token,
            seeded_at=time.monotonic(),
            view=_view(rows),
        )
        return slot.window.view


def reset_reference_cache() -> None:
    """Forget every cached window (tests; a process that swapped its database files)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _slot(key: tuple[str, str]) -> _Slot:
    with _CACHE_LOCK:
        slot = _CACHE.get(key)
        if slot is None:
            slot = _CACHE[key] = _Slot()
            while len(_CACHE) > _MAX_CACHED_NAMESPACES:
                _CACHE.popitem(last=False)
        else:
            _CACHE.move_to_end(key)
        return slot


def _advance(window: _Window, namespace: str, backend: StorageBackend, token: NamespaceChangeToken) -> bool:
    """Merge the change feed since ``window.token``; False when only a reseed is exact."""
    old = window.token
    if (
        token.store != old.store
        or token.delete_epoch != old.delete_epoch
        or token.insert_seq < old.insert_seq
        or token.top_updated_at < old.top_updated_at
    ):
        return False  # something was deleted (or the file was replaced)
    changed: list[MemoryEntry] | None = None
    for limit in _FEED_LIMITS:
        changed = backend.entries_changed_since(namespace, old, limit=limit)
        if changed is not None:
            break
    if changed is None:
        return False
    merged = _merge(window, changed)
    if merged is None:
        return False
    window.rows, window.exhaustive = merged
    window.token = token
    window.view = _view(window.rows)
    return True


def _merge(window: _Window, changed: list[MemoryEntry]) -> tuple[list[_Row], bool] | None:
    changed_ids = {entry.id for entry in changed}
    # Rows ranked below a full buffer's last row were never part of it.
    floor = window.rows[-1].key if window.rows and not window.exhaustive else None
    rows = [row for row in window.rows if row.key[1] not in changed_ids]
    for entry in changed:
        if entry.status != MemoryStatus.ACTIVE:  # use_enum_values: may be the plain string
            continue  # retired / archived: leaves the reference
        row = _row(entry)
        if floor is None or row.key >= floor:
            rows.append(row)
    rows.sort(key=lambda row: row.key, reverse=True)
    if len(rows) > _REFERENCE_FETCH_LIMIT:
        return rows[:_REFERENCE_FETCH_LIMIT], False
    if not window.exhaustive and len(rows) < _REFERENCE_FETCH_LIMIT:
        return None  # a ranked row left: rows below the buffer would move up
    return rows, window.exhaustive


def _row(entry: MemoryEntry) -> _Row:
    text = entry.content + entry.detail
    return _Row(
        key=(entry.updated_at.isoformat(), entry.id),
        clean=entry.metadata.get("quarantined") != "true" and entry.metadata.get("system_canary") != "true",
        blank=not text.strip(),
        char_len=float(len(entry.content) + len(entry.detail)),
        byte_len=float(len(text.encode("utf-8"))),
        tag_count=float(len(entry.tags)),
        importance=float(entry.importance),
    )


def _ranked(entries: list[MemoryEntry]) -> list[_Row]:
    return sorted((_row(entry) for entry in entries), key=lambda row: row.key, reverse=True)


def _stats(rows: list[_Row]) -> AnomalyStats:
    if not rows:
        return AnomalyStats(sample_count=0, dimensions={})
    return AnomalyStats(
        sample_count=len(rows),
        dimensions={
            "entry_length": series_stats([row.byte_len for row in rows]),
            "tag_count": series_stats([row.tag_count for row in rows]),
            "importance": series_stats([row.importance for row in rows]),
        },
    )


def _view(ranked: list[_Row]) -> ReferenceView:
    window = [row for row in ranked if row.clean][:_ROLLING_WINDOW]
    scored = [row for row in window if not row.blank]
    return ReferenceView(
        stats=_stats(window),
        lengths=[row.char_len for row in scored],
        tag_counts=[row.tag_count for row in scored],
    )
