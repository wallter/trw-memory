"""Parsing, the parsed-row cache, and the append-log layout of the warm JSONL sidecar.

Split from ``_warm`` (effective-LOC gate).

**Layout (2026-09-18).** The sidecar is an append log: a row whose id already
appears supersedes the earlier row, and the live view is the LAST row per id,
positioned where that last row sits. That is exactly the order the previous
rewrite-in-place writer produced (an updated row moved to the end), so every
reader sees the same logical rows as before. Rows without an id are all kept.
Superseded and corrupt lines are compaction debt (``ParsedSidecar.dead``); a
writer rewrites the file once that debt exceeds the live row count (or
:data:`COMPACT_MIN_DEAD`), so the file stays within ~2x its live size and each
appended row pays O(1) amortized compaction bytes.

Why: recall mirrors ``last_accessed_at`` for the k rows it returned into this
file. As a rewrite that cost a ``json.dumps`` of every row plus O(file) bytes
per recall (~15 ms at 420 rows, ~127 ms at 5,000). An append is O(k).

**Cache.** Parsed rows are cached against the file's ``(mtime_ns, size,
inode)``. Readers take no file lock: every writer changes that key (an append
grows the size, a compaction replaces the inode), so another process's write
is always seen as a miss. A writer re-seeds the cache, while still holding the
sidecar's RMW lock, with exactly what a fresh parse of the new bytes returns:
after an append it extends the cached parse in place (O(rows appended); a
copy-on-write dict copy was ~2.5 ms at 20,000 rows), and in-process readers
take a snapshot of the rows under the parse's own lock.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path
from typing import cast

import structlog

logger = structlog.get_logger(__name__)

SidecarKey = tuple[int, int, int]
SidecarRows = list[tuple[int, dict[str, object]]]
#: Live-row index: the row's id, or ``("", line)`` for an id-less row (never superseded).
RowIndex = dict[object, tuple[int, dict[str, object]]]

#: Minimum dead-line count before a writer compacts, so small sidecars are not
#: rewritten on every other write.
COMPACT_MIN_DEAD = 256

#: Entry fields recall's access bookkeeping changes (``increment_recall_access``).
#: An update that differs from the live row ONLY in these is appended; any other
#: change rewrites the file, so a superseded row never retains stale content.
ACCESS_FIELDS = frozenset({"last_accessed_at", "access_count", "recall_count", "sync_seq", "last_synced_at"})


def _row_key(line_number: int, rec: dict[str, object]) -> object:
    return str(rec.get("id", "")) or ("", line_number)


def _without_access_fields(rec: dict[str, object]) -> dict[str, object]:
    entry = rec.get("entry")
    if not isinstance(entry, dict):
        return rec
    return {**rec, "entry": {k: v for k, v in entry.items() if k not in ACCESS_FIELDS}}


def _row_namespace(rec: dict[str, object]) -> str | None:
    """The namespace an id-bearing row's entry payload names, if any."""
    entry = rec.get("entry")
    if not rec.get("id") or not isinstance(entry, dict):
        return None
    namespace = entry.get("namespace")
    return namespace if isinstance(namespace, str) else None


def access_only_change(old: dict[str, object], new: dict[str, object]) -> bool:
    """Return whether *new* differs from *old* in access bookkeeping only (or not at all)."""
    return _without_access_fields(old) == _without_access_fields(new)


class ParsedSidecar:
    """The live rows of one sidecar version plus what a writer needs to extend it.

    ``next_line`` is the 1-based line an appended row lands on once a torn tail
    (``torn_tail``: the file does not end in a newline) has been terminated.
    ``dead`` counts non-blank lines that are not live rows. A writer that
    appended to the file this describes calls :meth:`extend` (in place, O(rows
    appended)); readers take :attr:`rows`, a snapshot, so the two never race.
    """

    def __init__(self, index: RowIndex, next_line: int, torn_tail: bool, dead: int) -> None:
        self._index = index
        self.next_line = next_line
        self.torn_tail = torn_tail
        self.dead = dead
        self._lock = threading.Lock()
        # Live rows per entry namespace, so a containment check is O(namespaces).
        self._namespaces: Counter[str] = Counter(
            ns for _line, rec in index.values() if (ns := _row_namespace(rec)) is not None
        )

    @property
    def rows(self) -> SidecarRows:
        with self._lock:
            return list(self._index.values())

    def __len__(self) -> int:
        return len(self._index)

    def rows_except(self, omit: frozenset[str]) -> SidecarRows:
        """Live rows whose id is not in *omit*, in file order.

        The difference is a C-level set operation; only the kept rows are
        touched, so a caller omitting almost every row pays for the rest.
        """
        if not omit:
            return self.rows
        with self._lock:
            kept = [self._index[key] for key in self._index.keys() - omit]
        kept.sort(key=lambda row: row[0])
        return kept

    def names_other_namespace(self, namespace: str) -> bool:
        """Whether any live id-bearing row's entry names a namespace other than *namespace*."""
        with self._lock:
            return any(count and ns != namespace for ns, count in self._namespaces.items())

    def live(self, entry_id: str) -> dict[str, object] | None:
        with self._lock:
            hit = self._index.get(entry_id) if entry_id else None
        return hit[1] if hit is not None else None

    def compaction_due_after(self, rows: list[dict[str, object]]) -> bool:
        """Whether appending *rows* (unique ids) would leave more debt than :func:`_compaction_due` allows."""
        with self._lock:
            superseded = sum(1 for rec in rows if str(rec.get("id", "")) in self._index)
        live = len(self._index) + len(rows) - superseded
        return _compaction_due(self.dead + superseded, live)

    def merged(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """The live rows once *rows* are applied, in file order (for a rewrite; O(file))."""
        replaced = {str(rec.get("id", "")) for rec in rows} - {""}
        kept = [rec for key, (_line, rec) in self._snapshot_items() if key not in replaced]
        return kept + rows

    def _snapshot_items(self) -> list[tuple[object, tuple[int, dict[str, object]]]]:
        with self._lock:
            return list(self._index.items())

    def extend(self, rows: list[dict[str, object]]) -> None:
        """Apply *rows* as appended at :attr:`next_line` (tail terminated first); O(len(rows))."""
        with self._lock:
            for offset, rec in enumerate(rows):
                line_number = self.next_line + offset
                key = _row_key(line_number, rec)
                superseded = self._index.pop(key, None)
                if superseded is not None:
                    self.dead += 1
                    if (old_ns := _row_namespace(superseded[1])) is not None:
                        self._namespaces[old_ns] -= 1
                self._index[key] = (line_number, rec)
                if (new_ns := _row_namespace(rec)) is not None:
                    self._namespaces[new_ns] += 1
            self.next_line += len(rows)
            self.torn_tail = False


def _compaction_due(dead: int, live: int) -> bool:
    return dead > max(live, COMPACT_MIN_DEAD)


def sidecar_key(sidecar: Path) -> SidecarKey | None:
    """Return the cache key for *sidecar*, or ``None`` when it cannot be stat'd."""
    try:
        stat = sidecar.stat()
    except OSError:  # trw-fail-silent-allow: no key means "do not cache"; the caller parses or invalidates
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _skip(sidecar: Path, line_number: int, error_class: str) -> None:
    logger.warning(
        "warm_tier_sidecar_corrupt_record_skipped",
        path=str(sidecar),
        line_number=line_number,
        error_class=error_class,
    )


def parse_sidecar(sidecar: Path) -> ParsedSidecar:
    """Parse the sidecar from disk (see ``WarmTierStore._iter_sidecar_records``)."""
    data = sidecar.read_bytes()
    segments = data.split(b"\n")
    index: RowIndex = {}
    non_blank = 0
    for line_number, byte_line in enumerate(segments, start=1):
        if not byte_line.strip():
            continue
        non_blank += 1
        try:
            line_s = byte_line.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            _skip(sidecar, line_number, type(exc).__name__)
            continue
        if not line_s:
            continue
        try:
            rec = json.loads(line_s)
        except json.JSONDecodeError as exc:
            _skip(sidecar, line_number, type(exc).__name__)
            continue
        if not isinstance(rec, dict):
            # Structurally valid JSON that is not a record object (e.g. a bare
            # list or scalar) cannot satisfy the row schema; treat it as corrupt
            # so callers never have to guard ``rec.get(...)``.
            _skip(sidecar, line_number, "NotAnObject")
            continue
        row = cast("dict[str, object]", rec)
        key = _row_key(line_number, row)
        index.pop(key, None)  # a later row supersedes: re-insert at its position
        index[key] = (line_number, row)
    torn_tail = bool(data) and not data.endswith(b"\n")
    next_line = len(segments) + 1 if torn_tail else len(segments)
    return ParsedSidecar(index, next_line, torn_tail, non_blank - len(index))


class SidecarCache:
    """One sidecar's parsed rows, valid only while the file keeps its key."""

    def __init__(self) -> None:
        self._entry: tuple[SidecarKey, ParsedSidecar] | None = None
        self._lock = threading.Lock()

    def get(self, key: SidecarKey | None) -> ParsedSidecar | None:
        with self._lock:
            entry = self._entry
        if key is None or entry is None or entry[0] != key:
            return None
        return entry[1]

    def put(self, key: SidecarKey | None, parsed: ParsedSidecar) -> None:
        with self._lock:
            self._entry = (key, parsed) if key is not None else None

    def invalidate(self) -> None:
        with self._lock:
            self._entry = None

    def record_rewrite(self, sidecar: Path, rows: list[dict[str, object]]) -> None:
        """Re-seed after the caller replaced the file with *rows*, one JSON line each.

        Must be called while the caller still holds the sidecar's RMW lock.
        *rows* must be what parsing those lines yields (a parsed or
        ``json.loads``-round-tripped dict per line); the cache owns them after.
        """
        parsed = ParsedSidecar({}, 1, False, 0)
        parsed.extend(rows)
        self.put(sidecar_key(sidecar), parsed)

    def record_append(self, sidecar: Path, parsed: ParsedSidecar, rows: list[dict[str, object]]) -> None:
        """Re-seed after the caller appended *rows* to the file *parsed* describes.

        Same locking and row contract as :meth:`record_rewrite`. *parsed* is
        extended in place, so it must not be used to describe the old file after.
        """
        parsed.extend(rows)
        self.put(sidecar_key(sidecar), parsed)
