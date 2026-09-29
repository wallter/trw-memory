"""``anchor_postings``: the ``(namespace, file) -> entry`` index over ``memories.anchors`` (PRD-CORE-332).

A lesson's code anchors (``MemoryEntry.anchors``, at most 3, each naming a
repo-relative ``file``) live in the unindexed JSON ``anchors`` column. This
module owns the derived inverted index over that column, the shape of
``memory_tags``: one posting per distinct normalized anchor file per row.

- :func:`normalize_anchor_file` -- the ONE key definition; the write path, the
  migration backfill and (later) the read path all go through it, so they
  cannot disagree about what a file key is.
- :func:`replace_anchor_postings` -- re-point one row's postings (write path;
  called from ``_crud_index_ops._replace_postings`` inside the row write's own
  lock and transaction).
- :func:`rebuild_anchor_postings` -- drop every posting and re-derive them all
  from the column. Idempotent. The schema-12 migration uses it as its backfill,
  and the two recovery writers that restore rows with raw ``INSERT`` (salvage
  restore, cold-tier rebuild) call it before their commit.
- :func:`anchored_entries` -- the read (``SQLiteBackend.anchored_to``, FR03): one
  indexed lookup on ``(namespace, file)``.

Soundness scope of the rebuild: every row whose ``anchors`` parses as a JSON
list of objects gets exactly the postings of its objects' normalizable
``file`` strings. A row whose value is not such a list is skipped and counted
(``malformed_skipped``), never repaired and never raised. SQL ``NULL`` or an
empty string is "no anchors", not malformed. Whether an anchor still names an
existing file is ``anchor_validity``'s question, not this index's.

CONTRACT (write helpers): the caller holds the backend lock and owns the
commit, like every ``_crud_index_ops`` helper.
"""

from __future__ import annotations

import json
import posixpath
import sqlite3
from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple

import structlog

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryStatus

if TYPE_CHECKING:
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)

CREATE_ANCHOR_POSTINGS = """
CREATE TABLE IF NOT EXISTS anchor_postings (
    namespace TEXT NOT NULL,
    file      TEXT NOT NULL,
    entry_id  TEXT NOT NULL,
    PRIMARY KEY (namespace, file, entry_id)
) WITHOUT ROWID
"""

CREATE_IDX_ANCHOR_POSTINGS_ENTRY = (
    "CREATE INDEX IF NOT EXISTS idx_anchor_postings_entry ON anchor_postings(namespace, entry_id)"
)

_INSERT_POSTING = "INSERT OR IGNORE INTO anchor_postings(namespace, file, entry_id) VALUES(?, ?, ?)"

_BATCH = 1_000

#: How an anchored read ranks its rows (PRD-CORE-332 FR03); the interface default sorts the same way.
ANCHORED_ORDER = "importance DESC, updated_at DESC, id ASC"

#: The postings drive the read: ``CROSS JOIN`` fixes them as the outer loop, so each hit is one
#: ``memories`` primary-key probe. A plain ``id IN (...)`` let the planner walk the whole namespace
#: through an importance index instead (measured on an un-ANALYZEd store).
_ANCHORED_FROM = (
    "(SELECT entry_id AS anchored_id FROM anchor_postings WHERE namespace = ? AND file = ?) CROSS JOIN memories"
)


class BackfillCounts(NamedTuple):
    """What :func:`rebuild_anchor_postings` did."""

    rows_scanned: int
    postings_written: int
    malformed_skipped: int


def normalize_anchor_file(value: object) -> str | None:
    """The posting key for an anchor ``file``, or ``None`` when it cannot be one.

    ``posixpath.normpath`` (which also strips a leading ``./`` and collapses
    ``//``), case preserved. Empty, ``.``, absolute and ``..``-bearing values
    (checked on the raw components, so ``a/../b`` is refused rather than
    silently re-keyed to ``b``) yield no key. POSIX paths only.
    """
    if not isinstance(value, str) or not value.strip() or value.startswith("/"):
        return None
    if ".." in value.split("/"):
        return None
    key = posixpath.normpath(value)
    return None if key == "." else key


def _anchor_file(anchor: object) -> object:
    """The raw ``file`` of an ``Anchor`` model or of its serialized dict form."""
    return anchor.get("file") if isinstance(anchor, dict) else getattr(anchor, "file", None)


def anchor_files(anchors: Iterable[object]) -> list[str]:
    """The distinct normalized files of *anchors*, first occurrence first."""
    keys = (normalize_anchor_file(_anchor_file(anchor)) for anchor in anchors)
    return list(dict.fromkeys(key for key in keys if key is not None))


def replace_anchor_postings(conn: sqlite3.Connection, namespace: str, entry_id: str, anchors: Iterable[object]) -> None:
    """Re-point ``(namespace, entry_id)``'s postings at *anchors*' current files."""
    conn.execute("DELETE FROM anchor_postings WHERE namespace = ? AND entry_id = ?", (namespace, entry_id))
    rows = [(namespace, key, entry_id) for key in anchor_files(anchors)]
    if rows:
        conn.executemany(_INSERT_POSTING, rows)


def _parse_anchor_list(raw: object) -> list[object] | None:
    """The column's anchors as a list, ``[]`` for no value, ``None`` when malformed."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, (str, bytes)) else None
    except ValueError:  # trw-fail-silent-allow: a malformed row is counted by the caller and reported in anchor_postings_backfilled, never raised (PRD-CORE-332 FR02)
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        return None
    return parsed


def rebuild_anchor_postings(conn: sqlite3.Connection, *, trigger: str) -> BackfillCounts:
    """Delete every posting and re-derive them all from ``memories.anchors``; log the counts.

    Idempotent. *trigger* names the caller (``migration``, ``salvage_restore``,
    ``cold_rebuild``) in the ``anchor_postings_backfilled`` event.
    """
    conn.execute("DELETE FROM anchor_postings")
    scanned = written = malformed = 0
    # A second cursor on the same connection streams the rows, so a large store is never held in memory.
    source = conn.cursor()
    try:
        source.execute("SELECT namespace, id, anchors FROM memories")
        while batch := source.fetchmany(_BATCH):
            for namespace, entry_id, raw in batch:
                scanned += 1
                anchors = _parse_anchor_list(raw)
                if anchors is None:
                    malformed += 1
                    continue
                rows = [(namespace, key, entry_id) for key in anchor_files(anchors)]
                conn.executemany(_INSERT_POSTING, rows)
                written += len(rows)
    finally:
        source.close()
    counts = BackfillCounts(scanned, written, malformed)
    logger.info("anchor_postings_backfilled", trigger=trigger, **counts._asdict())
    return counts


def migrate_v12_anchor_postings(cursor: sqlite3.Cursor) -> None:
    """Schema 12 (PRD-CORE-332 FR02): create ``anchor_postings`` and backfill it from the column.

    Additive. Runs inside ``ensure_schema``'s one migration transaction, after
    its pre-bump snapshot, so an interruption leaves ``user_version`` and the
    store as they were; the ``user_version`` gate keeps it from re-running.
    """
    cursor.execute(CREATE_ANCHOR_POSTINGS)
    cursor.execute(CREATE_IDX_ANCHOR_POSTINGS_ENTRY)
    rebuild_anchor_postings(cursor.connection, trigger="migration")


def anchored_entries(
    backend: SQLiteBackend, namespace: str, file: str, *, status: MemoryStatus | None, limit: int
) -> list[MemoryEntry]:
    """Up to *limit* (at most ``MAX_RECALL_LIMIT``) rows of *namespace* anchored to *file*, by :data:`ANCHORED_ORDER`.

    *file* goes through :func:`normalize_anchor_file`; one it refuses reads nothing.
    The postings' primary key finds the ids and the ``memories`` key fetches them,
    so the read never walks the namespace. The caller holds a fresh connection.
    """
    from trw_memory.retrieval.recall_policy import MAX_RECALL_LIMIT
    from trw_memory.storage._query_ops import _execute_resilient

    key = normalize_anchor_file(file)
    if key is None or limit <= 0:
        return []
    where_sql = "memories.namespace = ? AND memories.id = anchored_id"
    params: list[object] = [namespace, key, namespace]
    if status is not None:
        where_sql += " AND status = ?"
        params.append(MemoryStatus(status).value)
    query = backend._fetch_query(
        where_sql=where_sql,
        params=params,
        order_by=ANCHORED_ORDER,
        limit=min(limit, MAX_RECALL_LIMIT),
        table=_ANCHORED_FROM,
    )
    sql, bound = query.build()
    try:
        with backend._lock:
            return _execute_resilient(backend, sql, bound, fetch_query=query)
    except (sqlite3.Error, ValueError, KeyError) as exc:
        raise StorageError(f"Failed to read anchored entries: {exc}", path=str(backend._db_path)) from exc
