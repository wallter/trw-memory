"""Check a SQLite file trw-memory did not write before any backend opens it.

``SQLiteBackend`` reads a store while it opens it (``PRAGMA quick_check``, the migration
snapshot's ``SELECT 1 FROM memories``, the migrations themselves). A caller-supplied file can make
such a read endless: a ``memories`` VIEW over a non-terminating recursive CTE, or a trigger that
runs one. For a copy a checkout supplies to the shared daemon, that pins a worker for every tenant
(rc8, B71-74).

So such a file is registered with ``_connection.untrusted_store`` (a deadline and a value cap on
every connection opened on it) and refused unless its schema holds only what trw-memory creates: tables and indexes, plus the ``memories_fts`` (fts5) and
``vec_memories`` (vec0) virtual tables. Views and triggers carry statements. CHECK constraints,
generated columns and expression or partial indexes carry expressions that ``quick_check`` and the
migrations' writes evaluate, and one expression can allocate a huge value in a single step, which no
progress handler interrupts (rc8 C12); so a CHECK other than trw-memory's own two is refused, and so
is any of the others. The schema is classified in SQL, so no schema text of the copy's (which
can be very large) reaches Python. The file's ``quick_check`` then runs, and a pass is recorded, so
the backend opened next, under the same deadline, does not scan it again.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from trw_memory.exceptions import StorageError
from trw_memory.storage._connection import connect, mark_verified

#: SQLite stores the CREATE text with ``IF NOT EXISTS`` removed. A virtual table is admitted only as
#: trw-memory's own ``_schema.CREATE_MEMORIES_FTS`` and ``ensure_vec_table`` write it (any integer
#: width, since a checkout's vectors need not match the daemon's), options included:
#: an fts5 option such as ``prefix`` multiplies the index the backend's open builds, in steps no
#: progress handler or length limit reaches (rc8 C12).
_REFUSED = """SELECT type || ' ' || substr(name, 1, 64) FROM sqlite_master AS m
WHERE type IN ('view', 'trigger') OR (sql LIKE 'CREATE VIRTUAL%'
    AND sql NOT IN ('CREATE VIRTUAL TABLE memories_fts USING fts5(
    id UNINDEXED,
    namespace UNINDEXED,
    content,
    detail,
    tags,
    tokenize=''unicode61 remove_diacritics 1''
)', 'CREATE VIRTUAL TABLE vec_memories USING vec0(embedding float[' || CAST(substr(sql, 62) AS INTEGER) || '])'))
    OR replace(replace(replace(sql, 'CHECK (weight >= 0.0 AND weight <= 1.0)', ''),
        'CHECK (bidirectional IN (0, 1))', ''), 'verification_checked_at', '') LIKE '%check%'
    OR (type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL%' AND EXISTS (SELECT 1 FROM pragma_table_xinfo(m.name) WHERE hidden > 1))
    OR (type = 'index' AND (sql LIKE '%where%' OR EXISTS (SELECT 1 FROM pragma_index_xinfo(m.name) WHERE cid = -2)))
ORDER BY 1 LIMIT 20"""


def verify_untrusted_store(db_path: Path) -> None:
    """Raise ``StorageError`` unless *db_path* holds only trw-memory's own schema and passes
    ``quick_check``. Call it inside ``untrusted_store``: nothing else bounds these reads."""
    conn = connect(db_path, dbapi=sqlite3, timeout=0.0, check_same_thread=False, cached_statements=0)
    try:
        if refused := [row[0] for row in conn.execute(_REFUSED)]:
            raise StorageError(f"{db_path.name} holds schema objects trw-memory never creates: {', '.join(refused)}")
        if [tuple(row) for row in conn.execute("PRAGMA quick_check")] != [("ok",)]:
            raise StorageError(f"{db_path.name} fails its integrity check")
    finally:
        conn.close()
    mark_verified(db_path)
