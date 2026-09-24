"""Legacy ``#hype{n}`` derived-vector cleanup, split off ``_vector_ops.py``.

Belongs to the ``sqlite_backend.py`` facade (via ``_vector_ops.py``'s
re-export) — moved out of ``_vector_ops.py`` (PRD-CORE-291 slice 3) when that
module crossed the effective-LOC ceiling. Covers the namespace-scoped
enumeration and atomic deletion of orphaned hype-derived vectors:

- ``hype_sibling_ids`` — public enumeration of a parent's noncanonical
  derived-vector siblings.
- ``delete_hype_siblings`` — delete those siblings inside the caller's
  transaction.
- ``_legacy_sibling_ids`` / ``_hype_like_pattern`` — shared query internals.

CONTRACT: :func:`delete_hype_siblings` calls ``delete_vector_internal``
through the ``_vector_ops`` module object (not a direct name import), so a
test that patches ``trw_memory.storage._vector_ops.delete_vector_internal``
still observes the patched behaviour here — the same seam
``test_hype_lifecycle.py::test_cleanup_failure_rolls_back_and_reopens``
already relies on.

``_vector_ops.py`` re-exports every public name here (imported there) so
every existing ``from trw_memory.storage._vector_ops import
delete_hype_siblings`` (and the ``_sqlite_backend_mixins.py`` import) call
site keeps working.
"""

from __future__ import annotations

import _thread
import sqlite3
from typing import Any

from trw_memory._hype_ids import parent_of_hype_id
from trw_memory.storage import _vector_ops


def _hype_like_pattern(parent_id: str) -> str:
    """SQL LIKE pattern matching a parent's ``{parent_id}#hype{n}`` siblings.

    Parent ids are opaque caller-supplied strings, so escape every SQLite LIKE
    metacharacter before appending the wildcard that captures ``hype{n}``.
    """
    escaped_parent_id = parent_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped_parent_id}#hype%"


def hype_sibling_ids(
    conn: Any,
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    parent_id: str,
    namespace: str,
) -> list[str]:
    """Enumerate only namespace-owned, noncanonical legacy derived vectors.

    Canonical membership, not suffix spelling, establishes ownership. Orphans
    remain for a future canonical-index rebuild. SQL failures must propagate.
    """
    if not vec_available:
        raise NotImplementedError("legacy vector cleanup unavailable: sqlite-vec is not available")
    with lock:
        return _legacy_sibling_ids(conn, parent_id=parent_id, namespace=namespace)


def _legacy_sibling_ids(conn: Any, *, parent_id: str, namespace: str) -> list[str]:
    """Caller holds the connection lock (and write transaction for deletion)."""
    if conn.execute("SELECT 1 FROM memories WHERE namespace = ? AND id = ?", (namespace, parent_id)).fetchone() is None:
        return []
    rows = conn.execute(
        "SELECT vi.entry_id FROM vec_index vi "
        "WHERE vi.namespace = ? AND vi.entry_id LIKE ? ESCAPE '\\' "
        "AND EXISTS (SELECT 1 FROM memories p WHERE p.id = ? AND p.namespace = vi.namespace) "
        "AND NOT EXISTS (SELECT 1 FROM memories m "
        "WHERE m.id = vi.entry_id AND m.namespace = vi.namespace)",
        (namespace, _hype_like_pattern(parent_id), parent_id),
    ).fetchall()
    return [str(row[0]) for row in rows if parent_of_hype_id(str(row[0])) == parent_id]


def delete_hype_siblings(
    conn: Any,
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    parent_id: str,
    namespace: str,
    skip_commit: bool = False,
) -> int:
    """Delete namespace-qualified legacy vectors inside the caller's transaction.

    The backend wrapper supplies a transaction even for standalone calls, so
    canonical membership cannot change between enumeration and deletion.
    """
    with lock:
        if not vec_available:
            raise NotImplementedError("legacy vector cleanup unavailable: sqlite-vec is not available")
        sibling_ids = _legacy_sibling_ids(conn, parent_id=parent_id, namespace=namespace)
        try:
            for sibling_id in sibling_ids:
                # Module-qualified lookup (not a direct-imported name binding)
                # so a monkeypatch of ``_vector_ops.delete_vector_internal``
                # is observed here at call time.
                _vector_ops.delete_vector_internal(conn, sibling_id, namespace, allow_unavailable=False)
            if not skip_commit:
                conn.commit()
        except sqlite3.Error:
            _vector_ops._rollback_standalone_write(conn, skip_commit=skip_commit)
            raise
    return len(sibling_ids)
