"""SQLite vec extension operations.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``SQLiteBackend.upsert_vector`` etc. become 1-line
delegators that pass instance state.

Helpers covering vec_index/vec_memories CRUD + KNN search:

- ``delete_vector_internal`` — internal "remove if present" used by both
  the public delete + by store/update flows that re-write a vector.
- ``delete_vector`` — public delete with vec-availability gate.
- ``vector_exists`` — single-row probe.
- ``existing_vector_ids`` — bulk set lookup for backfill skip.
- ``upsert_vector`` — INSERT OR IGNORE into vec_index, then DELETE +
  INSERT into vec_memories (idempotent upsert).
- ``search_vectors`` — KNN MATCH search via sqlite-vec.
- ``get_stored_embeddings`` — bulk lookup of packed embedding blobs.

Every helper short-circuits with the ``vec_available=False`` early-return
so callers don't need to gate.

Extracted as PRD-DIST-245 Phase 1 batch 85.
"""

from __future__ import annotations

import contextlib
import sqlite3
import struct
import threading
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _is_optional_vec_unavailable_error(exc: sqlite3.Error) -> bool:
    """Return True when SQLite cannot open sqlite-vec's ``vec0`` module."""
    message = str(exc).lower()
    return "no such module: vec0" in message or ("no such module" in message and "vec" in message)


def delete_vector_internal(conn: Any, entry_id: str) -> None:
    """Remove the vector row for *entry_id* (no-op if absent). Caller holds the lock."""
    try:
        row = conn.execute("SELECT rowid FROM vec_index WHERE entry_id = ?", (entry_id,)).fetchone()
        if row is None:
            return
        rowid: int = row[0]
        conn.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
        conn.execute("DELETE FROM vec_index WHERE rowid = ?", (rowid,))
    except sqlite3.Error as exc:
        if _is_optional_vec_unavailable_error(exc):
            logger.warning(
                "vector_index_unavailable",
                op="delete",
                entry_id=entry_id,
                detail=str(exc),
                hint="sqlite-vec virtual table unavailable; canonical memory row operation continues",
            )
            return
        raise


def delete_vector(
    conn: Any,
    lock: threading.Lock,
    *,
    vec_available: bool,
    entry_id: str,
    skip_commit: bool = False,
) -> bool:
    """Public vector-row deletion helper for warm-tier maintenance.

    When ``skip_commit`` is True the delete is staged but NOT committed — used
    when the caller is inside a backend ``transaction()`` block so the vector
    delete batches into the caller's outermost COMMIT instead of prematurely
    committing their open transaction. This mirrors the v0.9.1 ``_crud_ops``
    defer-commit fix and the ``upsert_vector`` ``skip_commit`` flag; deleting
    here unconditionally was the same premature-commit bug class missed in 0.9.1.
    """
    if not vec_available:
        return False
    with lock:
        before = conn.total_changes
        delete_vector_internal(conn, entry_id)
        if not skip_commit:
            conn.commit()
        return bool(conn.total_changes > before)


def vector_exists(conn: Any, *, vec_available: bool, entry_id: str) -> bool:
    """Return whether vec_index currently contains *entry_id*."""
    if not vec_available:
        return False
    try:
        row = conn.execute("SELECT 1 FROM vec_index WHERE entry_id = ?", (entry_id,)).fetchone()
        return row is not None
    except sqlite3.Error as exc:
        if _is_optional_vec_unavailable_error(exc):
            logger.warning(
                "vector_index_unavailable",
                op="exists",
                entry_id=entry_id,
                detail=str(exc),
                hint="sqlite-vec virtual table unavailable; treating vector row as absent",
            )
            return False
        raise


def existing_vector_ids(
    conn: Any,
    lock: threading.Lock,
    *,
    vec_available: bool,
) -> set[str]:
    """Return the set of entry IDs that currently have a stored vector.

    Empty set when sqlite-vec is unavailable. Single-query bulk lookup.
    """
    if not vec_available:
        return set()
    try:
        with lock:
            rows = conn.execute("SELECT entry_id FROM vec_index").fetchall()
    except sqlite3.Error as exc:
        # Real SQL error here (vec_available was already True) → surface at
        # warning so a bulk backfill doesn't silently re-embed everything on a
        # transient table error; only the vec0-absent case stays at debug.
        if _is_optional_vec_unavailable_error(exc):
            logger.debug("existing_vector_ids_query_failed", exc_info=True)
        else:
            logger.warning("existing_vector_ids_query_failed", exc_info=True)
        return set()
    return {str(row[0]) for row in rows}


def _hype_like_pattern(parent_id: str) -> str:
    """SQL LIKE pattern matching a parent's ``{parent_id}#hype{n}`` siblings.

    ``#`` is not a LIKE wildcard, and ``%`` after it captures every ``hype{n}``
    suffix. Parent ids are opaque app strings; a parent id that itself contained
    LIKE wildcards (``%``/``_``) could over-match, but parent ids here are
    engine-generated uuids/keys, so no ESCAPE clause is needed.
    """
    return f"{parent_id}#hype%"


def hype_sibling_ids(
    conn: Any,
    lock: threading.Lock,
    *,
    vec_available: bool,
    parent_id: str,
) -> list[str]:
    """Return the stored ``{parent_id}#hype{n}`` sibling ids for *parent_id*.

    PRD-CORE-195 FR05: enumerates a parent's HyPE siblings via a bounded
    ``LIKE`` scan on ``vec_index``. Empty list when sqlite-vec is unavailable.
    """
    if not vec_available:
        return []
    try:
        with lock:
            rows = conn.execute(
                "SELECT entry_id FROM vec_index WHERE entry_id LIKE ?",
                (_hype_like_pattern(parent_id),),
            ).fetchall()
    except sqlite3.Error as exc:
        if _is_optional_vec_unavailable_error(exc):
            logger.debug("hype_sibling_ids_query_failed", exc_info=True)
        else:
            logger.warning("hype_sibling_ids_query_failed", exc_info=True)
        return []
    return [str(row[0]) for row in rows]


def delete_hype_siblings(
    conn: Any,
    lock: threading.Lock,
    *,
    vec_available: bool,
    parent_id: str,
    skip_commit: bool = False,
) -> int:
    """Delete all ``{parent_id}#hype{n}`` sibling vectors for *parent_id*.

    PRD-CORE-195 FR05: idempotent purge used on forget + on UPDATE
    (purge-then-regenerate). Returns the number of sibling rows removed.
    No-op (returns 0) when sqlite-vec is unavailable. When ``skip_commit`` is
    True the deletes batch into the caller's open ``transaction()`` COMMIT —
    same defer-commit contract as :func:`delete_vector`.
    """
    if not vec_available:
        return 0
    sibling_ids = hype_sibling_ids(conn, lock, vec_available=vec_available, parent_id=parent_id)
    if not sibling_ids:
        return 0
    with lock:
        for sibling_id in sibling_ids:
            delete_vector_internal(conn, sibling_id)
        if not skip_commit:
            conn.commit()
    return len(sibling_ids)


def upsert_vector(
    conn: Any,
    lock: threading.Lock,
    *,
    vec_available: bool,
    dim: int,
    entry_id: str,
    embedding: list[float],
    skip_commit: bool = False,
) -> None:
    """Insert or update a vector in vec_memories. No-op when sqlite-vec absent.

    When ``skip_commit`` is True the write is staged but NOT committed — used
    when the caller is inside a backend ``transaction()`` block so the vector
    write commits atomically with the row write at the outermost COMMIT
    (mirrors the ``delete_vector_internal`` / ``delete_vector`` split). On the
    vec-unavailable fallback the connection-wide ``rollback()`` is likewise
    suppressed so an in-flight outer transaction is left intact for its owner.
    """
    if not vec_available:
        return
    if len(embedding) != dim:
        # A fixed-dim vec0 table cannot hold a wrong-length vector (e.g. an
        # embedding-model swap leaving config.embedding_dim stale). Skip the
        # vector write the same way the vec-unavailable path does: the canonical
        # row + BM25 still provide retrieval. struct.pack would otherwise raise
        # an uncaught struct.error and fail the whole store transaction.
        logger.warning(
            "vector_dimension_mismatch",
            op="upsert",
            entry_id=entry_id,
            expected_dim=dim,
            actual_dim=len(embedding),
            hint="embedding length != backend dim; canonical memory write is preserved, vector skipped",
        )
        return
    emb_bytes = struct.pack(f"{dim}f", *embedding)
    try:
        with lock:
            conn.execute("INSERT OR IGNORE INTO vec_index(entry_id) VALUES(?)", (entry_id,))
            row = conn.execute("SELECT rowid FROM vec_index WHERE entry_id = ?", (entry_id,)).fetchone()
            rowid: int = row[0]
            conn.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
            conn.execute(
                "INSERT INTO vec_memories(rowid, embedding) VALUES(?, ?)",
                (rowid, emb_bytes),
            )
            if not skip_commit:
                conn.commit()
    except sqlite3.Error as exc:
        if _is_optional_vec_unavailable_error(exc):
            if not skip_commit:
                # Standalone upsert: undo the partial vec writes. Inside a
                # transaction we must NOT rollback — that would discard the
                # owner's outer batch; leave cleanup to the outermost handler.
                with contextlib.suppress(sqlite3.Error):
                    conn.rollback()
            logger.warning(
                "vector_index_unavailable",
                op="upsert",
                entry_id=entry_id,
                detail=str(exc),
                hint="sqlite-vec virtual table unavailable; canonical memory write is preserved",
            )
            return
        raise
    logger.debug("vector_upserted", entry_id=entry_id)


def search_vectors(
    conn: Any,
    lock: threading.Lock,
    *,
    vec_available: bool,
    dim: int,
    query_embedding: list[float],
    top_k: int = 25,
) -> list[tuple[str, float]]:
    """KNN search in vec_memories. Empty list when sqlite-vec absent."""
    if not vec_available:
        return []
    if len(query_embedding) != dim:
        # A query vector whose length differs from the indexed dim (model swap)
        # cannot match the fixed-dim vec0 table. Degrade to "no dense hits" so
        # the caller falls back to BM25, rather than raising an uncaught
        # struct.error from the pack below.
        logger.debug(
            "vector_search_dimension_mismatch",
            expected_dim=dim,
            actual_dim=len(query_embedding),
        )
        return []
    query_bytes = struct.pack(f"{dim}f", *query_embedding)
    try:
        with lock:
            rows = conn.execute(
                """
                SELECT vi.entry_id, vm.distance
                FROM vec_memories vm
                JOIN vec_index vi ON vm.rowid = vi.rowid
                WHERE vm.embedding MATCH ? AND k = ?
                ORDER BY vm.distance
                """,
                (query_bytes, top_k),
            ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]
    except sqlite3.Error as exc:
        # Keep the graceful BM25-only fallback (return []), but surface a REAL
        # SQL error (corruption, I/O) at warning — only the expected
        # vec0-module-absent case stays at debug. Otherwise vector search silently
        # degrades with no operator signal (the compounding-pipeline silent-rot class).
        if _is_optional_vec_unavailable_error(exc):
            logger.debug("vector_search_error", exc_info=True)
        else:
            logger.warning("vector_search_error", exc_info=True)
        return []


def get_stored_embeddings(
    conn: Any,
    lock: threading.Lock,
    *,
    vec_available: bool,
    entry_ids: list[str],
) -> dict[str, list[float]]:
    """Load stored vectors for the requested entry IDs."""
    if not vec_available or not entry_ids:
        return {}
    placeholders = ", ".join(["?"] * len(entry_ids))
    sql = f"""
        SELECT vi.entry_id, vm.embedding
        FROM vec_memories vm
        JOIN vec_index vi ON vm.rowid = vi.rowid
        WHERE vi.entry_id IN ({placeholders})
    """  # noqa: S608 — placeholder count is derived from entry_ids length only; values bound separately
    try:
        with lock:
            rows = conn.execute(sql, entry_ids).fetchall()
    except sqlite3.Error as exc:
        # Match search_vectors: only the expected vec0-module-absent case stays
        # at debug. A REAL SQL error (corruption, I/O, locked DB) returns {} —
        # which a bulk-backfill caller reads as "no stored embeddings" and
        # re-embeds everything — so surface it at warning, not silently.
        if _is_optional_vec_unavailable_error(exc):
            logger.debug("vector_load_error", exc_info=True)
        else:
            logger.warning("vector_load_error", exc_info=True)
        return {}

    embeddings: dict[str, list[float]] = {}
    for row in rows:
        raw = row[1]
        if raw is None:
            continue
        blob = bytes(raw)
        if len(blob) % 4 != 0:
            logger.debug(
                "vector_load_skipped_invalid_blob",
                entry_id=str(row[0]),
                blob_len=len(blob),
            )
            continue
        dim_len = len(blob) // 4
        embeddings[str(row[0])] = list(struct.unpack(f"{dim_len}f", blob))
    return embeddings
