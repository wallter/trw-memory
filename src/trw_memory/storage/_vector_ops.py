"""SQLite vec extension operations.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``SQLiteBackend.upsert_vector`` etc. become 1-line
delegators that pass instance state.

7 helpers covering vec_index/vec_memories CRUD + KNN search:

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

import sqlite3
import struct
import threading
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def delete_vector_internal(conn: Any, entry_id: str) -> None:
    """Remove the vector row for *entry_id* (no-op if absent). Caller holds the lock."""
    row = conn.execute("SELECT rowid FROM vec_index WHERE entry_id = ?", (entry_id,)).fetchone()
    if row is None:
        return
    rowid: int = row[0]
    conn.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
    conn.execute("DELETE FROM vec_index WHERE rowid = ?", (rowid,))


def delete_vector(
    conn: Any,
    lock: threading.Lock,
    *,
    vec_available: bool,
    entry_id: str,
) -> bool:
    """Public vector-row deletion helper for warm-tier maintenance."""
    if not vec_available:
        return False
    with lock:
        before = conn.total_changes
        delete_vector_internal(conn, entry_id)
        conn.commit()
        return bool(conn.total_changes > before)


def vector_exists(conn: Any, *, vec_available: bool, entry_id: str) -> bool:
    """Return whether vec_index currently contains *entry_id*."""
    if not vec_available:
        return False
    row = conn.execute("SELECT 1 FROM vec_index WHERE entry_id = ?", (entry_id,)).fetchone()
    return row is not None


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
    except sqlite3.Error:
        logger.debug("existing_vector_ids_query_failed", exc_info=True)
        return set()
    return {str(row[0]) for row in rows}


def upsert_vector(
    conn: Any,
    lock: threading.Lock,
    *,
    vec_available: bool,
    dim: int,
    entry_id: str,
    embedding: list[float],
) -> None:
    """Insert or update a vector in vec_memories. No-op when sqlite-vec absent."""
    if not vec_available:
        return
    emb_bytes = struct.pack(f"{dim}f", *embedding)
    with lock:
        conn.execute("INSERT OR IGNORE INTO vec_index(entry_id) VALUES(?)", (entry_id,))
        row = conn.execute("SELECT rowid FROM vec_index WHERE entry_id = ?", (entry_id,)).fetchone()
        rowid: int = row[0]
        conn.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
        conn.execute(
            "INSERT INTO vec_memories(rowid, embedding) VALUES(?, ?)",
            (rowid, emb_bytes),
        )
        conn.commit()
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
    except sqlite3.Error:
        logger.debug("vector_search_error", exc_info=True)
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
    except sqlite3.Error:
        logger.debug("vector_load_error", exc_info=True)
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
