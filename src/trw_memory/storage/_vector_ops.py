"""SQLite vec extension operations.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``SQLiteBackend.upsert_vector`` etc. become 1-line
delegators that pass instance state.

Helpers covering vec_index/vec_memories CRUD + KNN search:

- ``delete_vector_internal`` — internal "remove if present" used by both
  the public delete + by store/update flows that re-write a vector.
- ``delete_vector`` — public delete with vec-availability gate.
- ``purge_vectors_for`` — chunked bulk delete for whole-namespace purges.
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

import _thread
import contextlib
import hashlib
import sqlite3
import struct
from collections.abc import Sequence
from typing import Any

import structlog

from trw_memory._hype_ids import parent_of_hype_id
from trw_memory.embeddings.provenance import StoredVector, VectorProvenance
from trw_memory.storage._sql_utils import iter_bind_chunks

logger = structlog.get_logger(__name__)

# Namespace-scoped KNN over-fetch: sqlite-vec applies its ``k`` limit to the
# MATCH scan, so a namespace predicate cannot be pushed into the KNN itself.
# We request more candidates than ``top_k`` and post-filter by namespace, then
# truncate. The factor trades wasted scan against the risk of returning fewer
# than ``top_k`` in-namespace hits when other namespaces dominate the global
# nearest neighbours; the cap bounds the worst-case scan cost.
_NAMESPACE_OVERFETCH_FACTOR = 4
_NAMESPACE_OVERFETCH_CAP = 500


def _is_optional_vec_unavailable_error(exc: sqlite3.Error) -> bool:
    """Return True when SQLite cannot open sqlite-vec's ``vec0`` module."""
    message = str(exc).lower()
    return "no such module: vec0" in message or ("no such module" in message and "vec" in message)


def _is_vec_table_dimension_error(exc: sqlite3.Error) -> bool:
    """Return True when vec0 rejected a vector whose length the TABLE disagrees with.

    Distinct from the ``len(embedding) != dim`` pre-check in ``upsert_vector``, which
    compares against the backend's *configured* dim. A ``vec0`` table fixes its width
    at CREATE time, so a store whose table was created under a different
    ``embedding_dim`` — an embedding-model swap, or a database file shared with a
    differently-configured process — passes that pre-check and is then rejected by
    SQLite. Before this predicate existed the rejection propagated as an uncaught
    ``OperationalError`` and failed the whole store, which is the exact outcome the
    pre-check's own comment says it exists to prevent.
    """
    return "dimension mismatch" in str(exc).lower()


def _rollback_standalone_write(conn: Any, *, skip_commit: bool) -> None:
    """Undo a failed write unless its surrounding transaction owns rollback."""
    if not skip_commit:
        with contextlib.suppress(sqlite3.Error):
            conn.rollback()


def delete_vector_internal(conn: Any, entry_id: str, namespace: str, *, allow_unavailable: bool = True) -> None:
    """Remove the ``(namespace, entry_id)`` vector row (no-op if absent).

    PRD-CORE-245 FR02/FR03: ``vec_index`` is keyed ``UNIQUE (namespace,
    entry_id)`` under schema 5, so a bare id can address two rows. Caller holds
    the lock.
    """
    try:
        row = conn.execute(
            "SELECT rowid FROM vec_index WHERE namespace = ? AND entry_id = ?", (namespace, entry_id)
        ).fetchone()
        if row is None:
            return
        rowid: int = row[0]
        conn.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
        conn.execute("DELETE FROM vec_index WHERE rowid = ?", (rowid,))
    except sqlite3.Error as exc:
        if allow_unavailable and _is_optional_vec_unavailable_error(exc):
            logger.warning(
                "vector_index_unavailable",
                op="delete",
                entry_id=entry_id,
                detail=str(exc),
                hint="sqlite-vec virtual table unavailable; canonical memory row operation continues",
            )
            return
        raise


def purge_vectors_for(conn: Any, namespace: str, entry_ids: Sequence[str]) -> None:
    """Bulk-remove the vectors of *entry_ids* within *namespace*.

    The chunked counterpart to :func:`delete_vector_internal`, and the sibling
    of ``_crud_ops.purge_edges_for`` / ``purge_tag_postings_for``: the whole-
    namespace purge used to call ``delete_vector_internal`` once per entry,
    which is one SELECT plus two DELETEs per row — 3N statements where the
    other sidecar cleanups on the same code path issue one per bind chunk.
    ``vec_memories`` is addressed by rowid, so the rowids come from a subquery
    on ``vec_index`` and ``vec_index`` is then deleted by the same predicate.

    CONTRACT: mirrors its siblings — the caller MUST already hold
    ``backend._lock`` and own the commit, so the purge batches into the
    caller's outermost ``COMMIT`` and the whole namespace delete stays atomic.
    Callers gate on ``vec_available`` themselves.
    """
    if not entry_ids:
        return
    for chunk in iter_bind_chunks(list(entry_ids), reserved_bindings=1):
        placeholders = ",".join("?" for _ in chunk)
        conn.execute(
            "DELETE FROM vec_memories WHERE rowid IN ("  # noqa: S608 — placeholders is ? repeated; ids are parameterised
            f"SELECT rowid FROM vec_index WHERE namespace = ? AND entry_id IN ({placeholders}))",
            (namespace, *chunk),
        )
        conn.execute(
            f"DELETE FROM vec_index WHERE namespace = ? AND entry_id IN ({placeholders})",  # noqa: S608 — placeholders is ? repeated; ids are parameterised
            (namespace, *chunk),
        )


def delete_vector(
    conn: Any,
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    entry_id: str,
    namespace: str,
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
    try:
        with lock:
            before = conn.total_changes
            delete_vector_internal(conn, entry_id, namespace)
            if not skip_commit:
                conn.commit()
            return bool(conn.total_changes > before)
    except sqlite3.Error:
        _rollback_standalone_write(conn, skip_commit=skip_commit)
        raise


def vector_exists(conn: Any, *, vec_available: bool, entry_id: str, namespace: str) -> bool:
    """Return whether vec_index currently contains ``(namespace, entry_id)``."""
    if not vec_available:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM vec_index WHERE namespace = ? AND entry_id = ?", (namespace, entry_id)
        ).fetchone()
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
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    namespace: str | None = None,
) -> set[str]:
    """Return the set of entry IDs that currently have a stored vector.

    Empty set when sqlite-vec is unavailable. Single-query bulk lookup.

    When *namespace* is provided the lookup reads ``vec_index``'s own namespace
    column directly (PRD-CORE-245 FR02 added it), avoiding both the join to
    ``memories`` the old implementation needed and the full-table scan that
    would load every tenant's vector ids. ``namespace=None`` keeps the unscoped
    full scan used by the coverage probe.
    """
    if not vec_available:
        return set()
    try:
        with lock:
            if namespace is None:
                rows = conn.execute("SELECT entry_id FROM vec_index").fetchall()
            else:
                # INNER JOIN scopes to one namespace's rows. Vectors whose
                # canonical memory row is in another namespace (or absent) are
                # excluded, so the result never spans tenants.
                rows = conn.execute(
                    "SELECT entry_id FROM vec_index WHERE namespace = ?",
                    (namespace,),
                ).fetchall()
    except sqlite3.Error as exc:  # trw-fail-silent-allow: vec0 being absent is an expected optional-dependency state that degrades to BM25/keyword; a REAL SQL error (corruption, I/O, locked DB) is separated out and surfaced at warning instead of being folded into this empty return
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
                delete_vector_internal(conn, sibling_id, namespace, allow_unavailable=False)
            if not skip_commit:
                conn.commit()
        except sqlite3.Error:
            _rollback_standalone_write(conn, skip_commit=skip_commit)
            raise
    return len(sibling_ids)


def upsert_vector(
    conn: Any,
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    dim: int,
    entry_id: str,
    namespace: str,
    embedding: list[float],
    skip_commit: bool = False,
    provenance: VectorProvenance | None = None,
) -> None:
    """Insert or update the ``(namespace, entry_id)`` vector. No-op when sqlite-vec absent.

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
    if provenance is not None and not provenance.matches_vector(embedding):
        raise ValueError("vector provenance does not match the vector being stored")
    proof_json = provenance.to_json() if provenance is not None else None
    try:
        with lock:
            # Optional vector failures can be swallowed while an outer canonical
            # row transaction still commits. Preserve the old vector AND proof
            # with an operation savepoint, not an outer-transaction rollback.
            if not conn.in_transaction:
                conn.execute("BEGIN")
            conn.execute("SAVEPOINT trw_vector_upsert")
            try:
                conn.execute("INSERT OR IGNORE INTO vec_index(entry_id, namespace) VALUES(?, ?)", (entry_id, namespace))
                row = conn.execute(
                    "SELECT rowid FROM vec_index WHERE namespace = ? AND entry_id = ?", (namespace, entry_id)
                ).fetchone()
                rowid: int = row[0]
                conn.execute("UPDATE vec_index SET provenance_json = ? WHERE rowid = ?", (proof_json, rowid))
                conn.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO vec_memories(rowid, embedding) VALUES(?, ?)",
                    (rowid, emb_bytes),
                )
            except sqlite3.Error:
                conn.execute("ROLLBACK TO SAVEPOINT trw_vector_upsert")
                conn.execute("RELEASE SAVEPOINT trw_vector_upsert")
                raise
            conn.execute("RELEASE SAVEPOINT trw_vector_upsert")
            if not skip_commit:
                conn.commit()
    except sqlite3.Error as exc:
        _rollback_standalone_write(conn, skip_commit=skip_commit)
        if _is_optional_vec_unavailable_error(exc):
            logger.warning(
                "vector_index_unavailable",
                op="upsert",
                entry_id=entry_id,
                detail=str(exc),
                hint="sqlite-vec virtual table unavailable; canonical memory write is preserved",
            )
            return
        if _is_vec_table_dimension_error(exc):
            logger.warning(
                "vector_dimension_mismatch",
                op="upsert",
                entry_id=entry_id,
                expected_dim=dim,
                actual_dim=len(embedding),
                detail=str(exc),
                hint=(
                    "vec0 table width disagrees with this embedding; the table was created under a "
                    "different embedding_dim. Canonical memory write is preserved, vector skipped — "
                    "rebuild the vector index to restore dense retrieval for this store"
                ),
            )
            return
        raise
    logger.debug("vector_upserted", entry_id=entry_id)


def search_vectors(
    conn: Any,
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    dim: int,
    query_embedding: list[float],
    top_k: int = 25,
    namespace: str | None = None,
) -> list[tuple[str, float]]:
    """KNN search in vec_memories. Empty list when sqlite-vec absent.

    When *namespace* is provided results are scoped to that namespace, closing a
    cross-namespace data-isolation leak: ``vec_index`` carries no namespace
    column, so an unscoped KNN can surface entry ids whose canonical memory row
    belongs to another tenant. sqlite-vec requires the ``k = ?`` KNN limit on
    the MATCH scan itself (a namespace predicate cannot be pushed into that
    scan), so we OVER-FETCH ``k`` (k * over_fetch_factor, capped), JOIN to
    ``memories`` to filter by namespace, then truncate to ``top_k``. This keeps
    the requested count met as long as the namespace holds enough near neighbours
    within the over-fetch window. ``namespace=None`` keeps the legacy behaviour.
    """
    if not vec_available or top_k <= 0:
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
            if namespace is None:
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
            # Over-fetch then post-filter by namespace. The KNN k applies to the
            # MATCH scan; the namespace filter happens in the JOIN to memories.
            knn_k = min(max(top_k * _NAMESPACE_OVERFETCH_FACTOR, top_k), _NAMESPACE_OVERFETCH_CAP)
            rows = conn.execute(
                """
                SELECT vi.entry_id, vm.distance
                FROM vec_memories vm
                JOIN vec_index vi ON vm.rowid = vi.rowid
                JOIN memories mem ON mem.id = vi.entry_id
                WHERE vm.embedding MATCH ? AND k = ? AND mem.namespace = ?
                ORDER BY vm.distance
                """,
                (query_bytes, knn_k, namespace),
            ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows[:top_k]]
    except sqlite3.Error as exc:  # trw-fail-silent-allow: vec0 being absent is an expected optional-dependency state that degrades to BM25/keyword; a REAL SQL error (corruption, I/O, locked DB) is separated out and surfaced at warning instead of being folded into this empty return
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
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    entry_ids: list[str],
    namespace: str | None = None,
) -> dict[str, list[float]]:
    """Load vectors for requested IDs, optionally scoped before blob decoding.

    ``None`` preserves the legacy cross-namespace lookup (duplicate IDs remain
    ambiguous). Every other value, including the empty string, is an exact
    namespace predicate. Callers carrying authorized candidates should provide it.
    """
    if not vec_available or not entry_ids:
        return {}
    try:
        with lock:
            rows = []
            for chunk in iter_bind_chunks(entry_ids, reserved_bindings=int(namespace is not None)):
                placeholders = ", ".join(["?"] * len(chunk))
                sql = f"""
                    SELECT vi.entry_id, vm.embedding
                    FROM vec_memories vm
                    JOIN vec_index vi ON vm.rowid = vi.rowid
                    WHERE vi.entry_id IN ({placeholders})
                """  # noqa: S608
                params: list[object] = list(chunk)
                if namespace is not None:
                    sql += " AND vi.namespace = ?"
                    params.append(namespace)
                rows.extend(conn.execute(sql, params).fetchall())
    except sqlite3.Error as exc:  # trw-fail-silent-allow: vec0 being absent is an expected optional-dependency state that degrades to BM25/keyword; a REAL SQL error (corruption, I/O, locked DB) is separated out and surfaced at warning instead of being folded into this empty return
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


def get_vector_records(
    conn: Any,
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    entry_ids: list[str],
    namespace: str,
) -> dict[str, StoredVector]:
    """Read scoped vector bytes and proof together; never infer legacy proof.

    Read-only legacy layouts may lack the additive column. Their vectors remain
    available as unknown evidence. Malformed or stale proof is also unknown.
    """
    if not isinstance(namespace, str):
        raise TypeError("get_vector_records requires an explicit namespace string")
    if not vec_available or not entry_ids:
        return {}
    try:
        with lock:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(vec_index)").fetchall()}
            proof_column = "vi.provenance_json" if "provenance_json" in columns else "NULL"
            rows = []
            for chunk in iter_bind_chunks(entry_ids, reserved_bindings=1):
                placeholders = ", ".join("?" for _ in chunk)
                # Identifiers are fixed local literals; all caller values bind.
                sql = (
                    f"SELECT vi.entry_id, vm.embedding, {proof_column} "  # noqa: S608 -- fixed local SQL fragments only
                    "FROM vec_memories vm JOIN vec_index vi ON vm.rowid = vi.rowid "
                    f"WHERE vi.entry_id IN ({placeholders}) AND vi.namespace = ?"
                )
                rows.extend(conn.execute(sql, [*chunk, namespace]).fetchall())
    except sqlite3.Error:  # trw-fail-silent-allow: the optional-dependency case is already gated above (`if not vec_available: return {}`), so EVERY error reaching here is real and every one is surfaced at warning -- no error is folded silently into this empty return
        logger.warning("vector_record_load_error", exc_info=True)
        return {}
    records: dict[str, StoredVector] = {}
    for entry_id, raw, proof_json in rows:
        if raw is None:
            continue
        blob = bytes(raw)
        if len(blob) % 4 != 0:
            continue
        embedding = tuple(struct.unpack(f"{len(blob) // 4}f", blob))
        proof = VectorProvenance.from_json(proof_json)
        # Same check as ``proof.matches_vector(embedding)``: the proof digests the
        # float32 packing of the vector, which IS this blob, so hash it directly
        # (recall reads a whole candidate pool through here).
        if proof is not None and (
            len(embedding) != proof.space.dimensions or hashlib.sha256(blob).hexdigest() != proof.vector_sha256
        ):
            proof = None
        records[str(entry_id)] = StoredVector(embedding=embedding, provenance=proof)
    return records
