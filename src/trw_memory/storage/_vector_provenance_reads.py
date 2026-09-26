"""Vector-provenance read queries, split off ``_vector_ops.py``.

Belongs to the ``sqlite_backend.py`` facade (via ``_vector_ops.py``'s
re-export) — moved out of ``_vector_ops.py`` (PRD-CORE-291 slice 3) when that
module crossed the effective-LOC ceiling. Both helpers read
``vec_index.provenance_json`` (never write it) and classify or verify it with
:class:`~trw_memory.embeddings.provenance.VectorProvenance`:

- ``vector_space_census`` — count a namespace's stored vectors by the
  embedding space their provenance CLAIMS (no re-hash against the blob).
- ``get_vector_records`` — read scoped vector bytes and proof together,
  re-hashing the blob to invalidate a claim that disagrees with its bytes.

``_vector_ops.py`` re-exports both (imported there) so every existing
``from trw_memory.storage._vector_ops import get_vector_records`` call site
keeps working.
"""

from __future__ import annotations

import _thread
import hashlib
import sqlite3
import struct
from typing import Any

import structlog

from trw_memory.embeddings.provenance import EmbeddingSpace, StoredVector, VectorProvenance
from trw_memory.storage._sql_utils import iter_bind_chunks

logger = structlog.get_logger(__name__)


def vector_space_census(
    conn: Any,
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    namespace: str,
) -> dict[EmbeddingSpace | None, int] | None:
    """Count *namespace*'s rows that have a stored vector, by the embedding space its provenance claims.

    Only vectors of existing ``memories`` rows count, and each row once per space (C12 rc4): an orphan
    vector, or a second vector of one row, must not stand in for a row that has none.

    Reads only ``vec_index.provenance_json`` -- no vector blob is loaded -- and
    classifies each row with :meth:`VectorProvenance.from_json`, so a NULL,
    malformed or wrongly-shaped record counts under ``None`` (unknown space),
    never as any real space. Keys compare the FULL :class:`EmbeddingSpace`
    identity. This is the provenance CLAIM: unlike :func:`get_vector_records`
    it does not re-hash blobs against ``vector_sha256``, so a claim that
    disagrees with its bytes is counted under the claimed space.

    ``None`` means the census could not be taken (sqlite-vec unavailable or a
    SQL error): callers must not read it as "no stale vectors".
    """
    if not isinstance(namespace, str):
        raise TypeError("vector_space_census requires an explicit namespace string")
    if not vec_available:
        return None
    try:
        with lock:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(vec_index)").fetchall()}
            proof_column = "v.provenance_json" if "provenance_json" in columns else "NULL"
            rows = conn.execute(
                f"SELECT v.entry_id, {proof_column} FROM vec_index v "  # noqa: S608 -- fixed local SQL fragment only
                "JOIN memories m ON m.namespace = v.namespace AND m.id = v.entry_id WHERE v.namespace = ?",
                (namespace,),
            ).fetchall()
    except sqlite3.Error:  # trw-fail-silent-allow: None is the typed "no census" signal; logged at warning
        logger.warning("vector_space_census_error", exc_info=True)
        return None
    census: dict[EmbeddingSpace | None, set[str]] = {}
    for entry_id, raw in rows:
        proof = VectorProvenance.from_json(raw)
        census.setdefault(proof.space if proof is not None else None, set()).add(entry_id)
    return {space: len(ids) for space, ids in census.items()}


def get_vector_records(
    conn: Any,
    lock: _thread.LockType | _thread.RLock,
    *,
    vec_available: bool,
    entry_ids: list[str],
    namespace: str,
    strict: bool = False,
) -> dict[str, StoredVector]:
    """Read scoped vector bytes and proof together; never infer legacy proof.

    Read-only legacy layouts may lack the additive column. Their vectors remain
    available as unknown evidence. Malformed or stale proof is also unknown.
    A failed read returns no records unless *strict*, which raises it: an
    unread vector must not pass for an absent one where absence decides.
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
        if strict:
            raise
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
