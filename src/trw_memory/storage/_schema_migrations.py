"""The additive schema deltas after the v5 composite-key rebuild (schema 6..11, 13).

Split out of :mod:`trw_memory.storage._schema` to keep that module under the
350-effective-LOC gate (PRD-CORE-312 merge). Each function here is one
forward-only, idempotent ``_MIGRATIONS[n]`` delta; the REGISTRATION stays in
``_schema`` (one visible ordered table), which also re-exports every name so
``_schema._migrate_vN_*`` imports keep working. This module imports nothing
from ``_schema`` -- the DDL the deltas share with the bootstrap storm lives
here and ``_schema`` imports it -- so there is no load-order cycle. Schema 12
(PRD-CORE-332) lives in :mod:`trw_memory.storage._anchor_index`.
"""

from __future__ import annotations

import contextlib
import sqlite3

from trw_memory.embeddings.provenance import VectorProvenance

__all__ = [
    "CREATE_IDX_SOURCE_IDENTITY",
    "CREATE_IDX_VEC_SPACE",
    "CREATE_MEMORIES_FTS_ROWID",
    "_migrate_v6_vector_provenance",
    "_migrate_v7_retire_wiki_refs",
    "_migrate_v8_quarantine_review_namespace",
    "_migrate_v9_vector_space_key",
    "_migrate_v10_fts_rowid_map",
    "_migrate_v11_source_identity_index",
    "_migrate_v13_evidence_level",
]

#: Schema 11 (PRD-CORE-331 FR06, B71-102): ``ids_by_source`` (daemon dedup /
#: source-identity lookups) filters on ``(namespace, source_identity)`` with no
#: index over that pair, so it scans every row in the namespace. Plain column
#: index — the untrusted-store allowlist (``_untrusted_store._REFUSED``) only
#: refuses expression and partial indexes, so this is accepted unchanged.
CREATE_IDX_SOURCE_IDENTITY = (
    "CREATE INDEX IF NOT EXISTS idx_memories_namespace_source_identity ON memories(namespace, source_identity)"
)

#: Schema 9: covers the census's ``GROUP BY space_key`` and the gate's seek, and ``entry_id`` for the join.
CREATE_IDX_VEC_SPACE = "CREATE INDEX IF NOT EXISTS idx_vec_index_space ON vec_index(namespace, space_key, entry_id)"


def _migrate_v6_vector_provenance(cursor: sqlite3.Cursor) -> None:
    """Add proof storage without inventing provenance for existing vectors."""
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(vec_index)").fetchall()}
    if columns and "provenance_json" not in columns:
        cursor.execute("ALTER TABLE vec_index ADD COLUMN provenance_json TEXT DEFAULT NULL")


def _migrate_v7_retire_wiki_refs(cursor: sqlite3.Cursor) -> None:
    """Drop the retired ``wiki_refs`` sidecar table and its indexes (W10, trw-memory 4.0.0).

    ``trw_memory.wiki`` (the ``memory_wiki_lint`` tool, the ``wiki-lint`` CLI
    verb, and the ``query_wiki_*_refs`` backend methods) was removed with no
    replacement. This delta only drops the sidecar edge index it maintained;
    an entry's own ``metadata`` column — where a wiki payload actually lived —
    is untouched by a bare ``DROP TABLE`` and stays readable. Idempotent: a
    fresh database that never created ``wiki_refs`` (see
    ``_bootstrap_and_backfill``, which stopped creating it at this version)
    hits ``IF EXISTS`` no-ops on both statements.
    """
    cursor.execute("DROP INDEX IF EXISTS idx_wiki_refs_source")
    cursor.execute("DROP INDEX IF EXISTS idx_wiki_refs_target")
    cursor.execute("DROP TABLE IF EXISTS wiki_refs")


def _migrate_v8_quarantine_review_namespace(cursor: sqlite3.Cursor) -> None:
    """Add ``namespace`` to a pre-existing ``quarantine_reviews`` table (Q3).

    ``quarantine_reviews`` is created lazily, outside this module's DDL, by
    ``security._runtime_quarantine.append_review_log``/``get_status_history`` —
    so most databases (anything that never quarantined anything) simply do not
    have the table, and this is a no-op for them; a database created by the
    fixed code already gets the column from the updated ``CREATE TABLE IF NOT
    EXISTS`` those functions issue.

    For a database that DOES carry the pre-fix table: add the column (additive,
    idempotent — a duplicate-column ``OperationalError`` is suppressed like
    every other ALTER in this file), then best-effort backfill it from
    ``memories`` in the SAME database file (the quarantine DB's own store of
    still-quarantined rows) by matching on id.

    A row is backfilled ONLY when both hold:
      1. exactly one namespace currently names this id in ``memories`` here, and
      2. this id was quarantined (a ``"quarantined"`` review-log row) exactly
         ONCE, ever, in this database.

    (2) closes an adversarial-review counter-example to a naive "match on
    current `memories`" backfill: if namespace A's row for id ``x`` was
    approved (and so deleted from this quarantine DB) and namespace B *later*
    quarantined its OWN, unrelated ``x``, condition (1) alone would match
    uniquely to B — mislabelling A's historical reviewer and decision as B's,
    reproducing the exact cross-namespace leak this migration exists to close.
    Two review-log rows for the same id is exactly that signal, so such an id
    is left at ``''`` (unknown) rather than guessed, even where (1) holds.
    """
    tables = {
        str(row[0])
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'quarantine_reviews'"
        ).fetchall()
    }
    if "quarantine_reviews" not in tables:
        return
    columns = {str(row[1]) for row in cursor.execute("PRAGMA table_info(quarantine_reviews)").fetchall()}
    if "namespace" not in columns:
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute("ALTER TABLE quarantine_reviews ADD COLUMN namespace TEXT NOT NULL DEFAULT ''")
    cursor.execute(
        """
        UPDATE quarantine_reviews
        SET namespace = (SELECT namespace FROM memories WHERE memories.id = quarantine_reviews.learning_id)
        WHERE (namespace IS NULL OR namespace = '')
          AND (SELECT COUNT(*) FROM memories WHERE memories.id = quarantine_reviews.learning_id) = 1
          AND (
                SELECT COUNT(*) FROM quarantine_reviews r2
                WHERE r2.learning_id = quarantine_reviews.learning_id AND r2.decision = 'quarantined'
              ) = 1
        """
    )


def _migrate_v9_vector_space_key(cursor: sqlite3.Cursor) -> None:
    """Add and index ``vec_index.space_key``, then backfill it (B71-83).

    Each key is computed through :meth:`VectorProvenance.from_json`, the same
    validation every reader applies, so a NULL, malformed or wrongly-shaped
    record gets ``NULL`` and a valid one its claimed space's
    :attr:`EmbeddingSpace.key`. One streaming ``UPDATE`` through a SQL function
    rather than rowid chunks read into Python: nothing is held in memory, and
    chunking would not bound the transaction anyway (the whole storm is one).
    Idempotent (a rerun recomputes the same keys) and, like every delta, rolled
    back whole if interrupted, so the next open resumes from the unmigrated
    store. There is no downgrade: an older build refuses a schema-9 store
    (``SchemaDowngradeError``); the column is additive and it reads nothing else.
    """
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(vec_index)").fetchall()}
    if "provenance_json" not in columns:  # no vector table yet (``ensure_vec_table`` makes the v9 shape), or pre-v6
        return
    if "space_key" not in columns:
        cursor.execute("ALTER TABLE vec_index ADD COLUMN space_key TEXT DEFAULT NULL")
    cursor.execute(CREATE_IDX_VEC_SPACE)
    key = lambda raw: proof.space.key if (proof := VectorProvenance.from_json(raw)) else None  # noqa: E731
    cursor.connection.create_function("trw_space_key", 1, key, deterministic=True)
    cursor.execute("UPDATE vec_index SET space_key = trw_space_key(provenance_json)")


CREATE_MEMORIES_FTS_ROWID = """
CREATE TABLE IF NOT EXISTS memories_fts_rowid (
    namespace TEXT NOT NULL,
    id        TEXT NOT NULL,
    fts_rowid INTEGER NOT NULL,
    PRIMARY KEY (namespace, id)
)
"""


def _migrate_v10_fts_rowid_map(cursor: sqlite3.Cursor) -> None:
    """Create ``memories_fts_rowid`` (PRD-CORE-330).

    Additive only: this delta does not backfill from ``memories_fts`` because
    that table may not exist yet at this point in a fresh database's bootstrap
    (``ensure_fts_table`` runs later, after ``ensure_schema`` returns). For an
    EXISTING store, where ``memories_fts`` already holds rows, the backfill
    happens in ``ensure_fts_table`` itself (also idempotent, re-run on every
    open) rather than here, so the same one-shot backfill logic covers both
    "freshly created table" and "pre-existing table missing its map rows"
    without duplicating it in two places.
    """
    cursor.execute(CREATE_MEMORIES_FTS_ROWID)


def _migrate_v11_source_identity_index(cursor: sqlite3.Cursor) -> None:
    """Create ``idx_memories_namespace_source_identity`` (PRD-CORE-331 FR06, B71-102).

    Additive only: ``source_identity`` is a base column of ``CREATE_MEMORIES``
    (never a ``MIGRATE_COLS`` backfill), so it already exists on every store
    reaching this delta, and the plain ``CREATE INDEX IF NOT EXISTS`` is a
    no-op on a store that already has it (e.g. a fresh bootstrap that built it
    via ``MEMORIES_INDEXES``).
    """
    cursor.execute(CREATE_IDX_SOURCE_IDENTITY)


def _migrate_v13_evidence_level(cursor: sqlite3.Cursor) -> None:
    """Add the PRD-CORE-312-FR01 ``evidence_level`` column (schema 13; 12 is CORE-332).

    Additive-only and idempotent: an existing row keeps every value it had and
    reads back with ``evidence_level='unknown'`` (NFR02) -- never a silently
    promoted "verified". A database already carrying the column (fresh
    bootstrap via ``MIGRATE_COLS``) is a no-op. Registered explicitly (rather
    than relying only on the unconditional ``MIGRATE_COLS`` backfill inside
    ``_bootstrap_and_backfill``) to match this file's own precedent (schema 4)
    for every new column: the bootstrap storm is skipped entirely by
    ``ensure_schema``'s fast path once a store is already stamped at
    ``SCHEMA_VERSION``, so the version bump is what makes an ALREADY-stamped
    database (e.g. any store at 10 or 11 today) pick the column up at all.
    """
    columns = {str(row[1]) for row in cursor.execute("PRAGMA table_info(memories)").fetchall()}
    if "evidence_level" not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN evidence_level TEXT DEFAULT 'unknown'")
