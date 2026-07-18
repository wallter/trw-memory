"""DDL definitions and schema-management helpers for the SQLite backend.

Contains all ``CREATE TABLE``, ``CREATE INDEX``, and migration logic
extracted from :mod:`sqlite_backend` so that the backend module stays
focused on read/write operations.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------
#
# ``PRAGMA user_version`` gates the DDL/migration storm so it runs exactly
# once per database, then is skipped on every subsequent open (fast path).
#
# Adoption contract (trw-memory ships to real user databases in the wild):
#   * A database written by TODAY's released build carries user_version=0
#     (no prior build set it) yet may already hold every column. Opening it
#     with this build re-runs the fully-idempotent bootstrap once (all ALTERs
#     duplicate-suppressed => zero row mutation) and STAMPS it to
#     ``SCHEMA_VERSION`` — never a destructive rewrite.
#   * A FRESH database (user_version=0, no tables) migrates 0 -> current.
#   * A database whose user_version EXCEEDS ``SCHEMA_VERSION`` (an older build
#     opening a newer db) fails LOUDLY via :class:`SchemaDowngradeError`
#     BEFORE any DDL runs, rather than silently mis-reading it.
#
# ``SCHEMA_VERSION`` == 1 is the baseline shape produced by
# ``_bootstrap_and_backfill`` (all columns through PRD-CORE-194 validity +
# PRD-CORE-132 feedback counters). ``SCHEMA_VERSION`` == 2 is the PRD-CORE-181
# FR06 ``memory_model_v2_importance_type`` cutover: canonical ``importance`` +
# valid ``type`` (missing/empty ``type`` -> ``pattern``; conflicting
# impact/importance or an invalid ``type`` BLOCKS with no version bump). A
# user_version 0 database normalises through the v1 bootstrap FIRST, then the
# v2 delta. Bump this and register a delta in ``_MIGRATIONS`` for each future
# forward-only migration.
SCHEMA_VERSION = 3


class SchemaDowngradeError(RuntimeError):
    """A database was opened by a trw-memory build older than the one that wrote it.

    Raised when ``PRAGMA user_version`` exceeds :data:`SCHEMA_VERSION`. Failing
    loudly here prevents a stale build from silently mis-reading (and possibly
    corrupting) a database written by a newer schema.
    """


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    detail            TEXT DEFAULT '',
    tags              TEXT DEFAULT '[]',
    evidence          TEXT DEFAULT '[]',
    importance        REAL DEFAULT 0.5,
    status            TEXT DEFAULT 'active',
    recurrence        INTEGER DEFAULT 1,
    namespace         TEXT DEFAULT 'default',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    last_accessed_at  TEXT,
    valid_from        TEXT,
    invalid_from      TEXT,
    invalidated_by    TEXT,
    access_count      INTEGER DEFAULT 0,
    session_count     INTEGER DEFAULT 0,
    q_value           REAL DEFAULT 0.5,
    q_observations    INTEGER DEFAULT 0,
    source            TEXT DEFAULT 'agent',
    source_identity   TEXT DEFAULT '',
    client_profile    TEXT DEFAULT '',
    model_id          TEXT DEFAULT '',
    merged_from       TEXT DEFAULT '[]',
    consolidated_from TEXT DEFAULT '[]',
    consolidated_into TEXT,
    metadata          TEXT DEFAULT '{}',
    vector_clock      TEXT DEFAULT '{}',
    remote_id         TEXT,
    published_to_platform INTEGER DEFAULT 0,
    pending_delete    INTEGER DEFAULT 0,
    cross_validated   INTEGER DEFAULT 0,
    outcome_history   TEXT DEFAULT '[]',
    assertions        TEXT DEFAULT '[]',
    anchors           TEXT DEFAULT '[]',
    anchor_validity   REAL DEFAULT 1.0,
    type              TEXT DEFAULT 'pattern',
    nudge_line        TEXT DEFAULT '',
    expires_at        TEXT DEFAULT '',
    confidence        TEXT DEFAULT 'unverified',
    task_type         TEXT DEFAULT '',
    domain            TEXT DEFAULT '[]',
    phase_origin      TEXT DEFAULT '',
    phase_affinity    TEXT DEFAULT '[]',
    team_origin       TEXT DEFAULT '',
    protection_tier   TEXT DEFAULT 'normal',
    sessions_surfaced INTEGER DEFAULT 0,
    avg_rework_delta  TEXT DEFAULT NULL,
    outcome_correlation TEXT DEFAULT '',
    sync_hash         TEXT DEFAULT '',
    sync_seq          INTEGER DEFAULT 0,
    last_synced_at    TEXT,
    recall_count      INTEGER DEFAULT 0,
    helpful_count     INTEGER DEFAULT 0,
    unhelpful_count   INTEGER DEFAULT 0
)
"""

CREATE_IDX_NAMESPACE = "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)"
CREATE_IDX_STATUS = "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)"

# Composite indexes for the recall list / search ORDER BY clauses
# (F-007). Forward-only additive migration — IF NOT EXISTS, no
# destructive schema change. ``list_entries`` orders by updated_at DESC
# within a namespace; ``search`` orders by importance DESC, updated_at DESC.
CREATE_IDX_NS_UPDATED = "CREATE INDEX IF NOT EXISTS idx_memories_ns_updated ON memories(namespace, updated_at DESC)"
CREATE_IDX_NS_IMPORTANCE = (
    "CREATE INDEX IF NOT EXISTS idx_memories_ns_importance ON memories(namespace, importance DESC, updated_at DESC)"
)

# Enterprise-scale composite indexes (millions of entries). These cover the
# most common multi-predicate filters that the bare ``idx_memories_namespace``
# / ``idx_memories_status`` single-column indexes cannot serve efficiently.
#   * (namespace, status)            — list_entries filtering on both at once
#   * (namespace, status, importance) — tri-predicate min_importance recall
#   * (status, updated_at)           — status-only lifecycle sweeps by recency
# Forward-only additive migration — IF NOT EXISTS, no destructive change.
CREATE_IDX_NS_STATUS = "CREATE INDEX IF NOT EXISTS idx_memories_ns_status ON memories(namespace, status)"
CREATE_IDX_NS_STATUS_IMP = (
    "CREATE INDEX IF NOT EXISTS idx_memories_ns_status_imp ON memories(namespace, status, importance)"
)
CREATE_IDX_STATUS_UPDATED = "CREATE INDEX IF NOT EXISTS idx_memories_status_updated ON memories(status, updated_at)"

CREATE_GRAPH_EDGES = """
CREATE TABLE IF NOT EXISTS memory_graph_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    edge_type       TEXT NOT NULL,
    weight          REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    created_at      TEXT NOT NULL,
    edge_metadata   TEXT DEFAULT '{}',
    UNIQUE (source_id, target_id, edge_type)
)
"""

CREATE_IDX_MGE_SOURCE = "CREATE INDEX IF NOT EXISTS idx_mge_source ON memory_graph_edges(source_id, edge_type)"
CREATE_IDX_MGE_TARGET = "CREATE INDEX IF NOT EXISTS idx_mge_target ON memory_graph_edges(target_id, edge_type)"

CREATE_WIKI_REFS = """
CREATE TABLE IF NOT EXISTS wiki_refs (
    source_entry_id TEXT NOT NULL,
    source_slug     TEXT NOT NULL,
    target_slug     TEXT NOT NULL,
    ref_type        TEXT NOT NULL,
    label           TEXT DEFAULT '',
    bidirectional   INTEGER NOT NULL DEFAULT 1 CHECK (bidirectional IN (0, 1)),
    namespace       TEXT NOT NULL DEFAULT 'default',
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (source_entry_id, target_slug, ref_type),
    FOREIGN KEY (source_entry_id) REFERENCES memories(id) ON DELETE CASCADE
)
"""

CREATE_IDX_WIKI_REFS_SOURCE = "CREATE INDEX IF NOT EXISTS idx_wiki_refs_source ON wiki_refs(source_slug, namespace)"
CREATE_IDX_WIKI_REFS_TARGET = "CREATE INDEX IF NOT EXISTS idx_wiki_refs_target ON wiki_refs(target_slug, namespace)"

CREATE_NAMESPACES = """
CREATE TABLE IF NOT EXISTS memory_namespaces (
    namespace_id  TEXT PRIMARY KEY,
    team_id       TEXT,
    created_at    TEXT NOT NULL,
    expires_at    TEXT,
    status        TEXT NOT NULL DEFAULT 'active'
)
"""

CREATE_IDX_MN_STATUS = "CREATE INDEX IF NOT EXISTS idx_mn_status ON memory_namespaces(status, expires_at)"


# ---------------------------------------------------------------------------
# Schema bootstrap + migrations
# ---------------------------------------------------------------------------


def _bootstrap_and_backfill(cursor: sqlite3.Cursor) -> None:
    """Run the full idempotent DDL + column-backfill storm (v0 -> v1).

    Every statement is idempotent (``CREATE ... IF NOT EXISTS`` or an
    ``ALTER ... ADD COLUMN`` whose duplicate-column ``OperationalError`` is
    suppressed), so this normalises a database of ANY historical column count
    to the v1 shape without introspection and without mutating existing rows.
    Does NOT commit — the caller (:func:`ensure_schema`) owns the transaction.
    """
    cursor.execute(CREATE_MEMORIES)
    cursor.execute(CREATE_GRAPH_EDGES)
    cursor.execute(CREATE_WIKI_REFS)
    cursor.execute(CREATE_NAMESPACES)

    # Migration: rename columns from older schema versions.
    # Must run BEFORE index creation since indexes reference new names.
    _rename_cols = [
        ("memories", "impact", "importance"),
        ("memories", "expires", "expires_at"),
        ("memory_graph_edges", "relation", "edge_type"),
    ]
    for table, old_name, new_name in _rename_cols:
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")

    cursor.execute(CREATE_IDX_NAMESPACE)
    cursor.execute(CREATE_IDX_STATUS)
    cursor.execute(CREATE_IDX_NS_UPDATED)
    cursor.execute(CREATE_IDX_NS_IMPORTANCE)
    cursor.execute(CREATE_IDX_NS_STATUS)
    cursor.execute(CREATE_IDX_NS_STATUS_IMP)
    cursor.execute(CREATE_IDX_STATUS_UPDATED)
    cursor.execute(CREATE_IDX_MGE_SOURCE)
    cursor.execute(CREATE_IDX_MGE_TARGET)
    cursor.execute(CREATE_IDX_WIKI_REFS_SOURCE)
    cursor.execute(CREATE_IDX_WIKI_REFS_TARGET)

    # Migration: add missing columns to memory_namespaces
    for col_name, col_def in [
        ("team_id", "TEXT"),
        ("expires_at", "TEXT"),
        ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ]:
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute(f"ALTER TABLE memory_namespaces ADD COLUMN {col_name} {col_def}")

    cursor.execute(CREATE_IDX_MN_STATUS)

    # Migration: add new columns for sync + graph (Sprint 37)
    _migrate_cols = [
        ("metadata", "TEXT DEFAULT '{}'"),
        ("vector_clock", "TEXT DEFAULT '{}'"),
        ("remote_id", "TEXT"),
        ("published_to_platform", "INTEGER DEFAULT 0"),
        ("pending_delete", "INTEGER DEFAULT 0"),
        ("cross_validated", "INTEGER DEFAULT 0"),
        ("outcome_history", "TEXT DEFAULT '[]'"),
        ("assertions", "TEXT DEFAULT '[]'"),
    ]
    # Migration: add provenance columns (PRD-CORE-099)
    _migrate_cols += [
        ("client_profile", "TEXT DEFAULT ''"),
        ("model_id", "TEXT DEFAULT ''"),
    ]
    # Migration: add PRD-CORE-110 typed entry fields
    _migrate_cols += [
        ("session_count", "INTEGER DEFAULT 0"),
        ("type", "TEXT DEFAULT 'pattern'"),
        ("nudge_line", "TEXT DEFAULT ''"),
        ("expires_at", "TEXT DEFAULT ''"),
        ("confidence", "TEXT DEFAULT 'unverified'"),
        ("task_type", "TEXT DEFAULT ''"),
        ("domain", "TEXT DEFAULT '[]'"),
        ("phase_origin", "TEXT DEFAULT ''"),
        ("phase_affinity", "TEXT DEFAULT '[]'"),
        ("team_origin", "TEXT DEFAULT ''"),
        ("protection_tier", "TEXT DEFAULT 'normal'"),
    ]
    # Migration: add PRD-CORE-111 anchor fields
    _migrate_cols += [
        ("anchors", "TEXT DEFAULT '[]'"),
        ("anchor_validity", "REAL DEFAULT 1.0"),
    ]
    # Migration: add PRD-CORE-108 outcome attribution fields
    _migrate_cols += [
        ("sessions_surfaced", "INTEGER DEFAULT 0"),
        ("avg_rework_delta", "TEXT DEFAULT NULL"),
        ("outcome_correlation", "TEXT DEFAULT ''"),
    ]
    # Migration: add PRD-INFRA-051 sync pipeline delta tracking
    _migrate_cols += [
        ("sync_hash", "TEXT DEFAULT ''"),
        ("sync_seq", "INTEGER DEFAULT 0"),
        ("last_synced_at", "TEXT"),
    ]
    # Migration: add PRD-CORE-132 feedback lifecycle counters
    _migrate_cols += [
        ("recall_count", "INTEGER DEFAULT 0"),
        ("helpful_count", "INTEGER DEFAULT 0"),
        ("unhelpful_count", "INTEGER DEFAULT 0"),
    ]
    # Migration: add PRD-CORE-194 bi-temporal validity fields. Additive-only,
    # nullable; absent valid_from = open validity (back-filled to created_at
    # on read by the row mapper / model validator, never by a rewrite). No
    # destructive ALTER, zero existing rows mutated on read (NFR01).
    _migrate_cols += [
        ("valid_from", "TEXT"),
        ("invalid_from", "TEXT"),
        ("invalidated_by", "TEXT"),
    ]
    for col_name, col_def in _migrate_cols:
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")

    with contextlib.suppress(sqlite3.OperationalError):
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_sync_seq ON memories(sync_seq)")

    # Migration: add edge_metadata column for PRD-CORE-107 typed edges
    with contextlib.suppress(sqlite3.OperationalError):
        cursor.execute("ALTER TABLE memory_graph_edges ADD COLUMN edge_metadata TEXT DEFAULT '{}'")


# Forward-only migration deltas keyed by the target user_version. v1 has no
# per-version delta because ``_bootstrap_and_backfill`` normalises any legacy
# shape to v1 without one (an arbitrary historical column count cannot be
# expressed as a single ordered diff). ``_MIGRATIONS[2]`` is the PRD-CORE-181
# FR06 ``memory_model_v2_importance_type`` SQLite delta — imported lazily below
# to avoid an import cycle with :mod:`_memory_model_v2`.
_MIGRATIONS: dict[int, Callable[[sqlite3.Cursor], None]] = {}


_CANONICAL_MEMORY_TYPES = frozenset({"incident", "pattern", "convention", "hypothesis", "workaround"})
_CANONICAL_CONFIDENCE = frozenset({"unverified", "low", "medium", "high", "verified"})


def _migrate_v3_legacy_enums(cursor: sqlite3.Cursor) -> None:
    """Canonicalise legacy enum values without discarding their exact form.

    Older producers persisted ``type='gotcha'`` and numeric confidence scores.
    Both are valid historical semantics but cannot populate today's enums.  The
    canonical value is deliberately conservative while the original value is
    retained in metadata, making this migration lossless and idempotent.
    """
    columns = {str(row[1]) for row in cursor.execute("PRAGMA table_info(memories)").fetchall()}
    if "metadata" not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN metadata TEXT DEFAULT '{}'")
    rows = cursor.execute("SELECT id, type, confidence, metadata FROM memories").fetchall()
    for entry_id, type_raw, confidence_raw, metadata_raw in rows:
        type_value = str(type_raw or "").strip()
        confidence_value = str(confidence_raw or "").strip()
        invalid_type = type_value not in _CANONICAL_MEMORY_TYPES
        invalid_confidence = confidence_value not in _CANONICAL_CONFIDENCE
        if not invalid_type and not invalid_confidence:
            continue

        try:
            parsed_metadata = json.loads(str(metadata_raw or "{}"))
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed_metadata = {"legacy_metadata_raw": str(metadata_raw)}
        metadata = parsed_metadata if isinstance(parsed_metadata, dict) else {"legacy_metadata_raw": metadata_raw}

        canonical_type = type_value
        if invalid_type:
            metadata.setdefault("legacy_memory_type", type_value)
            # Historical ``gotcha`` rows in TRW are audit findings; incident is
            # the taxonomy's loss-minimising canonical category. Unknown future
            # values remain readable as patterns with their exact value retained.
            canonical_type = "incident" if type_value == "gotcha" else "pattern"

        canonical_confidence = confidence_value
        if invalid_confidence:
            metadata.setdefault("legacy_confidence", confidence_value)
            # A numeric score is not proof of the enum's validation lifecycle.
            canonical_confidence = "unverified"

        cursor.execute(
            "UPDATE memories SET type = ?, confidence = ?, metadata = ? WHERE id = ?",
            (canonical_type, canonical_confidence, json.dumps(metadata, sort_keys=True), str(entry_id)),
        )


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create core tables, run column migrations, and build indexes.

    Gated by ``PRAGMA user_version`` so the idempotent DDL/migration storm runs
    exactly once per database and is skipped (fast path) on every later open.

    * ``user_version > SCHEMA_VERSION`` -> :class:`SchemaDowngradeError` raised
      BEFORE any DDL (older build must not silently mis-read a newer db).
    * ``user_version == SCHEMA_VERSION`` -> return immediately (fast path).
    * otherwise -> bootstrap/backfill to v1, apply any registered
      ``_MIGRATIONS`` deltas up to ``SCHEMA_VERSION``, then stamp the version.

    Safe to call multiple times (all operations are idempotent).

    Args:
        conn: An open SQLite connection.

    Raises:
        SchemaDowngradeError: the database was written by a newer trw-memory.
    """
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise SchemaDowngradeError(
            f"database schema user_version={current} is newer than this "
            f"trw-memory build supports (max {SCHEMA_VERSION}); refusing to "
            "open it to avoid silent mis-reads — upgrade trw-memory."
        )
    if current == SCHEMA_VERSION:
        return

    cursor = conn.cursor()
    try:
        # v0 -> v1: normalise any legacy shape via the idempotent storm. Also
        # runs for an already-fully-migrated wild db (user_version=0) — every
        # statement is a no-op there, so it is stamped, never rewritten.
        _bootstrap_and_backfill(cursor)
        for version in range(current + 1, SCHEMA_VERSION + 1):
            migrate = _MIGRATIONS.get(version)
            if migrate is not None:
                migrate(cursor)
        # PRAGMA cannot bind parameters; SCHEMA_VERSION is a trusted int literal.
        cursor.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    except Exception:
        # A blocked migration (e.g. MigrationBlocked from the v2 delta) must
        # leave NO partial writes and NO version bump — roll the whole DDL /
        # migration storm back before re-raising (PRD-CORE-181-FR06).
        with contextlib.suppress(sqlite3.Error):
            conn.rollback()
        raise
    finally:
        cursor.close()


# Register the PRD-CORE-181 FR06 v2 delta. Imported here (module bottom) rather
# than at the top so :mod:`_memory_model_v2` can lazily import ``ensure_schema``
# / ``SCHEMA_VERSION`` from this module without a circular import at load time.
from trw_memory.storage._memory_model_v2 import (  # noqa: E402
    migrate_sqlite_importance_type as _migrate_v2_memory_model,
)

_MIGRATIONS[2] = _migrate_v2_memory_model
_MIGRATIONS[3] = _migrate_v3_legacy_enums


CREATE_MEMORIES_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    content,
    detail,
    tags,
    tokenize='unicode61 remove_diacritics 1'
)
"""


def ensure_fts_table(conn: sqlite3.Connection) -> bool:
    """Create and pre-populate the memories_fts FTS5 virtual table.

    Returns True when FTS5 is available and the table is ready.
    Safe to call multiple times (idempotent). On first call, bulk-imports
    all existing memories rows so legacy entries are searchable immediately.
    """
    try:
        conn.execute(CREATE_MEMORIES_FTS)
        row = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()
        if row is not None and int(row[0]) == 0:
            conn.execute(
                "INSERT INTO memories_fts(id, content, detail, tags) "
                "SELECT id, content, COALESCE(detail, ''), COALESCE(tags, '[]') FROM memories"
            )
        conn.commit()
        return True
    except sqlite3.OperationalError:
        return False


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Create the ``vec_memories`` virtual table and its companion index.

    Args:
        conn: An open SQLite connection (with sqlite-vec loaded).
        dim: Embedding vector dimension.
    """
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(embedding float[{dim}])")
    # Companion table to map rowid <-> entry_id
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vec_index (rowid INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT UNIQUE NOT NULL)"
    )
    conn.commit()
