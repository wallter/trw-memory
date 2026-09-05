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
from pathlib import Path

import structlog

from trw_memory.storage._schema_backup import _main_database_path, snapshot_before_migration

logger = structlog.get_logger(__name__)

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
#
# ``SCHEMA_VERSION`` == 4 is PRD-CORE-231-FR02's additive ``verification_status``
# column. IMPORTANT: adding a column to ``_migrate_cols`` alone is NOT enough for
# an already-stamped database — ``ensure_schema`` short-circuits on
# ``user_version == SCHEMA_VERSION`` and never re-runs the backfill storm. Every
# new column therefore needs BOTH the ``_migrate_cols`` entry (fresh/legacy
# bootstrap) AND a registered ``_MIGRATIONS`` delta plus this bump.
#
# ``SCHEMA_VERSION`` == 5 is PRD-CORE-245: namespace becomes a containment
# boundary. ``memories`` is re-keyed on ``PRIMARY KEY (namespace, id)``, every
# id-referencing sidecar (``memories_fts``, ``vec_index``,
# ``memory_graph_edges``) gains the namespace discriminator, the inverted tag
# index ``memory_tags`` replaces the materialised ``tag_cooccurrence`` edges,
# and PRD-CORE-244's column changes ride the same rebuild (``anchor_validity``
# becomes nullable, ``verification_checked_at`` is added, and
# ``sessions_surfaced`` / ``avg_rework_delta`` / ``outcome_correlation`` are
# dropped). It is registered ONCE, in :mod:`trw_memory.storage._schema_v5`.
SCHEMA_VERSION = 5

#: The highest schema version whose delta DROPS or RENAMES rather than adding.
#: ``ensure_schema`` snapshots the store before migrating a database below this
#: and not above it, so an additive delta does not pay for a whole-file copy.
_LAST_DESTRUCTIVE_SCHEMA_VERSION = 5


class SchemaDowngradeError(RuntimeError):
    """A database was opened by a trw-memory build older than the one that wrote it.

    Raised when ``PRAGMA user_version`` exceeds :data:`SCHEMA_VERSION`. Failing
    loudly here prevents a stale build from silently mis-reading (and possibly
    corrupting) a database written by a newer schema.
    """


class SchemaLockError(RuntimeError):
    """The migration write lock could not be taken, so nothing was migrated.

    ``ensure_schema`` decides whether to migrate from a ``PRAGMA user_version``
    read taken INSIDE a ``BEGIN IMMEDIATE`` transaction, because that read is
    only authoritative while no other opener can commit. When the lock cannot
    be acquired within the connection's ``busy_timeout`` the version is simply
    unknown — and "unknown" must never be resolved to "already current", which
    would hand back a connection whose schema this build has not verified. The
    open is refused instead, with no writes and ``user_version`` unchanged.
    """


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id                TEXT NOT NULL,
    content           TEXT NOT NULL,
    detail            TEXT DEFAULT '',
    tags              TEXT DEFAULT '[]',
    evidence          TEXT DEFAULT '[]',
    importance        REAL DEFAULT 0.5,
    status            TEXT DEFAULT 'active',
    recurrence        INTEGER DEFAULT 1,
    namespace         TEXT NOT NULL DEFAULT 'default',
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
    anchor_validity   REAL DEFAULT NULL,
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
    sync_hash         TEXT DEFAULT '',
    sync_seq          INTEGER DEFAULT 0,
    last_synced_at    TEXT,
    recall_count      INTEGER DEFAULT 0,
    helpful_count     INTEGER DEFAULT 0,
    unhelpful_count   INTEGER DEFAULT 0,
    verification_status TEXT DEFAULT NULL,
    verification_checked_at TEXT DEFAULT '',
    PRIMARY KEY (namespace, id)
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
CREATE_IDX_SYNC_SEQ = "CREATE INDEX IF NOT EXISTS idx_memories_sync_seq ON memories(sync_seq)"

#: Every index declared over ``memories``. The v5 rebuild drops and renames the
#: table, which drops its indexes with it, so both the bootstrap storm and
#: :mod:`trw_memory.storage._schema_v5` replay this ONE list — a new index added
#: here is automatically recreated by the migration.
MEMORIES_INDEXES: tuple[str, ...] = (
    CREATE_IDX_NAMESPACE,
    CREATE_IDX_STATUS,
    CREATE_IDX_NS_UPDATED,
    CREATE_IDX_NS_IMPORTANCE,
    CREATE_IDX_NS_STATUS,
    CREATE_IDX_NS_STATUS_IMP,
    CREATE_IDX_STATUS_UPDATED,
    CREATE_IDX_SYNC_SEQ,
)

CREATE_GRAPH_EDGES = """
CREATE TABLE IF NOT EXISTS memory_graph_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace       TEXT NOT NULL DEFAULT 'default',
    source_id       TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    edge_type       TEXT NOT NULL,
    weight          REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    created_at      TEXT NOT NULL,
    edge_metadata   TEXT DEFAULT '{}',
    UNIQUE (namespace, source_id, target_id, edge_type)
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
    FOREIGN KEY (namespace, source_entry_id) REFERENCES memories(namespace, id) ON DELETE CASCADE
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

# PRD-CORE-245 FR02: one vector per (namespace, entry_id), not per bare id —
# a single-column UNIQUE would let a store in namespace B destroy the vector
# of namespace A's row with the same id.
CREATE_VEC_INDEX = """
CREATE TABLE IF NOT EXISTS vec_index (
    rowid     INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id  TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    UNIQUE (namespace, entry_id)
)
"""

# PRD-CORE-245 FR07: inverted tag index. Replaces the 98,288 materialised
# ``tag_cooccurrence`` edges with a (namespace, tag, entry_id) posting list the
# bounded derivation in :mod:`trw_memory.retrieval.tag_derivation` queries on
# demand. ``WITHOUT ROWID`` because the whole row IS the key — measured 7.80 MiB
# against the 16.04 MiB of edge rows it replaces.
CREATE_MEMORY_TAGS = """
CREATE TABLE IF NOT EXISTS memory_tags (
    namespace  TEXT NOT NULL,
    tag        TEXT NOT NULL,
    entry_id   TEXT NOT NULL,
    PRIMARY KEY (namespace, tag, entry_id)
) WITHOUT ROWID
"""

CREATE_IDX_MEMORY_TAGS_ENTRY = "CREATE INDEX IF NOT EXISTS idx_memory_tags_entry ON memory_tags(namespace, entry_id)"


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
    cursor.execute(CREATE_MEMORY_TAGS)

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
    cursor.execute(CREATE_IDX_MEMORY_TAGS_ENTRY)

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
        ("anchor_validity", "REAL DEFAULT NULL"),
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
    # Migration: add PRD-CORE-231-FR02 persisted verification verdict.
    # Additive-only, nullable; a pre-migration row reads back as
    # ``verification_status=None`` (no adverse verdict recorded).
    # PRD-CORE-244-FR03 adds the companion ``verification_checked_at`` stamp;
    # "" means no verification pass has ever examined this entry.
    _migrate_cols += [
        ("verification_status", "TEXT DEFAULT NULL"),
        ("verification_checked_at", "TEXT DEFAULT ''"),
    ]
    for col_name, col_def in _migrate_cols:
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")

    # Indexes over ``memories`` are built LAST: several of them name a column
    # that the backfill above is what adds to a legacy table (idx_memories_sync_seq
    # over ``sync_seq``), so creating them earlier fails on exactly the databases
    # the backfill exists to normalise.
    for statement in MEMORIES_INDEXES:
        with contextlib.suppress(sqlite3.OperationalError):
            cursor.execute(statement)

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


def _migrate_v4_verification_status(cursor: sqlite3.Cursor) -> None:
    """Add the PRD-CORE-231-FR02 ``verification_status`` column.

    Additive-only and idempotent: existing rows keep every value they had and
    read back with ``verification_status=None`` ("no adverse verdict recorded").
    A database already carrying the column (fresh bootstrap) is a no-op.
    """
    columns = {str(row[1]) for row in cursor.execute("PRAGMA table_info(memories)").fetchall()}
    if "verification_status" not in columns:
        cursor.execute("ALTER TABLE memories ADD COLUMN verification_status TEXT DEFAULT NULL")


def _log_migration_applied(
    conn: sqlite3.Connection,
    *,
    from_version: int,
    backup_path: Path | None,
) -> None:
    """Emit the one operator-facing record that a store was rewritten.

    Names the file, the version transition, the snapshot that can restore it,
    and the row census — enough for an operator to reconcile "why did my MCP
    server start failing" against "which file changed, and where is the copy".
    Namespace NAMES are deliberately absent (NFR03): the census is logged as
    shape, never as labels.
    """
    try:
        rows = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        namespaces = int(conn.execute("SELECT COUNT(DISTINCT namespace) FROM memories").fetchone()[0])
    except sqlite3.Error:
        rows, namespaces = -1, -1
    logger.info(
        "schema_migration_applied",
        database=str(_main_database_path(conn) or ":memory:"),
        from_version=from_version,
        to_version=SCHEMA_VERSION,
        backup=str(backup_path) if backup_path is not None else None,
        rows=rows,
        namespaces=namespaces,
        detail=(
            "Other processes holding this file open are still running the "
            "previous trw-memory build and will fail on write until they are "
            "restarted."
        ),
    )


def _user_version(conn: sqlite3.Connection) -> int:
    """Read ``PRAGMA user_version`` off *conn* as an int."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _guard_downgrade(current: int) -> None:
    """Refuse a store written by a build newer than this one.

    Migrations are forward-only, so a version above :data:`SCHEMA_VERSION` can
    never become readable by waiting: it is checked on the cheap pre-read (so
    an upgrade race fails fast with the actionable message instead of blocking
    on a write lock) and again under the lock (so a newer build that migrates
    the file while this one waits is still refused rather than rebuilt over).
    """
    if current <= SCHEMA_VERSION:
        return
    raise SchemaDowngradeError(
        f"this store is at schema {current}; this trw-memory build only "
        f"understands schema {SCHEMA_VERSION}. A newer build has already "
        "migrated the file. Restart this process (for an MCP client, "
        "restart the MCP server or run /mcp to reconnect) so it loads the "
        "newer trw-memory, or upgrade trw-memory. Refusing to open the "
        "store to avoid silent mis-reads."
    )


def _acquire_migration_lock(conn: sqlite3.Connection) -> bool:
    """Take the store's single write lock before the migration decision is made.

    Returns whether THIS call opened the transaction, so the caller only ends a
    transaction it owns. A caller that already holds one keeps it: the lock
    cannot be taken twice, and stealing the caller's transaction boundary would
    commit its work early.

    ``BEGIN IMMEDIATE`` blocks for the connection's ``busy_timeout`` (30 s on
    the standard open profile) and then fails. That failure is surfaced as
    :class:`SchemaLockError`, never absorbed: falling through would leave the
    caller deciding the migration from a version read that another process is
    concurrently invalidating, which is the whole defect this lock exists to
    close.
    """
    if conn.in_transaction:
        logger.warning(
            "schema_migration_lock_delegated",
            database=str(_main_database_path(conn) or ":memory:"),
            detail=(
                "ensure_schema was called inside a caller-owned transaction, so it "
                "could not take the migration write lock itself; exactly-once "
                "application depends on that transaction being a write transaction."
            ),
        )
        return False
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.Error as exc:
        database = str(_main_database_path(conn) or ":memory:")
        logger.warning("schema_migration_lock_unavailable", database=database, error=str(exc))
        raise SchemaLockError(
            f"could not take the migration write lock on {database} within this "
            f"connection's busy_timeout ({exc}); refusing to open the store, because "
            f"whether it still needs the schema {SCHEMA_VERSION} migration cannot be "
            "established while another process holds the lock. Retry the open once "
            "the concurrent writer finishes."
        ) from exc
    return True


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create core tables, run column migrations, and build indexes.

    Gated by ``PRAGMA user_version`` so the idempotent DDL/migration storm runs
    exactly once per database and is skipped (fast path) on every later open.

    The gate is double-checked at the SQLite level. ``user_version`` is read
    once OUTSIDE any transaction as a cheap filter, and — when that read says a
    migration may be due — again INSIDE a ``BEGIN IMMEDIATE`` transaction,
    which is the only read that decides anything. Without the second read every
    concurrent opener that sampled the pre-migration version re-runs the whole
    storm: six stdio servers booting against one store right after an upgrade
    measured six full schema-5 rebuilds, converging only because the rebuild
    happens to be idempotent (PRD-CORE-244-NFR02).

    * ``user_version > SCHEMA_VERSION`` -> :class:`SchemaDowngradeError` raised
      BEFORE any DDL (older build must not silently mis-read a newer db).
    * ``user_version == SCHEMA_VERSION`` -> return immediately (fast path).
    * otherwise -> take the write lock, re-read the version, and either return
      (another opener already migrated) or bootstrap/backfill to v1, apply any
      registered ``_MIGRATIONS`` deltas up to ``SCHEMA_VERSION``, and stamp it.

    Safe to call multiple times (all operations are idempotent).

    Args:
        conn: An open SQLite connection.

    Raises:
        SchemaDowngradeError: the database was written by a newer trw-memory.
        SchemaLockError: the migration write lock could not be acquired, so
            whether a migration is due is unknown. Nothing was migrated.
        SchemaBackupError: a destructive delta was due but the pre-migration
            snapshot could not be written -- or whether one was needed could
            not even be established. Nothing was migrated and ``user_version``
            is unchanged.
    """
    precheck = _user_version(conn)
    _guard_downgrade(precheck)
    if precheck == SCHEMA_VERSION:
        return

    owns_lock = _acquire_migration_lock(conn)
    cursor = conn.cursor()
    try:
        # The authoritative read. Everything the pre-check said is re-derived
        # here, under the write lock, where no other opener can commit between
        # the read and the rebuild it authorises.
        current = _user_version(conn)
        _guard_downgrade(current)
        if current == SCHEMA_VERSION:
            if owns_lock:
                conn.rollback()
            logger.info(
                "schema_migration_already_applied",
                database=str(_main_database_path(conn) or ":memory:"),
                version=current,
                detail="another opener applied the migration while this one waited for the write lock",
            )
            return

        # PRD-CORE-245-NFR02/NFR04: schema 5 is the first delta that drops and
        # renames tables, and ``ensure_schema`` runs it automatically on the
        # first open by a new build — on a store other processes may still hold
        # open. Snapshot the pre-migration bytes before the storm, and refuse
        # the migration outright if the snapshot cannot be written — or if
        # whether one is needed cannot be read at all. The refusal raises from
        # inside the transaction, which rolls back below, so it still leaves
        # ``user_version`` and every row exactly as they were.
        #
        # The snapshot runs while this connection HOLDS the write lock, which
        # is why it reads through a sibling handle (see
        # ``_schema_backup._open_snapshot_source``) and why exactly one is
        # written per migration rather than one per racing opener.
        #
        # Gated on the DESTRUCTIVE deltas specifically. A snapshot copies the
        # whole database (2.1 s for 186 MB), which is the right price to pay
        # once for a rebuild-and-rename and the wrong price to pay on every
        # future additive ALTER. Raise this bound when a later delta is
        # destructive too.
        needs_snapshot = current < _LAST_DESTRUCTIVE_SCHEMA_VERSION
        backup_path = (
            snapshot_before_migration(conn, from_version=current, to_version=SCHEMA_VERSION) if needs_snapshot else None
        )

        # The whole storm — bootstrap, every registered delta, and the version
        # stamp — is ONE transaction, so an interruption leaves user_version at
        # its previous value with every original row intact rather than a
        # half-rebuilt store. Under WAL a concurrent reader on another
        # connection sees either the full pre- or the full post-migration
        # schema, never an intermediate one (NFR04).
        #
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
        _log_migration_applied(conn, from_version=current, backup_path=backup_path)
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
from trw_memory.storage._schema_v5 import (  # noqa: E402
    migrate_v5_namespace_boundary as _migrate_v5_namespace_boundary,
)

_MIGRATIONS[2] = _migrate_v2_memory_model
_MIGRATIONS[3] = _migrate_v3_legacy_enums
_MIGRATIONS[4] = _migrate_v4_verification_status
_MIGRATIONS[5] = _migrate_v5_namespace_boundary


CREATE_MEMORIES_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    namespace UNINDEXED,
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
                "INSERT INTO memories_fts(id, namespace, content, detail, tags) "
                "SELECT id, namespace, content, COALESCE(detail, ''), COALESCE(tags, '[]') FROM memories"
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
    conn.execute(CREATE_VEC_INDEX)
    conn.commit()
