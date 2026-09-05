"""Schema-5 delta — namespace becomes a containment boundary (PRD-CORE-245).

SQLite cannot alter a primary key in place, so the ``memories`` table is
rebuilt-renamed-reindexed. The same rebuild carries every other change in the
consolidated schema-5 migration table:

* ``memories`` is re-keyed on ``PRIMARY KEY (namespace, id)`` (FR01).
* ``wiki_refs`` foreign key is rewritten to the composite parent (FR01).
* ``memory_graph_edges`` gains ``namespace`` and a namespace-qualified
  uniqueness constraint, and every ``tag_cooccurrence`` row is dropped (FR02,
  FR07).
* ``vec_index`` uniqueness becomes ``(namespace, entry_id)`` (FR02).
* ``memories_fts`` is dropped so :func:`trw_memory.storage._schema.ensure_fts_table`
  rebuilds it with its new ``namespace`` column on the same open (FR02).
* ``memory_tags`` is backfilled from the ``tags`` column (FR07).
* ``anchor_validity`` is nulled for every row that carries no anchors,
  ``verification_checked_at`` is added, and ``sessions_surfaced`` /
  ``avg_rework_delta`` / ``outcome_correlation`` are dropped by simply not being
  named in the copy (PRD-CORE-244 FR01/FR03/FR08).

The delta is registered exactly once, at the bottom of
:mod:`trw_memory.storage._schema`. It owns no transaction: ``ensure_schema``
opens one ``BEGIN IMMEDIATE`` around the whole storm, so an interruption leaves
``PRAGMA user_version`` at 4 with every original row intact.
"""

from __future__ import annotations

import sqlite3

import structlog

logger = structlog.get_logger(__name__)

__all__ = ["MigrationCensusMismatchError", "migrate_v5_namespace_boundary"]


class MigrationCensusMismatchError(RuntimeError):
    """The schema-5 rebuild did not preserve a table's row count.

    Covers ``memories`` (row count + per-namespace census) and the three
    sidecar tables the rebuild also rewrites — ``vec_index``,
    ``memory_graph_edges`` (excluding the ``tag_cooccurrence`` rows FR07 drops
    on purpose), and ``wiki_refs``. Raised from inside ``ensure_schema``'s
    transaction so the whole delta rolls back and ``PRAGMA user_version`` is
    never stamped on a store whose rows the rebuild failed to carry across
    intact.
    """


#: Columns whose copied value is not a plain passthrough.
#:
#: ``namespace`` is ``NOT NULL`` under schema 5 but was merely defaulted before,
#: so a legacy NULL is coalesced rather than rejected. ``anchor_validity``
#: becomes NULL for every row that carries no anchors — PRD-CORE-244-FR01's
#: "unassessed is not a perfect score" backfill, expressed inside the one copy
#: pass the rebuild already performs.
#: Columns a copy expression READS besides the one it writes. When the source
#: table predates the dependency the expression cannot run, so the plain column
#: is copied instead.
_COPY_DEPENDENCIES: dict[str, str] = {"anchor_validity": "anchors"}

_COPY_EXPRESSIONS: dict[str, str] = {
    "namespace": "COALESCE(namespace, 'default')",
    "anchor_validity": (
        "CASE WHEN anchors IS NULL OR TRIM(COALESCE(anchors, '')) IN ('', '[]') THEN NULL ELSE anchor_validity END"
    ),
}


def _rename_table_in_ddl(ddl: str, table: str, new_name: str) -> str:
    """Retarget a ``CREATE TABLE IF NOT EXISTS <table> (`` statement at *new_name*."""
    marker = f"EXISTS {table} ("
    if marker not in ddl:
        raise ValueError(f"DDL does not declare table {table!r}")
    return ddl.replace(marker, f"EXISTS {new_name} (", 1)


def _table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    row = cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _census(cursor: sqlite3.Cursor) -> tuple[int, tuple[tuple[str, int], ...]]:
    """Return ``(row_count, sorted (namespace, count) pairs)`` for ``memories``."""
    total = int(cursor.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    rows = cursor.execute(
        "SELECT COALESCE(namespace, 'default') AS ns, COUNT(*) FROM memories GROUP BY ns ORDER BY ns"
    ).fetchall()
    return total, tuple((str(name), int(count)) for name, count in rows)


def _sidecar_census(cursor: sqlite3.Cursor) -> dict[str, int]:
    """Return row counts for every sidecar table the schema-5 rebuild carries across.

    ``_rebuild_graph_edges`` and ``_rebuild_vec_index`` use ``INSERT OR IGNORE``
    to survive uniqueness collisions their new composite keys could in theory
    introduce; ``_rebuild_wiki_refs`` is a plain ``INSERT``. None of the three
    had an independent row-count check before this — only ``memories`` did — so
    a future collision (or a bug in one of these rebuilds) could silently drop
    a row with no operator-visible signal. This same function is called both
    before and after the rebuild so the counts are directly comparable.

    ``memory_graph_edges`` excludes ``edge_type = 'tag_cooccurrence'`` on
    purpose: PRD-CORE-244 FR07 always drops those rows (see
    ``_rebuild_graph_edges``), so counting them would make every migration a
    false mismatch. The exclusion is a no-op on the "after" call, since the
    rebuild has already removed every such row by then.

    ``vec_index`` may not exist at all (sqlite-vec never loaded against this
    database); its count is 0 in that case on both sides, which never mismatches.
    """
    edges = int(
        cursor.execute("SELECT COUNT(*) FROM memory_graph_edges WHERE edge_type != 'tag_cooccurrence'").fetchone()[0]
    )
    vec = (
        int(cursor.execute("SELECT COUNT(*) FROM vec_index").fetchone()[0]) if _table_exists(cursor, "vec_index") else 0
    )
    wiki = int(cursor.execute("SELECT COUNT(*) FROM wiki_refs").fetchone()[0])
    return {"memory_graph_edges": edges, "vec_index": vec, "wiki_refs": wiki}


def _rebuild_memories(cursor: sqlite3.Cursor) -> None:
    from trw_memory.storage._schema import CREATE_MEMORIES, MEMORIES_INDEXES
    from trw_memory.storage._shared import ENTRY_COLUMNS

    # Copy only the columns the SOURCE table actually has. A database written by
    # a build old enough to predate a column never got it — ``_migrate_cols``
    # only backfills the columns it enumerates — and naming it in the SELECT
    # would fail the whole migration on exactly the legacy shapes the bootstrap
    # exists to normalise. Anything omitted takes the new table's DEFAULT.
    present = {str(row[1]) for row in cursor.execute("PRAGMA table_info(memories)").fetchall()}
    copied = [column for column in ENTRY_COLUMNS if column in present]
    columns = ", ".join(copied)
    selects = ", ".join(
        _COPY_EXPRESSIONS[column]
        if column in _COPY_EXPRESSIONS and _COPY_DEPENDENCIES.get(column, column) in present
        else column
        for column in copied
    )
    cursor.execute("DROP TABLE IF EXISTS memories_v5_rebuild")
    cursor.execute(_rename_table_in_ddl(CREATE_MEMORIES, "memories", "memories_v5_rebuild"))
    cursor.execute(
        f"INSERT INTO memories_v5_rebuild ({columns}) SELECT {selects} FROM memories"  # noqa: S608
    )
    cursor.execute("DROP TABLE memories")
    cursor.execute("ALTER TABLE memories_v5_rebuild RENAME TO memories")
    for statement in MEMORIES_INDEXES:
        cursor.execute(statement)


def _rebuild_graph_edges(cursor: sqlite3.Cursor) -> None:
    from trw_memory.storage._schema import (
        CREATE_GRAPH_EDGES,
        CREATE_IDX_MGE_SOURCE,
        CREATE_IDX_MGE_TARGET,
    )

    cursor.execute("DROP TABLE IF EXISTS memory_graph_edges_v5_rebuild")
    cursor.execute(_rename_table_in_ddl(CREATE_GRAPH_EDGES, "memory_graph_edges", "memory_graph_edges_v5_rebuild"))
    # ``tag_cooccurrence`` rows are simply not copied — FR07's deletion is the
    # absence of a WHERE-clause match, not a second DELETE pass over 98k rows.
    #
    # The correlated lookup of an edge's namespace matches ``memories`` on a
    # BARE id. That is safe HERE and only here: ``memories`` has already been
    # rebuilt with the composite key by this point, but every id is still
    # globally unique because the pre-v5 schema's sole primary key was ``id``,
    # so at most one row can match. It measured 0 ids duplicated across
    # namespaces on the reference store, and after this migration the ambiguity
    # cannot arise again because nothing else resolves an edge by bare id.
    cursor.execute(
        "INSERT OR IGNORE INTO memory_graph_edges_v5_rebuild "
        "(id, namespace, source_id, target_id, edge_type, weight, created_at, edge_metadata) "
        "SELECT e.id, "
        "COALESCE((SELECT m.namespace FROM memories m WHERE m.id = e.source_id LIMIT 1), 'default'), "
        "e.source_id, e.target_id, e.edge_type, e.weight, e.created_at, COALESCE(e.edge_metadata, '{}') "
        "FROM memory_graph_edges e WHERE e.edge_type != 'tag_cooccurrence'"
    )
    cursor.execute("DROP TABLE memory_graph_edges")
    cursor.execute("ALTER TABLE memory_graph_edges_v5_rebuild RENAME TO memory_graph_edges")
    cursor.execute(CREATE_IDX_MGE_SOURCE)
    cursor.execute(CREATE_IDX_MGE_TARGET)


def _rebuild_wiki_refs(cursor: sqlite3.Cursor) -> None:
    from trw_memory.storage._schema import (
        CREATE_IDX_WIKI_REFS_SOURCE,
        CREATE_IDX_WIKI_REFS_TARGET,
        CREATE_WIKI_REFS,
    )

    columns = "source_entry_id, source_slug, target_slug, ref_type, label, bidirectional, namespace, updated_at"
    cursor.execute("DROP TABLE IF EXISTS wiki_refs_v5_rebuild")
    cursor.execute(_rename_table_in_ddl(CREATE_WIKI_REFS, "wiki_refs", "wiki_refs_v5_rebuild"))
    cursor.execute(
        f"INSERT INTO wiki_refs_v5_rebuild ({columns}) SELECT {columns} FROM wiki_refs"  # noqa: S608
    )
    cursor.execute("DROP TABLE wiki_refs")
    cursor.execute("ALTER TABLE wiki_refs_v5_rebuild RENAME TO wiki_refs")
    cursor.execute(CREATE_IDX_WIKI_REFS_SOURCE)
    cursor.execute(CREATE_IDX_WIKI_REFS_TARGET)


def _rebuild_vec_index(cursor: sqlite3.Cursor) -> None:
    """Re-key ``vec_index`` on ``(namespace, entry_id)``, preserving every rowid.

    The rowid is the join key into the ``vec_memories`` vec0 table, so it must
    survive the rebuild verbatim or every stored vector is orphaned. Absent when
    sqlite-vec was never loaded against this database, in which case
    ``ensure_vec_table`` creates the schema-5 shape directly.
    """
    from trw_memory.storage._schema import CREATE_VEC_INDEX

    if not _table_exists(cursor, "vec_index"):
        return
    cursor.execute("DROP TABLE IF EXISTS vec_index_v5_rebuild")
    cursor.execute(_rename_table_in_ddl(CREATE_VEC_INDEX, "vec_index", "vec_index_v5_rebuild"))
    # HyPE sibling vectors carry a synthetic ``{parent}#hype{n}`` entry_id with
    # no ``memories`` row of their own; they inherit their parent's namespace.
    cursor.execute(
        "INSERT OR IGNORE INTO vec_index_v5_rebuild (rowid, entry_id, namespace) "
        "SELECT v.rowid, v.entry_id, COALESCE("
        "  (SELECT m.namespace FROM memories m WHERE m.id = v.entry_id LIMIT 1),"
        "  (SELECT m.namespace FROM memories m"
        "     WHERE INSTR(v.entry_id, '#hype') > 0"
        "       AND m.id = SUBSTR(v.entry_id, 1, INSTR(v.entry_id, '#hype') - 1) LIMIT 1),"
        "  'default') FROM vec_index v"
    )
    cursor.execute("DROP TABLE vec_index")
    cursor.execute("ALTER TABLE vec_index_v5_rebuild RENAME TO vec_index")


def _backfill_memory_tags(cursor: sqlite3.Cursor) -> int:
    """Populate ``memory_tags`` from the JSON ``tags`` column of every row.

    ``json_each`` is available in every SQLite build trw-memory supports
    (JSON1 has been compiled in by default since 3.38). Measured 115,660 rows in
    154.5 ms on the 9,366-entry reference store.
    """
    cursor.execute("DELETE FROM memory_tags")
    cursor.execute(
        "INSERT OR IGNORE INTO memory_tags (namespace, tag, entry_id) "
        "SELECT m.namespace, json_each.value, m.id FROM memories m, json_each(m.tags) "
        "WHERE json_valid(m.tags) AND json_type(m.tags) = 'array' AND json_each.value IS NOT NULL "
        "AND TRIM(CAST(json_each.value AS TEXT)) != ''"
    )
    return int(cursor.execute("SELECT COUNT(*) FROM memory_tags").fetchone()[0])


def _apply_rebuilds(cursor: sqlite3.Cursor) -> int:
    """Run every table rebuild and return the ``memory_tags`` row count."""
    from trw_memory.storage._schema import CREATE_IDX_MEMORY_TAGS_ENTRY, CREATE_MEMORY_TAGS

    _rebuild_memories(cursor)
    _rebuild_graph_edges(cursor)
    _rebuild_wiki_refs(cursor)
    _rebuild_vec_index(cursor)
    # The FTS5 virtual table gains a ``namespace`` column, which FTS5 cannot
    # ALTER in. It is a pure derived index over ``memories``, so it is dropped
    # here and repopulated by ``ensure_fts_table`` on the same open.
    cursor.execute("DROP TABLE IF EXISTS memories_fts")
    cursor.execute(CREATE_MEMORY_TAGS)
    cursor.execute(CREATE_IDX_MEMORY_TAGS_ENTRY)
    return _backfill_memory_tags(cursor)


def migrate_v5_namespace_boundary(cursor: sqlite3.Cursor) -> None:
    """Apply the whole consolidated schema-5 delta (PRD-CORE-245 + PRD-CORE-244).

    Runs inside ``ensure_schema``'s single transaction. Self-checking: the row
    count and per-namespace census are captured before the rebuild and compared
    after it, and any difference raises :class:`MigrationCensusMismatchError`
    so the transaction rolls back rather than stamping a partially migrated
    store. The same before/after comparison covers ``vec_index``,
    ``memory_graph_edges``, and ``wiki_refs`` — the three sidecar rebuilds that
    use ``INSERT OR IGNORE`` (or a plain ``INSERT``) with no census of their own.
    """
    before_total, before_census = _census(cursor)
    before_sidecars = _sidecar_census(cursor)

    # SQLite >= 3.25 rewrites every schema reference to a renamed table and
    # fails the rename when any object cannot be re-resolved. ``wiki_refs``
    # names ``memories`` in its foreign key, and the rebuild renames a table
    # INTO that name while the original is gone, so the modern behaviour must
    # be suspended for the duration (the recipe SQLite's own ALTER TABLE docs
    # give for a rebuild-rename).
    legacy_alter = int(cursor.execute("PRAGMA legacy_alter_table").fetchone()[0])
    cursor.execute("PRAGMA legacy_alter_table = ON")
    try:
        tag_rows = _apply_rebuilds(cursor)
    finally:
        cursor.execute(f"PRAGMA legacy_alter_table = {legacy_alter}")

    after_total, after_census = _census(cursor)
    if (after_total, after_census) != (before_total, before_census):
        raise MigrationCensusMismatchError(
            "schema 5 rebuild changed the memories census "
            f"(before: {before_total} rows in {len(before_census)} namespaces; "
            f"after: {after_total} rows in {len(after_census)} namespaces) — "
            "rolling back rather than stamping a partially migrated store"
        )

    after_sidecars = _sidecar_census(cursor)
    if after_sidecars != before_sidecars:
        drifted = ", ".join(
            f"{table} (before={before_sidecars[table]}, after={after_sidecars.get(table, 0)})"
            for table in sorted(before_sidecars)
            if before_sidecars[table] != after_sidecars.get(table, 0)
        )
        raise MigrationCensusMismatchError(
            f"schema 5 rebuild changed a sidecar table's row count: {drifted} — "
            "rolling back rather than stamping a partially migrated store"
        )

    # NFR03: counts only. A namespace label is operator-chosen but still
    # user data, so the census is logged as shape (how many namespaces, how
    # many rows in each) and never as names.
    logger.info(
        "schema_v5_migrated",
        rows=after_total,
        namespaces=len(after_census),
        namespace_row_counts=sorted(count for _, count in after_census),
        tag_rows=tag_rows,
    )
