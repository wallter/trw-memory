"""PRD-CORE-245 — the schema-5 census also covers the sidecar tables.

``_census`` in ``_schema_v5.py`` only compared ``memories`` before/after the
rebuild. ``vec_index`` (rebuilt with ``INSERT OR IGNORE``), ``memory_graph_edges``
(same), and ``wiki_refs`` (plain ``INSERT``) had no independent row-count check,
so a future collision or rebuild bug could silently drop a sidecar row with no
operator-visible signal. These tests exercise the extended
:func:`trw_memory.storage._schema_v5._sidecar_census` through the real
migration path — never a hand-built census.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import trw_memory.storage._dbapi  # noqa: F401  — installs pysqlite3 as ``sqlite3``
import trw_memory.storage._schema_v5 as schema_v5_module
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._schema import CREATE_VEC_INDEX, ensure_schema
from trw_memory.storage._schema_v5 import MigrationCensusMismatchError
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.unit

_EDGE_INSERT = (
    "INSERT INTO memory_graph_edges (namespace, source_id, target_id, edge_type, weight, created_at) "
    "VALUES (?, ?, ?, ?, 0.5, '2026-01-01T00:00:00+00:00')"
)


def _v4_fixture_with_sidecars(path: Path) -> None:
    """Write a v4-stamped store carrying real rows in every sidecar table.

    ``vec_index`` is created directly via its schema-5 DDL rather than relying
    on ``SQLiteBackend`` to have loaded the optional ``sqlite-vec`` extension,
    so the test is deterministic regardless of that dependency's availability
    (mirrors ``_rebuild_vec_index``'s own "absent -> created fresh" handling).
    """
    backend = SQLiteBackend(path)
    try:
        backend.store(MemoryEntry(id="M-a", content="a", namespace="project:x"))
        backend.store(MemoryEntry(id="M-b", content="b", namespace="project:y"))
    finally:
        backend.close()

    conn = sqlite3.connect(path)
    conn.execute(CREATE_VEC_INDEX)
    conn.execute(_EDGE_INSERT, ("project:x", "M-a", "M-b", "related"))
    conn.execute(_EDGE_INSERT, ("project:x", "M-a", "M-b", "tag_cooccurrence"))
    conn.execute(_EDGE_INSERT, ("project:y", "M-b", "M-a", "tag_cooccurrence"))
    conn.execute("INSERT INTO vec_index (entry_id, namespace) VALUES ('M-a', 'project:x')")
    conn.execute("INSERT INTO vec_index (entry_id, namespace) VALUES ('M-b', 'project:y')")
    conn.execute(
        "INSERT INTO wiki_refs (source_entry_id, source_slug, target_slug, ref_type, namespace, updated_at) "
        "VALUES ('M-a', 'slug-a', 'slug-b', 'related', 'project:x', '2026-01-01T00:00:00+00:00')"
    )
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()


def test_sidecar_rows_survive_migration_and_tag_cooccurrence_is_dropped(tmp_path: Path) -> None:
    """FR07 + the new census: non-tag_cooccurrence sidecar rows all survive; tag_cooccurrence does not."""
    db = tmp_path / "sidecars.db"
    _v4_fixture_with_sidecars(db)

    conn = sqlite3.connect(db)
    ensure_schema(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 5

    assert conn.execute("SELECT COUNT(*) FROM memory_graph_edges").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM memory_graph_edges WHERE edge_type = 'related'").fetchone()[0] == 1
    assert (
        conn.execute("SELECT COUNT(*) FROM memory_graph_edges WHERE edge_type = 'tag_cooccurrence'").fetchone()[0] == 0
    )
    assert conn.execute("SELECT COUNT(*) FROM vec_index").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM wiki_refs").fetchone()[0] == 1
    conn.close()


@pytest.mark.parametrize("table", ["vec_index", "memory_graph_edges", "wiki_refs"])
def test_a_dropped_sidecar_row_raises_census_mismatch_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    """A rebuild that silently drops one row is caught, not stamped (defense in depth).

    Each rebuild is wrapped to run for real and then delete one extra row it
    should not have dropped, proving the new sidecar census — not the
    pre-existing ``memories`` census — is what catches it.
    """
    db = tmp_path / f"drop-{table}.db"
    _v4_fixture_with_sidecars(db)

    rebuild_name = {
        "vec_index": "_rebuild_vec_index",
        "memory_graph_edges": "_rebuild_graph_edges",
        "wiki_refs": "_rebuild_wiki_refs",
    }[table]
    original = getattr(schema_v5_module, rebuild_name)

    def _dropping_rebuild(cursor: sqlite3.Cursor) -> None:
        original(cursor)
        cursor.execute(f"DELETE FROM {table} WHERE rowid = (SELECT MIN(rowid) FROM {table})")

    monkeypatch.setattr(schema_v5_module, rebuild_name, _dropping_rebuild)

    conn = sqlite3.connect(db)
    with pytest.raises(MigrationCensusMismatchError, match=table):
        ensure_schema(conn)

    # The rollback contract shared with the ``memories`` census (NFR02): a
    # raise inside the migration rolls the whole transaction back, so
    # user_version never advances past the pre-migration value.
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
    conn.close()
