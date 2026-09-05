"""W05 — a broken ``memory_tags`` read must not read back as "no neighbours".

``derive_tag_neighbours`` used to wrap its whole query block in
``except sqlite3.Error: return []``. The docstring justified that with the one
case it really covers — a store written before schema 5 has no ``memory_tags``
table — but the handler also swallowed a locked database, a corrupt page, and a
``memory_tags`` whose columns have drifted, and handed every one of them back as
the empty list a root with no tag relations returns. The traversal above it, and
the health probes above that, then read an unreadable store as an empty graph.

These tests inject the fault on the real path: a genuine SQLite table named
``memory_tags`` whose columns are not the ones the derivation selects, which is
exactly what a partially-migrated store looks like.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trw_memory.graph import DERIVED_EDGE_TYPE, graph_query
from trw_memory.models.config import MemoryConfig
from trw_memory.retrieval.tag_derivation import derive_tag_neighbours

pytestmark = pytest.mark.unit

_NAMESPACE = "project:visibility"


def _drifted_tag_index(path: Path) -> sqlite3.Connection:
    """A store whose ``memory_tags`` exists but does not carry the expected columns."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memory_tags (namespace TEXT, label TEXT, entry_id TEXT)")
    conn.commit()
    return conn


def test_drifted_tag_index_raises_instead_of_returning_empty(tmp_path: Path) -> None:
    """A read failure that is NOT the missing table reaches the caller."""
    conn = _drifted_tag_index(tmp_path / "drifted.db")
    try:
        with pytest.raises(sqlite3.OperationalError):
            derive_tag_neighbours(conn, "ROOT", namespace=_NAMESPACE, config=MemoryConfig())
    finally:
        conn.close()


def test_traversal_propagates_a_failed_derivation(tmp_path: Path) -> None:
    """graph_query surfaces the failure rather than returning a short walk.

    The materialised half of the traversal already propagates its SQLite errors;
    before this change the derived half alone answered ``[]``, so an explicit
    ``tag_cooccurrence`` request over an unreadable index looked like a root
    with no tag neighbours.
    """
    conn = _drifted_tag_index(tmp_path / "traverse.db")
    conn.execute("CREATE TABLE memory_graph_edges (source_id TEXT, target_id TEXT, edge_type TEXT, weight REAL)")
    conn.execute("CREATE TABLE memories (id TEXT, namespace TEXT)")
    conn.commit()
    try:
        with pytest.raises(sqlite3.OperationalError):
            graph_query(
                conn,
                ["ROOT"],
                depth=1,
                edge_types=[DERIVED_EDGE_TYPE],
                namespace=_NAMESPACE,
                config=MemoryConfig(),
            )
    finally:
        conn.close()


def test_pre_schema_5_store_still_degrades_to_empty(tmp_path: Path) -> None:
    """The one documented suppression survives: no ``memory_tags`` table at all."""
    conn = sqlite3.connect(tmp_path / "pre-schema-5.db")
    try:
        assert derive_tag_neighbours(conn, "ROOT", namespace=_NAMESPACE, config=MemoryConfig()) == []
    finally:
        conn.close()
