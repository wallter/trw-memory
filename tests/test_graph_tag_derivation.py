"""PRD-CORE-245 FR07 + NFR01/NFR05 — the tag relation is derived, not materialised.

Baseline this replaces, measured on the reference store 2026-09-03: 98,288
``tag_cooccurrence`` edges (95.96% of all edges), holding a mean 19.1 neighbours
per root against the 573.3 the same predicate yields over the full corpus — 3.3%
of the relation they claimed to store.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from trw_memory.graph import DERIVED_EDGE_TYPE, VALID_EDGE_TYPES, graph_query
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.tag_derivation import derive_tag_neighbours
from trw_memory.storage.sqlite_backend import SQLiteBackend

pytestmark = pytest.mark.unit

NAMESPACE = "project:derivation"


def _seed(backend: SQLiteBackend, count: int, *, namespace: str = NAMESPACE) -> None:
    """Store *count* entries over three independent tag axes.

    Independent moduli (3/4/5) rather than one shared ladder: every entry gets
    three DISTINCT tags, and two entries share two tags exactly when they agree
    on two axes -- which is the ``min_shared_tags=2`` predicate under test.
    """
    for index in range(count):
        backend.store(
            MemoryEntry(
                id=f"M-{index:05d}",
                content=f"entry {index} about retrieval",
                namespace=namespace,
                tags=[f"a{index % 3}", f"b{index % 4}", f"c{index % 5}"],
            )
        )


def test_derived_tag_edges_replace_materialised_rows(tmp_path: Path) -> None:
    """FR07: no tag edge is ever written, and the inverted index carries the relation."""
    backend = SQLiteBackend(tmp_path / "derive.db")
    try:
        _seed(backend, 60)

        materialised = backend._conn.execute(
            "SELECT COUNT(*) FROM memory_graph_edges WHERE edge_type = ?", (DERIVED_EDGE_TYPE,)
        ).fetchone()[0]
        assert materialised == 0

        # memory_tags holds one row per (namespace, tag, entry) pair. The ladder
        # above collides tags within an entry for some indices, so the count is
        # the DISTINCT pair count, which is what the index stores.
        expected_pairs = 60 * 3  # three distinct tags per entry
        stored_pairs = backend._conn.execute("SELECT COUNT(*) FROM memory_tags").fetchone()[0]
        assert stored_pairs == expected_pairs

        neighbours = derive_tag_neighbours(backend._conn, "M-00000", namespace=NAMESPACE, config=MemoryConfig())
        assert neighbours, "the relation must still be answerable after the rows are gone"
        assert all(n.shared_tags >= MemoryConfig().graph_tag_min_shared_tags for n in neighbours)
        assert all(0.0 <= n.weight <= 1.0 for n in neighbours)
    finally:
        backend.close()


def test_derivation_honours_each_cap_independently(tmp_path: Path) -> None:
    """Each of the three tunables changes the result on its own."""
    backend = SQLiteBackend(tmp_path / "caps.db")
    try:
        _seed(backend, 80)
        conn = backend._conn

        permissive = MemoryConfig(graph_tag_min_shared_tags=1, graph_tag_derive_top_k=200)
        wide = derive_tag_neighbours(conn, "M-00000", namespace=NAMESPACE, config=permissive)

        stricter = MemoryConfig(graph_tag_min_shared_tags=3, graph_tag_derive_top_k=200)
        narrow = derive_tag_neighbours(conn, "M-00000", namespace=NAMESPACE, config=stricter)
        assert len(narrow) < len(wide), "min_shared_tags must bind"

        capped = derive_tag_neighbours(
            conn,
            "M-00000",
            namespace=NAMESPACE,
            config=MemoryConfig(graph_tag_min_shared_tags=1, graph_tag_derive_top_k=3),
        )
        assert len(capped) == 3, "top_k must bind"

        # Every tag in this corpus has at least 16 postings, so a ceiling of 5
        # excludes all of them and the derivation returns nothing.
        noisy = derive_tag_neighbours(
            conn,
            "M-00000",
            namespace=NAMESPACE,
            config=MemoryConfig(graph_tag_min_shared_tags=1, graph_tag_max_tag_postings=5),
        )
        assert noisy == [], "max_tag_postings must bind"
    finally:
        backend.close()


def test_derived_neighbours_are_deterministically_ordered(tmp_path: Path) -> None:
    """Ordering is stated, not inherited from whatever the index returned."""
    backend = SQLiteBackend(tmp_path / "order.db")
    try:
        _seed(backend, 60)
        config = MemoryConfig(graph_tag_min_shared_tags=1, graph_tag_derive_top_k=25)
        first = derive_tag_neighbours(backend._conn, "M-00000", namespace=NAMESPACE, config=config)
        second = derive_tag_neighbours(backend._conn, "M-00000", namespace=NAMESPACE, config=config)
        assert first == second
        keys = [(-n.shared_tags, n.entry_id) for n in first]
        assert keys == sorted(keys), "shared-tag count descending, then entry id ascending"
    finally:
        backend.close()


def test_derivation_is_namespace_scoped(tmp_path: Path) -> None:
    """A neighbour from another namespace is not a neighbour."""
    backend = SQLiteBackend(tmp_path / "scoped.db")
    try:
        _seed(backend, 20)
        _seed(backend, 20, namespace="project:elsewhere")
        neighbours = derive_tag_neighbours(backend._conn, "M-00000", namespace=NAMESPACE, config=MemoryConfig())
        ids = {n.entry_id for n in neighbours}
        rows = backend._conn.execute("SELECT id FROM memories WHERE namespace = ?", (NAMESPACE,)).fetchall()
        assert ids <= {str(r[0]) for r in rows}
    finally:
        backend.close()


def test_untagged_namespace_returns_empty_without_raising(tmp_path: Path) -> None:
    """Negative path: nothing to derive is an empty list, not an error."""
    backend = SQLiteBackend(tmp_path / "empty.db")
    try:
        backend.store(MemoryEntry(id="M-bare", content="no tags", namespace=NAMESPACE))
        assert derive_tag_neighbours(backend._conn, "M-bare", namespace=NAMESPACE, config=MemoryConfig()) == []
    finally:
        backend.close()


def test_default_traversal_returns_no_derived_edge(tmp_path: Path) -> None:
    """FR07: graph_query walks only materialised types unless asked otherwise."""
    backend = SQLiteBackend(tmp_path / "traverse.db")
    try:
        _seed(backend, 40)
        default_walk = graph_query(backend._conn, ["M-00000"], depth=2, namespace=NAMESPACE)
        assert all(node["edge_type"] != DERIVED_EDGE_TYPE for node in default_walk)

        explicit = graph_query(
            backend._conn,
            ["M-00000"],
            depth=1,
            edge_types=[DERIVED_EDGE_TYPE],
            namespace=NAMESPACE,
            config=MemoryConfig(graph_tag_min_shared_tags=1),
        )
        assert explicit, "an explicit request must still be answerable"
        assert all(node["edge_type"] == DERIVED_EDGE_TYPE for node in explicit)
        # The type stays a member of the public vocabulary; only its backing changed.
        assert DERIVED_EDGE_TYPE in VALID_EDGE_TYPES
    finally:
        backend.close()


def test_derivation_latency_budget(tmp_path: Path) -> None:
    """NFR01: bounded single-root derivation stays at or below the 15 ms p50 budget."""
    backend = SQLiteBackend(tmp_path / "latency.db")
    try:
        _seed(backend, 2000)
        config = MemoryConfig()
        roots = [f"M-{i:05d}" for i in range(0, 2000, 40)]
        # Warm the page cache so the measurement is of the query, not of first I/O.
        derive_tag_neighbours(backend._conn, roots[0], namespace=NAMESPACE, config=config)

        timings: list[float] = []
        for root in roots:
            start = time.perf_counter()
            derive_tag_neighbours(backend._conn, root, namespace=NAMESPACE, config=config)
            timings.append((time.perf_counter() - start) * 1000.0)

        p50 = statistics.median(timings)
        assert p50 <= 15.0, f"derivation p50 {p50:.2f} ms exceeds the 15 ms budget over {len(timings)} roots"
    finally:
        backend.close()


def test_tunables_are_typed_and_bounded() -> None:
    """NFR05: every knob is a bounded Pydantic field, rejected at load when out of range."""
    config = MemoryConfig()
    assert config.graph_tag_min_shared_tags == 2
    assert config.graph_tag_max_tag_postings == 500
    assert config.graph_tag_derive_top_k == 25

    for field, bad in (
        ("graph_tag_min_shared_tags", 0),
        ("graph_tag_min_shared_tags", 11),
        ("graph_tag_max_tag_postings", 0),
        ("graph_tag_max_tag_postings", 100_001),
        ("graph_tag_derive_top_k", 0),
        ("graph_tag_derive_top_k", 201),
    ):
        with pytest.raises(ValidationError):
            MemoryConfig(**{field: bad})
