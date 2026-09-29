"""B71-99 / B71-28 (PRD-CORE-307 FR07): graph_query must bound the SQL work

a dense root can force, not just the nodes it returns.

Pre-fix (574756d31) ``graph_query`` grouped a source's *whole* adjacency list
(``GROUP BY target_id ... ORDER BY MIN(rowid)``) before any ``LIMIT`` could
apply, so a hub with far more edges than ``max_nodes`` made SQLite examine
and group every one of them, once per visited source. These tests measure
the actual rows examined (not just the rows returned) and the query plan
SQLite picks, so a regression back to grouping the whole adjacency fails
loudly instead of only showing up as a latency regression under load.
"""

from __future__ import annotations

import sqlite3

from trw_memory.graph import graph_query

from ._test_graph_support import _insert_edge, _insert_memory_row, _make_conn


class _CountingCursor:
    """Proxies a real sqlite3.Cursor; ``sqlite3.Cursor`` forbids attribute
    assignment, so counting needs its own wrapper rather than monkeypatching
    ``fetchall`` directly onto the cursor."""

    def __init__(self, cursor: sqlite3.Cursor, on_fetch: object) -> None:
        self._cursor = cursor
        self._on_fetch = on_fetch

    def fetchall(self) -> list[object]:
        rows = self._cursor.fetchall()
        self._on_fetch(len(rows))
        return rows

    def __getattr__(self, name: str) -> object:
        return getattr(self._cursor, name)


class _CountingConn:
    """Wraps a real sqlite3.Connection and counts rows fetched from any
    ``memory_graph_edges`` adjacency query graph_query issues (not the
    namespace/membership lookups, which are unrelated to the edge scan)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.edge_rows_fetched = 0

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> object:
        cursor = self._conn.execute(sql, params)
        if "memory_graph_edges" in sql and "target_id" in sql:
            return _CountingCursor(cursor, lambda n: setattr(self, "edge_rows_fetched", self.edge_rows_fetched + n))
        return cursor

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)


class TestGraphQueryEdgeScanBudget:
    def test_dense_hub_edge_scan_bounded_by_node_cap(self) -> None:
        """5000 edges from one hub, max_nodes=10: the edge scan must stay a
        small multiple of the node cap, not the full adjacency (5000)."""
        conn = _make_conn()
        for index in range(5000):
            _insert_edge(conn, "hub", f"n{index}", "similarity", 0.9)
        counting = _CountingConn(conn)

        results = graph_query(counting, ["hub"], depth=1, max_nodes=10)  # type: ignore[arg-type]

        assert len(results) == 10
        # A dense-adjacency scan bounded by budget stays well under the full
        # 5000-edge adjacency; pre-fix this assertion fails at 5000 because
        # GROUP BY forces SQLite to examine every row before LIMIT applies.
        assert counting.edge_rows_fetched < 500

    def test_query_plan_has_no_group_by_temp_btree(self) -> None:
        """The adjacency query graph_query issues must not need a temp
        b-tree to satisfy a GROUP BY — dedup now happens in Python."""
        conn = _make_conn()
        for index in range(20):
            _insert_edge(conn, "hub", f"n{index}", "similarity", 0.9)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)

        graph_query(conn, ["hub"], depth=1, max_nodes=5)

        conn.set_trace_callback(None)
        edge_statements = [s for s in statements if "memory_graph_edges" in s and "target_id" in s]
        assert edge_statements, "expected at least one adjacency query to be traced"
        for statement in edge_statements:
            plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
            plan_text = " ".join(str(row) for row in plan_rows).upper()
            assert "GROUP BY" not in plan_text

    def test_roots_beyond_traversal_bound_are_dropped_deterministically(self) -> None:
        """B71-28: recall can pass up to 10,000 roots; graph_query must
        truncate rather than walk them all. Deterministic = same prefix
        wins every call, not an arbitrary subset."""
        conn = _make_conn()
        many_roots = [f"root-{i}" for i in range(1200)]
        for root in many_roots:
            _insert_edge(conn, root, f"{root}-target", "similarity", 0.9)

        first = graph_query(conn, many_roots, depth=1)
        second = graph_query(conn, many_roots, depth=1)

        assert first == second
        # Only the first _MAX_TRAVERSAL_ROOTS (1000) roots' neighbours can
        # appear; the tail 200 roots are dropped, not walked.
        surfaced_ids = {result["id"] for result in first}
        assert "root-1199-target" not in surfaced_ids
        assert "root-0-target" in surfaced_ids


def _sqlite_step_count(fn: object, *args: object, **kwargs: object) -> int:
    """Run *fn* while counting SQLite VM opcodes via ``set_progress_handler(n=1)`` -- a proxy for
    rows SQLite actually walks, independent of the Python-level counter under test. Used because
    stock ``sqlite3`` exposes no ``sqlite3_stmt_status(FULLSCAN_STEP)`` binding; the progress
    handler is the closest built-in signal of real VM work done, and firing every 1 opcode gives
    fine-enough resolution to tell "scanned ~10 rows" from "scanned ~5000 rows" reliably."""
    conn = args[0]
    steps = 0

    def _tick() -> int:
        nonlocal steps
        steps += 1
        return 0

    conn.set_progress_handler(_tick, 1)
    try:
        fn(*args, **kwargs)
    finally:
        conn.set_progress_handler(None, 0)
    return steps


class TestGraphQueryEdgeScanBudgetCountsExaminedRows:
    """B71-135(b): the edge-scan budget must bound rows the SQL layer EXAMINES, not just rows it
    RETURNS after a namespace/type filter. Pre-fix, the adjacency query applied the namespace
    ``EXISTS`` filter and the edge-type filter inside SQL's own ``WHERE`` clause, so
    ``ORDER BY rowid LIMIT ?`` could not stop until it found enough MATCHING rows -- on a sparse
    match rate SQLite still had to walk almost the whole adjacency, while ``edges_scanned`` (which
    only tallied what came back) stayed near zero and never tripped the budget.
    """

    def test_sparse_namespace_match_does_not_blow_the_examined_budget(self) -> None:
        """5000 edges from one hub, all one edge type; only the LAST 10 targets (highest rowid)
        resolve into the requested namespace, so a namespace-filtered ``max_nodes=10`` query can
        only be satisfied by walking close to the whole adjacency if the namespace filter runs
        inside SQL. Both calls below pin ``edge_types=["similarity"]`` so the sole variable is the
        namespace filter: an edge-type equality is a genuine index seek on ``idx_mge_source``
        (source_id, edge_type) either way, so it must not become the confound this test measures.
        Calibrated against a raw, unfiltered fetch of exactly ``edge_scan_budget`` (500) rows on
        the SAME connection and data -- the true cost of the work the budget is meant to allow --
        rather than a fixed constant, so the assertion does not depend on SQLite-version-specific
        opcode counts."""
        conn = _make_conn()
        for index in range(5000):
            _insert_edge(conn, "hub", f"n{index}", "similarity", 0.9)
        _insert_memory_row(conn, "hub")
        for index in range(4990, 5000):
            _insert_memory_row(conn, f"n{index}")

        # What examining exactly `edge_scan_budget` (10 * 50 = 500) rows genuinely costs on this
        # SQLite build: the identical adjacency query graph_query issues, run directly for a full
        # 500-row page (max_nodes=10's edge_scan_budget), with no namespace filter to confound it.
        budget_page_steps = _sqlite_step_count(
            lambda c: c.execute(
                "SELECT target_id, edge_type, weight, rowid FROM memory_graph_edges "
                "WHERE source_id = ? AND edge_type IN (?) AND rowid > ? ORDER BY rowid LIMIT ?",
                ("hub", "similarity", 0, 500),
            ).fetchall(),
            conn,
        )

        filtered_steps = _sqlite_step_count(
            lambda c: graph_query(c, ["hub"], depth=1, edge_types=["similarity"], namespace="default", max_nodes=10),
            conn,
        )

        # A budget that truly bounds EXAMINED rows keeps the namespace-filtered call within a small
        # multiple of what a genuine 500-row page costs (the multiple covers the batched
        # target-namespace membership query the fix adds, once per page, not once per row). Pre-fix,
        # the namespace-filtered call walks close to the full 5000-row adjacency to find its 10
        # matches -- an order of magnitude more VM work than even a full budget page -- because the
        # namespace EXISTS filter lived inside the SQL LIMIT clause and only the RETURNED (matching)
        # rows were ever counted against the budget.
        assert filtered_steps < budget_page_steps * 4, (
            f"namespace-filtered call took {filtered_steps} VM steps vs {budget_page_steps} for a "
            "genuine 500-row page -- the edge-scan budget is bounding rows RETURNED, not EXAMINED"
        )

        # A budget that genuinely bounds examined rows to ~500 out of 5000 CANNOT be guaranteed to
        # find matches that only occur past rowid 4990 -- that is the budget doing its job, not a
        # correctness regression. Any id it does surface must still be a real match (never a false
        # positive smuggled past the filter by the Python-side rewrite).
        results = graph_query(conn, ["hub"], depth=1, edge_types=["similarity"], namespace="default", max_nodes=10)
        assert {r["id"] for r in results} <= {f"n{i}" for i in range(4990, 5000)}

        # With the matches moved into the budget's reach (the first 10 targets), the same call
        # finds every one of them -- the fix bounds work, it does not just return nothing.
        conn2 = _make_conn()
        for index in range(5000):
            _insert_edge(conn2, "hub", f"n{index}", "similarity", 0.9)
        _insert_memory_row(conn2, "hub")
        for index in range(10):
            _insert_memory_row(conn2, f"n{index}")
        reachable = graph_query(conn2, ["hub"], depth=1, edge_types=["similarity"], namespace="default", max_nodes=10)
        assert {r["id"] for r in reachable} == {f"n{i}" for i in range(10)}


class TestGraphQueryDefaultPathEdgeScanIsIndexBound:
    """B71-135(b) P2: the previous fix pinned ``edge_types=["similarity"]`` in every regression, so
    it never exercised the DEFAULT traversal (``edge_types=None``) -- P2 review found that path
    still sorts the whole adjacency. ``idx_mge_source`` is ``(source_id, edge_type)``; with
    edge_type unconstrained, a fixed source_id's rows are ordered by edge_type first, so
    ``ORDER BY rowid`` alone cannot be satisfied from the index and SQLite falls back to a temp
    B-tree sort of every matching row before any LIMIT applies -- work that scales with the WHOLE
    adjacency, not with ``max_nodes`` or ``edge_scan_budget``, on a store with more than one edge
    type present. The fix pages by ``(edge_type, rowid)`` -- the index's own key order plus its
    implicit rowid tiebreaker -- so the default path is a pure index seek too."""

    def test_default_traversal_plan_has_no_temp_btree(self) -> None:
        conn = _make_conn()
        for index in range(20):
            _insert_edge(conn, "hub", f"sim{index}", "similarity", 0.9)
            _insert_edge(conn, "hub", f"con{index}", "contradicts", 0.5)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)

        graph_query(conn, ["hub"], depth=1, max_nodes=5)

        conn.set_trace_callback(None)
        edge_statements = [s for s in statements if "memory_graph_edges" in s and "target_id" in s]
        assert edge_statements, "expected at least one adjacency query to be traced"
        for statement in edge_statements:
            plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
            plan_text = " ".join(str(row) for row in plan_rows).upper()
            assert "TEMP B-TREE" not in plan_text, f"default path still sorts: {plan_text}"

    def test_default_traversal_examined_rows_stay_flat_as_the_adjacency_grows(self) -> None:
        """Two edge types present (a temp-sort would need to materialise ALL of them): VM steps
        for a `max_nodes=10` default-path call must stay roughly flat between a 5k-edge and a
        50k-edge adjacency -- a temp-sort's cost instead scales with the total edge count."""

        def steps_for(edge_count: int) -> int:
            conn = _make_conn()
            for index in range(edge_count):
                _insert_edge(conn, "hub", f"sim{index}", "similarity", 0.9)
                _insert_edge(conn, "hub", f"con{index}", "contradicts", 0.5)
            return _sqlite_step_count(lambda c: graph_query(c, ["hub"], depth=1, max_nodes=10), conn)

        small_steps = steps_for(5_000)
        large_steps = steps_for(50_000)

        # A pure index seek costs the same regardless of table size; a full-adjacency temp sort
        # costs roughly 10x more table rows to sort at 50k edges than at 5k, so a generous 3x
        # ceiling on the ratio still fails loudly on a regression back to sorting.
        assert large_steps < small_steps * 3, (
            f"default-path steps grew from {small_steps} (5k edges) to {large_steps} (50k edges) -- "
            "the adjacency scan is no longer index-bound"
        )


class TestGraphQueryMultiPagePagingIsTwoBoundedSeeksNotAnOr:
    """B71-135(b) P1 (sol r2 finding against ad260b892): the P2 fix's single query --
    ``edge_type > ? OR (edge_type = ? AND rowid > ?)`` -- pages correctly on ONE page, but is not a
    bounded seek across MANY pages: SQLite re-walks the current type's earlier rows on every page to
    re-verify the OR, a per-page cost the P2 regressions never saw because they all finished on the
    first page (a namespace-sparse or default-path scenario that never needed a second). That re-walk
    is quadratic in how far into a type's group the cursor has advanced -- 146k -> 3.4M VM steps going
    from 100 to 500 edges per type group (measured on ad260b892).

    The fix drops the OR for two separate, always-bounded index seeks per page: (A) finish the
    CURRENT type (``edge_type = ? AND rowid > ?``, an equality-plus-range seek) and, only if that
    came up short, (B) take the remainder from LATER types (``edge_type > ?``, a range scan that
    starts exactly at the next type's first row). Neither ever re-examines a row from an earlier
    page.
    """

    #: Two edge types SHARING every target id, so after the first type's group is exhausted every
    #: row of the second type is an already-visited duplicate -- the exact shape that forces MANY
    #: small pages (each page's `remaining` budget barely shrinks) rather than one page that returns
    #: everything, which is what let the P2 regressions miss the per-page rescan cost entirely.
    @staticmethod
    def _make(per_group: int) -> sqlite3.Connection:
        conn = _make_conn()
        for index in range(per_group):
            _insert_edge(conn, "hub", f"t{index}", "anchored_to", 0.9)
            _insert_edge(conn, "hub", f"t{index}", "supports", 0.5)
        return conn

    def test_multi_page_traversal_scales_linearly_not_quadratically(self) -> None:
        def steps_for(per_group: int) -> int:
            conn = self._make(per_group)
            return _sqlite_step_count(lambda c: graph_query(c, ["hub"], depth=1, max_nodes=per_group + 1), conn)

        small_steps = steps_for(250)
        large_steps = steps_for(1_000)

        # Linear (4x the edges) costs about 4x the steps; a re-walk-per-page (quadratic) design
        # costs about 16x (matches the ~23x measured on ad260b892 for a 5x edge increase). A 6x
        # ceiling passes any genuinely linear design with headroom and fails loudly on a
        # regression back to the per-page rescan.
        assert large_steps < small_steps * 6, (
            f"multi-page steps grew from {small_steps} (250/group) to {large_steps} (1000/group) -- "
            "a 4x edge increase should cost ~4x the work, not ~16x (per-page rescan regression)"
        )

    def test_edges_scanned_equals_rows_actually_fetched(self) -> None:
        """Every row either bounded query returns must land in the budget counter -- no row
        SQLite hands back is invisible to it, and nothing is double-counted."""
        per_group = 300
        conn = _CountingConn(self._make(per_group))

        results = graph_query(conn, ["hub"], depth=1, max_nodes=per_group + 1)  # type: ignore[arg-type]

        assert len(results) == per_group  # the "supports" group is entirely already-visited duplicates
        # The full adjacency (2 * per_group) fits well inside edge_scan_budget ((per_group + 1) *
        # 50), so the traversal runs to natural exhaustion rather than an early budget cutoff --
        # every row of the real adjacency must have been fetched exactly once.
        assert conn.edge_rows_fetched == per_group * 2, (
            f"fetched {conn.edge_rows_fetched} rows for a {per_group * 2}-edge adjacency -- "
            "rows examined and edges_scanned have drifted apart"
        )

    def test_both_bounded_queries_use_index_seeks_with_no_temp_btree(self) -> None:
        conn = self._make(50)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)

        # per_group + 1: the first page fills entirely from "anchored_to" (all 50 are new) plus one
        # duplicate "supports" row, then every later page's `remaining` (1) forces a small LIMIT
        # that only "supports" duplicates can answer -- so query A ("continue within current type")
        # is exercised repeatedly across many pages, not just query B on a single call spanning
        # both groups.
        graph_query(conn, ["hub"], depth=1, max_nodes=51)

        conn.set_trace_callback(None)
        edge_statements = [s for s in statements if "memory_graph_edges" in s and "target_id" in s]
        # conn.set_trace_callback reports SQL with bound values already substituted, not `?`
        # placeholders, so the two query shapes are told apart by their comparison operator.
        same_type = [s for s in edge_statements if "edge_type = " in s]
        later_types = [s for s in edge_statements if "edge_type > " in s]
        assert same_type, "expected at least one 'continue within current type' query"
        assert later_types, "expected at least one 'move to later types' query"
        for statement in edge_statements:
            plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
            plan_text = " ".join(str(row) for row in plan_rows).upper()
            assert "TEMP B-TREE" not in plan_text, f"paging query is not a bounded seek: {plan_text}"
            assert "SEARCH " in plan_text, f"paging query is not an index seek: {plan_text}"
