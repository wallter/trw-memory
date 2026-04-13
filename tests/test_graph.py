"""Tests for trw_memory.graph -- knowledge graph edge creation, traversal,
cross-validation, and importance operations.

Uses an in-memory SQLite database with the DDL from sqlite_backend.py.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.graph import (
    IMPORTANCE_BOOST,
    _merge_cross_validated_entry,
    _safe_cosine_similarity,
    apply_importance_boost,
    apply_importance_decay,
    create_consolidation_edges,
    create_similarity_edges,
    create_tag_cooccurrence_edges,
    detect_cross_validation,
    graph_query,
    memory_decay_pass,
)
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# DDL (copied from sqlite_backend.py for in-memory test setup)
# ---------------------------------------------------------------------------

_CREATE_MEMORIES = """
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
    access_count      INTEGER DEFAULT 0,
    q_value           REAL DEFAULT 0.5,
    q_observations    INTEGER DEFAULT 0,
    source            TEXT DEFAULT 'agent',
    source_identity   TEXT DEFAULT '',
    merged_from       TEXT DEFAULT '[]',
    consolidated_from TEXT DEFAULT '[]',
    consolidated_into TEXT,
    metadata          TEXT DEFAULT '{}',
    vector_clock      TEXT DEFAULT '{}',
    remote_id         TEXT,
    published_to_platform INTEGER DEFAULT 0,
    pending_delete    INTEGER DEFAULT 0,
    cross_validated   INTEGER DEFAULT 0,
    outcome_history   TEXT DEFAULT '[]'
)
"""

_CREATE_GRAPH_EDGES = """
CREATE TABLE IF NOT EXISTS memory_graph_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    created_at  TEXT NOT NULL,
    edge_metadata TEXT DEFAULT '{}',
    UNIQUE (source_id, target_id, edge_type)
)
"""

_CREATE_IDX_MGE_SOURCE = "CREATE INDEX IF NOT EXISTS idx_mge_source ON memory_graph_edges(source_id, edge_type)"
_CREATE_IDX_MGE_TARGET = "CREATE INDEX IF NOT EXISTS idx_mge_target ON memory_graph_edges(target_id, edge_type)"


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the required schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute(_CREATE_MEMORIES)
    conn.execute(_CREATE_GRAPH_EDGES)
    conn.execute(_CREATE_IDX_MGE_SOURCE)
    conn.execute(_CREATE_IDX_MGE_TARGET)
    conn.commit()
    return conn


def _make_entry(
    entry_id: str,
    content: str = "content",
    tags: list[str] | None = None,
    importance: float = 0.5,
    consolidated_from: list[str] | None = None,
    cross_validated: bool = False,
    outcome_history: list[str] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        tags=tags or [],
        importance=importance,
        consolidated_from=consolidated_from or [],
        cross_validated=cross_validated,
        outcome_history=outcome_history or [],
    )


def _insert_memory_row(conn: sqlite3.Connection, entry_id: str, **overrides: object) -> None:
    """Insert a minimal memory row into the in-memory DB."""
    now = datetime.now(timezone.utc).isoformat()
    defaults = {
        "id": entry_id,
        "content": "content",
        "detail": "",
        "tags": "[]",
        "evidence": "[]",
        "importance": 0.5,
        "status": "active",
        "recurrence": 1,
        "namespace": "default",
        "created_at": now,
        "updated_at": now,
        "last_accessed_at": None,
        "access_count": 0,
        "q_value": 0.5,
        "q_observations": 0,
        "source": "agent",
        "source_identity": "",
        "merged_from": "[]",
        "consolidated_from": "[]",
        "consolidated_into": None,
        "metadata": "{}",
        "vector_clock": "{}",
        "remote_id": None,
        "published_to_platform": 0,
        "pending_delete": 0,
        "cross_validated": 0,
        "outcome_history": "[]",
    }
    defaults.update(overrides)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join(["?"] * len(defaults))
    conn.execute(
        f"INSERT INTO memories ({cols}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    conn.commit()


def _insert_edge(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
) -> None:
    """Insert an edge directly for test setup."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_graph_edges (source_id, target_id, edge_type, weight, created_at) VALUES (?, ?, ?, ?, ?)",
        (source_id, target_id, edge_type, weight, now),
    )
    conn.commit()


def _count_edges(conn: sqlite3.Connection) -> int:
    """Count total edges in the graph table."""
    row = conn.execute("SELECT COUNT(*) FROM memory_graph_edges").fetchone()
    return int(row[0]) if row else 0


def _merge_cross_validation_in_subprocess(storage_path: str, project_id: str) -> None:
    cfg = MemoryConfig(storage_backend="sqlite", storage_path=storage_path)
    with create_backend_from_config(cfg, "project:default") as storage:
        _merge_cross_validated_entry(storage, "e1", project_id, 0.97)


# Vectors for testing: v1 and v2 are nearly identical (high cosine sim)
_V1 = [1.0, 0.0, 0.0, 0.0]
_V2 = [0.99, 0.1, 0.0, 0.0]  # sim with V1 ~ 0.995
_V3 = [0.0, 0.0, 0.0, 1.0]  # orthogonal to V1


# ===========================================================================
# Edge Creation Tests
# ===========================================================================


class TestCreateSimilarityEdges:
    def test_creates_bidirectional_edges_above_threshold(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        candidates = [("e2", _V2)]

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=candidates)

        assert count == 2  # bidirectional
        edges = conn.execute("SELECT source_id, target_id, edge_type FROM memory_graph_edges").fetchall()
        assert len(edges) == 2
        sources = {(e[0], e[1]) for e in edges}
        assert ("e1", "e2") in sources
        assert ("e2", "e1") in sources

    def test_no_edges_at_or_below_threshold(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        # V1 and V3 are orthogonal (sim = 0.0), well below 0.75
        candidates = [("e2", _V3)]

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=candidates)

        assert count == 0
        assert _count_edges(conn) == 0

    def test_no_op_when_embedding_is_none(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        candidates = [("e2", _V2)]

        count = create_similarity_edges(entry, conn, embedding=None, candidate_embeddings=candidates)
        assert count == 0
        assert _count_edges(conn) == 0

    def test_no_op_when_candidates_is_none(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=None)
        assert count == 0

    def test_skips_self_reference(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        # Same entry id in candidates
        candidates = [("e1", _V1)]

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=candidates)
        assert count == 0


class TestCreateTagCooccurrenceEdges:
    def test_correct_jaccard_weight(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=["a", "b", "c"])
        candidates = [_make_entry("e2", tags=["b", "c", "d"])]

        count = create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)

        assert count == 2  # bidirectional
        row = conn.execute(
            "SELECT weight FROM memory_graph_edges WHERE source_id = 'e1' AND target_id = 'e2'"
        ).fetchone()
        # Jaccard: {b,c} / {a,b,c,d} = 2/4 = 0.5
        assert row is not None
        assert abs(row[0] - 0.5) < 0.001

    def test_updates_existing_edge_weight(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=["a", "b"])
        candidates = [_make_entry("e2", tags=["a", "b"])]

        # First creation
        create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)

        # Update with different tags (simulate changing tags)
        entry2 = _make_entry("e1", tags=["a", "b", "c"])
        candidates2 = [_make_entry("e2", tags=["a", "b", "d"])]
        create_tag_cooccurrence_edges(entry2, conn, candidate_entries=candidates2)

        edges = conn.execute(
            "SELECT weight FROM memory_graph_edges WHERE source_id = 'e1' AND target_id = 'e2' AND edge_type = 'tag_cooccurrence'"
        ).fetchall()
        assert len(edges) == 1  # upsert, not duplicate
        # Jaccard: {a,b} / {a,b,c,d} = 2/4 = 0.5
        assert abs(edges[0][0] - 0.5) < 0.001

    def test_no_edge_for_less_than_2_shared_tags(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=["a", "b"])
        candidates = [_make_entry("e2", tags=["a", "c"])]

        count = create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)
        assert count == 0

    def test_no_edge_for_empty_tags(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=[])
        candidates = [_make_entry("e2", tags=["a", "b"])]

        count = create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)
        assert count == 0

    def test_no_edge_when_candidate_has_empty_tags(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", tags=["a", "b"])
        candidates = [_make_entry("e2", tags=[])]

        count = create_tag_cooccurrence_edges(entry, conn, candidate_entries=candidates)
        assert count == 0


class TestCreateConsolidationEdges:
    def test_creates_edges_for_valid_source_ids(self) -> None:
        conn = _make_conn()
        # Insert source entries in the DB
        _insert_memory_row(conn, "src1")
        _insert_memory_row(conn, "src2")

        entry = _make_entry("consolidated-1", consolidated_from=["src1", "src2"])
        count = create_consolidation_edges(entry, conn)

        assert count == 2
        edges = conn.execute(
            "SELECT target_id, edge_type, weight FROM memory_graph_edges WHERE source_id = 'consolidated-1'"
        ).fetchall()
        assert len(edges) == 2
        for edge in edges:
            assert edge[1] == "consolidation"
            assert edge[2] == 1.0

    def test_skips_missing_source_ids(self) -> None:
        conn = _make_conn()
        # Only insert src1, not src2
        _insert_memory_row(conn, "src1")

        entry = _make_entry("consolidated-1", consolidated_from=["src1", "src2"])
        count = create_consolidation_edges(entry, conn)

        assert count == 1  # only src1 found

    def test_no_op_when_no_consolidated_from(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1", consolidated_from=[])

        count = create_consolidation_edges(entry, conn)
        assert count == 0


# ===========================================================================
# Graph Traversal Tests
# ===========================================================================


class TestGraphQuery:
    def test_depth_1_returns_direct_neighbors_only(self) -> None:
        conn = _make_conn()
        # A -> B -> C
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)

        results = graph_query(conn, ["A"], depth=1)

        ids = {r["id"] for r in results}
        assert ids == {"B"}
        assert all(r["depth"] == 1 for r in results)

    def test_depth_2_returns_neighbors_of_neighbors(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)

        results = graph_query(conn, ["A"], depth=2)

        ids = {r["id"] for r in results}
        assert ids == {"B", "C"}
        depths = {r["id"]: r["depth"] for r in results}
        assert depths["B"] == 1
        assert depths["C"] == 2

    def test_excludes_root_nodes(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "A", "similarity", 0.9)  # back-edge

        results = graph_query(conn, ["A"], depth=2)

        ids = {r["id"] for r in results}
        assert "A" not in ids
        assert "B" in ids

    def test_deduplicates_multipath_results(self) -> None:
        conn = _make_conn()
        # A -> B, A -> C, B -> D, C -> D (two paths to D)
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "A", "C", "similarity", 0.8)
        _insert_edge(conn, "B", "D", "similarity", 0.7)
        _insert_edge(conn, "C", "D", "similarity", 0.7)

        results = graph_query(conn, ["A"], depth=2)

        id_list = [r["id"] for r in results]
        # D should appear exactly once
        assert id_list.count("D") == 1

    def test_edge_types_filter(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "A", "C", "tag_cooccurrence", 0.5)

        results = graph_query(conn, ["A"], depth=1, edge_types=["similarity"])

        ids = {r["id"] for r in results}
        assert ids == {"B"}  # C excluded because it's tag_cooccurrence

    def test_clamps_depth_above_max(self) -> None:
        conn = _make_conn()
        # Chain: A -> B -> C -> D -> E (depth 4)
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)
        _insert_edge(conn, "C", "D", "similarity", 0.7)
        _insert_edge(conn, "D", "E", "similarity", 0.6)

        results = graph_query(conn, ["A"], depth=10)

        # Clamped to MAX_TRAVERSAL_DEPTH (3), so E at depth 4 should not appear
        ids = {r["id"] for r in results}
        assert "B" in ids
        assert "C" in ids
        assert "D" in ids
        assert "E" not in ids

    def test_empty_root_ids_returns_empty(self) -> None:
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)

        results = graph_query(conn, [], depth=2)
        assert results == []

    def test_isolated_node_returns_empty(self) -> None:
        conn = _make_conn()
        # No edges from X

        results = graph_query(conn, ["X"], depth=2)
        assert results == []


# ===========================================================================
# Importance Operation Tests
# ===========================================================================


class TestApplyImportanceBoost:
    def test_adds_default_boost_and_records_history(self) -> None:
        entry = _make_entry("e1", importance=0.5)
        result = apply_importance_boost(entry)

        assert abs(result.importance - (0.5 + IMPORTANCE_BOOST)) < 0.001
        assert len(result.outcome_history) == 1
        assert "importance_boost" in result.outcome_history[0]
        assert f"delta=+{IMPORTANCE_BOOST:.2f}" in result.outcome_history[0]

    def test_caps_at_1_0(self) -> None:
        entry = _make_entry("e1", importance=0.99)
        result = apply_importance_boost(entry, delta=0.1)

        assert result.importance == 1.0

    def test_sets_cross_validated_true(self) -> None:
        entry = _make_entry("e1", cross_validated=False)
        result = apply_importance_boost(entry)

        assert result.cross_validated is True

    def test_preserves_existing_outcome_history(self) -> None:
        entry = _make_entry("e1", outcome_history=["previous_event"])
        result = apply_importance_boost(entry)

        assert len(result.outcome_history) == 2
        assert result.outcome_history[0] == "previous_event"
        assert "importance_boost" in result.outcome_history[1]

    def test_concurrent_cross_validation_merges_both_project_boosts(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "project:default") as storage:
            backend = storage
            backend.store(_make_entry("e1", importance=0.5))

            threads = [
                threading.Thread(
                    target=_merge_cross_validated_entry,
                    args=(backend, "e1", project_id, 0.97),
                )
                for project_id in ("project-a", "project-b")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            updated = backend.get("e1")
            assert updated is not None
            assert updated.importance == 0.6
            assert updated.cross_validated is True
            assert sum("importance_boost" in item for item in updated.outcome_history) == 2
            assert sum("cross_validated:project_id=" in item for item in updated.outcome_history) == 2

    def test_cross_process_cross_validation_merges_both_project_boosts(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "project:default") as storage:
            storage.store(_make_entry("e1", importance=0.5))

        processes = [
            multiprocessing.Process(
                target=_merge_cross_validation_in_subprocess,
                args=(str(tmp_path), project_id),
            )
            for project_id in ("project-a", "project-b")
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0

        with create_backend_from_config(cfg, "project:default") as storage:
            updated = storage.get("e1")
            assert updated is not None
            assert updated.importance == 0.6
            assert updated.cross_validated is True
            assert sum("importance_boost" in item for item in updated.outcome_history) == 2
            assert sum("cross_validated:project_id=" in item for item in updated.outcome_history) == 2

    def test_twenty_sequential_boosts_cap_at_one_without_drift(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path))

        with create_backend_from_config(cfg, "project:default") as storage:
            storage.store(_make_entry("e1", importance=0.0))
            observed: list[float] = []

            for idx in range(20):
                updated, applied = _merge_cross_validated_entry(storage, "e1", f"project-{idx}", 0.97)
                assert applied is True
                assert updated is not None
                observed.append(updated.importance)

            assert observed[-1] == 1.0
            assert max(observed) == 1.0
            assert all(value <= 1.0 for value in observed)


class TestApplyImportanceDecay:
    def test_reduces_by_delta(self) -> None:
        entry = _make_entry("e1", importance=0.5)
        result = apply_importance_decay(entry, delta=0.1)

        assert abs(result.importance - 0.4) < 0.001
        assert len(result.outcome_history) == 1
        assert "importance_decay" in result.outcome_history[0]

    def test_floors_at_0_0(self) -> None:
        entry = _make_entry("e1", importance=0.05)
        result = apply_importance_decay(entry, delta=0.1)

        assert result.importance == 0.0


class TestMemoryDecayPass:
    def test_processes_qualifying_entries(self) -> None:
        conn = _make_conn()
        # Insert cross-validated entries with old last_accessed_at
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=1, last_accessed_at=old_date, importance=0.8)
        _insert_memory_row(conn, "e2", cross_validated=1, last_accessed_at=old_date, importance=0.6)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 2
        assert result["total_decayed"] == 2

        # Verify importance was reduced
        row = conn.execute("SELECT importance FROM memories WHERE id = 'e1'").fetchone()
        assert row is not None
        assert abs(row[0] - 0.7) < 0.001  # 0.8 - 0.1

        history_row = conn.execute("SELECT outcome_history FROM memories WHERE id = 'e1'").fetchone()
        assert history_row is not None
        assert "new_value=0.7000" in str(history_row[0])

    def test_skips_non_cross_validated_entries(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=0, last_accessed_at=old_date, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 0
        # Importance unchanged
        row = conn.execute("SELECT importance FROM memories WHERE id = 'e1'").fetchone()
        assert row is not None
        assert abs(row[0] - 0.8) < 0.001

    def test_respects_batch_size_limit(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        for i in range(10):
            _insert_memory_row(conn, f"e{i}", cross_validated=1, last_accessed_at=old_date, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=3)

        assert result["processed"] == 3
        assert result["remaining"] == 7  # 10 - 3

    def test_clamps_batch_size_to_1000(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        insert_sql = (
            "INSERT INTO memories ("
            "id, content, created_at, updated_at, last_accessed_at, cross_validated, importance"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        conn.executemany(
            insert_sql,
            [(f"e{i}", "content", old_date, old_date, old_date, 1, 0.8) for i in range(1_500)],
        )
        conn.commit()

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=2_000)

        assert result["processed"] == 1_000
        assert result["remaining"] == 500

    def test_rejects_non_positive_batch_size(self) -> None:
        conn = _make_conn()

        with pytest.raises(ValueError, match="batch_size must be positive"):
            memory_decay_pass(conn, cutoff_days=90, batch_size=0)

    def test_skips_recently_accessed_entries(self) -> None:
        conn = _make_conn()
        recent = datetime.now(timezone.utc).isoformat()
        _insert_memory_row(conn, "e1", cross_validated=1, last_accessed_at=recent, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 0

    def test_skips_never_accessed_fresh_entries(self) -> None:
        conn = _make_conn()
        recent = datetime.now(timezone.utc).isoformat()
        _insert_memory_row(conn, "e1", cross_validated=1, created_at=recent, last_accessed_at=None, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 0

    def test_decays_never_accessed_old_entries_by_created_at(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=1, created_at=old_date, last_accessed_at=None, importance=0.8)

        result = memory_decay_pass(conn, cutoff_days=90)

        assert result["processed"] == 1
        row = conn.execute("SELECT importance FROM memories WHERE id = 'e1'").fetchone()
        assert row is not None
        assert abs(row[0] - 0.7) < 0.001

    def test_decay_floors_at_zero(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"
        _insert_memory_row(conn, "e1", cross_validated=1, last_accessed_at=old_date, importance=0.05)

        memory_decay_pass(conn, cutoff_days=90)

        row = conn.execute("SELECT importance FROM memories WHERE id = 'e1'").fetchone()
        assert row is not None
        assert row[0] == 0.0  # MAX(0.05 - 0.1, 0.0) = 0.0


# ===========================================================================
# Cross-Validation Tests
# ===========================================================================


class TestDetectCrossValidation:
    def test_true_above_threshold(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        # Same vector = sim 1.0, well above 0.92
        remote = [("remote-1", "proj-b", _V1)]

        assert detect_cross_validation(entry, conn, embedding=_V1, remote_entries=remote)

    def test_false_below_threshold(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        # Orthogonal = sim 0.0
        remote = [("remote-1", "proj-b", _V3)]

        assert not detect_cross_validation(entry, conn, embedding=_V1, remote_entries=remote)

    def test_false_when_no_embedding(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")
        remote = [("remote-1", "proj-b", _V1)]

        assert not detect_cross_validation(entry, conn, embedding=None, remote_entries=remote)

    def test_false_when_no_remote_entries(self) -> None:
        conn = _make_conn()
        entry = _make_entry("e1")

        assert not detect_cross_validation(entry, conn, embedding=_V1, remote_entries=None)


# ===========================================================================
# Cosine Similarity Helper Tests
# ===========================================================================


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert abs(_safe_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 0.001

    def test_orthogonal_vectors(self) -> None:
        assert abs(_safe_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 0.001

    def test_empty_vectors(self) -> None:
        assert _safe_cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert _safe_cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert _safe_cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ===========================================================================
# FR05 (PRD-QUAL-038): Graph Edge Case Tests
# ===========================================================================


class TestGraphQueryEdgeCases:
    """FR05: graph_query edge cases -- cycles, empty roots, depth clamping."""

    def test_graph_query_circular_reference(self) -> None:
        """A->B->C->A cycle at depth=3 -> terminates, discovers B and C."""
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)
        _insert_edge(conn, "C", "A", "similarity", 0.7)  # back to root

        results = graph_query(conn, ["A"], depth=3)
        ids = {r["id"] for r in results}
        # A is a root node, so it won't appear in results; B and C discovered
        assert "B" in ids
        assert "C" in ids
        assert "A" not in ids
        # Should terminate without infinite loop (visited set prevents re-visiting)
        assert len(results) == 2

    def test_graph_query_empty_root_ids(self) -> None:
        """root_ids=[] -> empty list."""
        conn = _make_conn()
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        results = graph_query(conn, [], depth=2)
        assert results == []

    def test_graph_query_depth_clamping(self) -> None:
        """depth=10 -> clamped to MAX_TRAVERSAL_DEPTH (3)."""
        conn = _make_conn()
        # Chain: A->B->C->D->E (depth 4)
        _insert_edge(conn, "A", "B", "similarity", 0.9)
        _insert_edge(conn, "B", "C", "similarity", 0.8)
        _insert_edge(conn, "C", "D", "similarity", 0.7)
        _insert_edge(conn, "D", "E", "similarity", 0.6)

        results = graph_query(conn, ["A"], depth=10)
        ids = {r["id"] for r in results}
        # Clamped to 3, so B(1), C(2), D(3) are found; E(4) is not
        assert "B" in ids
        assert "C" in ids
        assert "D" in ids
        assert "E" not in ids

    def test_graph_query_disconnected_node(self) -> None:
        """Root with no edges -> empty results."""
        conn = _make_conn()
        # No edges from "lonely"
        results = graph_query(conn, ["lonely"], depth=2)
        assert results == []


class TestSafeCosineSimilarityEdgeCases:
    """FR05: _safe_cosine_similarity edge cases."""

    def test_safe_cosine_similarity_dimension_mismatch(self) -> None:
        """Different dimensions -> returns 0.0 (graceful degradation)."""
        result = _safe_cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        assert result == 0.0

    def test_safe_cosine_similarity_zero_vectors(self) -> None:
        """Zero vectors -> returns 0.0."""
        result = _safe_cosine_similarity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert result == 0.0


class TestCreateSimilarityEdgesEdgeCases:
    """FR05: create_similarity_edges skips self-references."""

    def test_create_similarity_edges_skip_self(self) -> None:
        """Candidate same as entry -> skipped."""
        conn = _make_conn()
        entry = _make_entry("self-ref")
        # Candidate has the same ID as the entry
        candidates = [("self-ref", _V1)]

        count = create_similarity_edges(entry, conn, embedding=_V1, candidate_embeddings=candidates)
        assert count == 0
        assert _count_edges(conn) == 0


class TestMemoryDecayPassBatch:
    """FR05: memory_decay_pass processes entries in batches."""

    def test_memory_decay_pass_batch(self) -> None:
        """Entries unused for cutoff_days get decay applied."""
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"

        # Insert 5 cross-validated entries with old access time
        for i in range(5):
            _insert_memory_row(
                conn,
                f"decay-{i}",
                cross_validated=1,
                last_accessed_at=old_date,
                importance=0.8,
            )

        result = memory_decay_pass(conn, cutoff_days=90, batch_size=3)

        # Only 3 processed due to batch_size
        assert result["processed"] == 3
        assert result["remaining"] == 2
        assert result["total_decayed"] == 3

        # Verify the first 3 had importance reduced
        decayed_count = 0
        for i in range(5):
            row = conn.execute("SELECT importance FROM memories WHERE id = ?", (f"decay-{i}",)).fetchone()
            if row and abs(row[0] - 0.7) < 0.001:  # 0.8 - 0.1 = 0.7
                decayed_count += 1
        assert decayed_count == 3

    def test_memory_decay_pass_peak_memory_under_512mb_for_50000_entries(self) -> None:
        conn = _make_conn()
        old_date = "2020-01-01T00:00:00+00:00"

        insert_sql = (
            "INSERT INTO memories ("
            "id, content, created_at, updated_at, last_accessed_at, cross_validated, importance"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        for start in range(0, 50_000, 5_000):
            rows = [
                (f"decay-{i}", "content", old_date, old_date, old_date, 1, 0.8)
                for i in range(start, start + 5_000)
            ]
            conn.executemany(insert_sql, rows)
        conn.commit()

        tracemalloc.start()
        try:
            result = memory_decay_pass(conn, cutoff_days=90, batch_size=1000)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert result["processed"] == 1000
        assert result["remaining"] == 49_000
        assert result["total_decayed"] == 1000
        assert peak < 512 * 1024 * 1024


class TestGraphQueryPerformance:
    def test_graph_query_p95_under_100ms_for_1000_nodes_and_5000_edges(self) -> None:
        conn = _make_conn()
        for node in range(1000):
            for edge_idx in range(5):
                target = (node + edge_idx + 1) % 1000
                _insert_edge(conn, f"n{node}", f"n{target}", "similarity", 0.9)

        timings_ms: list[float] = []
        for _ in range(20):
            started = time.perf_counter()
            results = graph_query(conn, ["n0"], depth=3)
            timings_ms.append((time.perf_counter() - started) * 1000)
            assert results

        p95_index = max(int(len(timings_ms) * 0.95) - 1, 0)
        p95_ms = sorted(timings_ms)[p95_index]
        assert p95_ms < 100
