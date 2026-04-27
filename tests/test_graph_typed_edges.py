"""Tests for PRD-CORE-107: Typed Graph Relationships + Conflict Resolution.

Tests cover:
- FR01: Typed edge schema extension (VALID_EDGE_TYPES, metadata, validation)
- FR02: Automatic co_anchored edge creation
- FR03: Conflict resolution (get_conflicts, filter_conflicts)
- FR04: Cluster detection
- FR05: Impact propagation
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from trw_memory.graph import (
    VALID_EDGE_TYPES,
    _upsert_edge,
    create_co_anchored_edges,
    detect_clusters,
    filter_conflicts,
    get_conflicts,
    propagate_impact,
)
from trw_memory.models.memory import Anchor, MemoryEntry
from trw_memory.storage._schema import ensure_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the full schema."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _make_entry(
    entry_id: str,
    content: str = "content",
    tags: list[str] | None = None,
    importance: float = 0.5,
    anchors: list[Anchor] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        tags=tags or [],
        importance=importance,
        anchors=anchors or [],
    )


def _insert_memory_row(
    conn: sqlite3.Connection,
    entry_id: str,
    *,
    importance: float = 0.5,
    anchors_json: str = "[]",
    outcome_history_json: str = "[]",
) -> None:
    """Insert a minimal memory row for graph tests."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memories (id, content, created_at, updated_at, importance, anchors, outcome_history) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (entry_id, "content", now, now, importance, anchors_json, outcome_history_json),
    )
    conn.commit()


def _insert_edge(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    metadata: dict[str, str] | None = None,
) -> None:
    """Insert an edge directly for test setup."""
    now = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata) if metadata else "{}"
    conn.execute(
        "INSERT INTO memory_graph_edges (source_id, target_id, edge_type, weight, created_at, edge_metadata) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source_id, target_id, edge_type, weight, now, meta_json),
    )
    conn.commit()


def _count_edges(conn: sqlite3.Connection, edge_type: str | None = None) -> int:
    """Count edges, optionally filtered by type."""
    if edge_type:
        row = conn.execute(
            "SELECT COUNT(*) FROM memory_graph_edges WHERE edge_type = ?",
            (edge_type,),
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) FROM memory_graph_edges").fetchone()
    return int(row[0]) if row else 0


def _get_edge_metadata(conn: sqlite3.Connection, source_id: str, target_id: str, edge_type: str) -> dict[str, str]:
    """Get edge metadata for a specific edge."""
    row = conn.execute(
        "SELECT edge_metadata FROM memory_graph_edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
        (source_id, target_id, edge_type),
    ).fetchone()
    if row and row[0]:
        return json.loads(row[0])  # type: ignore[no-any-return]
    return {}


# ===========================================================================
# FR01: Typed Edge Schema Extension
# ===========================================================================


class TestValidEdgeTypes:
    def test_valid_edge_types_frozenset(self) -> None:
        """VALID_EDGE_TYPES contains all 13 required edge types."""
        expected = {
            "similarity",
            "tag_cooccurrence",
            "consolidation",
            "anchored_to",
            "related_to",
            "same_root_cause",
            "depends_on",
            "produced",
            "motivated_by",
            "co_anchored",
            "supersedes",
            "evidence_for",
            "conflicts_with",
        }
        assert isinstance(VALID_EDGE_TYPES, frozenset)
        assert VALID_EDGE_TYPES == expected
        assert len(VALID_EDGE_TYPES) == 13

    def test_upsert_edge_with_metadata(self) -> None:
        """Edge with metadata roundtrips through DB correctly."""
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()
        metadata = {"anchor_file": "src/graph.py", "reason": "shared_anchor"}

        _upsert_edge(conn, "e1", "e2", "co_anchored", 0.8, now, metadata=metadata)
        conn.commit()

        row = conn.execute(
            "SELECT edge_metadata FROM memory_graph_edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
            ("e1", "e2", "co_anchored"),
        ).fetchone()
        assert row is not None
        parsed = json.loads(row[0])
        assert parsed["anchor_file"] == "src/graph.py"
        assert parsed["reason"] == "shared_anchor"

    def test_upsert_edge_without_metadata(self) -> None:
        """Edge without metadata stores empty JSON object."""
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()

        _upsert_edge(conn, "e1", "e2", "similarity", 0.9, now)
        conn.commit()

        row = conn.execute(
            "SELECT edge_metadata FROM memory_graph_edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
            ("e1", "e2", "similarity"),
        ).fetchone()
        assert row is not None
        parsed = json.loads(row[0])
        assert parsed == {}

    def test_upsert_edge_invalid_type_raises(self) -> None:
        """Invalid edge type raises ValueError."""
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()

        with pytest.raises(ValueError, match="Invalid edge type"):
            _upsert_edge(conn, "e1", "e2", "nonexistent_type", 0.5, now)

    def test_upsert_edge_all_valid_types_accepted(self) -> None:
        """All 13 valid edge types are accepted without error."""
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()

        for i, edge_type in enumerate(sorted(VALID_EDGE_TYPES)):
            _upsert_edge(conn, f"src-{i}", f"tgt-{i}", edge_type, 0.5, now)
        conn.commit()
        assert _count_edges(conn) == 13

    def test_edge_metadata_migration(self) -> None:
        """Schema migration adds edge_metadata column to memory_graph_edges."""
        conn = _make_conn()
        # ensure_schema already ran via _make_conn; verify column exists
        cursor = conn.execute("PRAGMA table_info(memory_graph_edges)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "edge_metadata" in columns

    def test_upsert_edge_metadata_size_limit(self) -> None:
        """Edge metadata exceeding 4096 bytes raises ValueError."""
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()
        # Create metadata that exceeds 4096 bytes when serialized
        large_metadata = {"key": "x" * 5000}

        with pytest.raises(ValueError, match="exceeds 4096 byte limit"):
            _upsert_edge(conn, "e1", "e2", "related_to", 0.5, now, metadata=large_metadata)

    def test_upsert_edge_metadata_within_limit(self) -> None:
        """Edge metadata within 4096 bytes is accepted."""
        conn = _make_conn()
        now = datetime.now(timezone.utc).isoformat()
        # Create metadata just under the limit
        ok_metadata = {"key": "x" * 100}

        _upsert_edge(conn, "e1", "e2", "related_to", 0.5, now, metadata=ok_metadata)
        conn.commit()
        assert _count_edges(conn) == 1


# ===========================================================================
# FR02: Automatic co_anchored Edge Creation
# ===========================================================================


class TestCoAnchoredEdges:
    def test_co_anchored_edge_creation(self) -> None:
        """Two entries sharing an anchor file get a co_anchored edge."""
        conn = _make_conn()
        anchor_data = [{"file": "src/graph.py", "symbol_name": "func_a", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_data))
        _insert_memory_row(conn, "e2", anchors_json=json.dumps(anchor_data))

        count = create_co_anchored_edges(conn, "e1", ["src/graph.py"])
        assert count >= 1
        assert _count_edges(conn, "co_anchored") >= 1

        # Verify metadata contains anchor_file
        meta = _get_edge_metadata(conn, "e1", "e2", "co_anchored")
        assert meta["anchor_file"] == "src/graph.py"

    def test_co_anchored_no_match(self) -> None:
        """No co_anchored edges created when no other entries share the anchor."""
        conn = _make_conn()
        anchor_data = [{"file": "src/unique.py", "symbol_name": "func_a", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_data))

        count = create_co_anchored_edges(conn, "e1", ["src/unique.py"])
        assert count == 0

    def test_co_anchored_cap_per_file(self) -> None:
        """Max edges per file is respected (max_per_file parameter)."""
        conn = _make_conn()
        anchor_data = [{"file": "src/big.py", "symbol_name": "func", "symbol_type": "function"}]

        # Create 10 entries all sharing same anchor file
        for i in range(10):
            _insert_memory_row(conn, f"e{i}", anchors_json=json.dumps(anchor_data))

        # Cap at 3 per file
        count = create_co_anchored_edges(conn, "e0", ["src/big.py"], max_per_file=3)
        assert count <= 3

    def test_co_anchored_multiple_files(self) -> None:
        """co_anchored edges created across multiple anchor files."""
        conn = _make_conn()
        # e1 has anchors in both a.py and b.py
        anchors_e1 = [
            {"file": "src/a.py", "symbol_name": "f", "symbol_type": "function"},
            {"file": "src/b.py", "symbol_name": "g", "symbol_type": "function"},
        ]
        anchor_a = [{"file": "src/a.py", "symbol_name": "f", "symbol_type": "function"}]
        anchor_b = [{"file": "src/b.py", "symbol_name": "g", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchors_e1))
        _insert_memory_row(conn, "e2", anchors_json=json.dumps(anchor_a))
        _insert_memory_row(conn, "e3", anchors_json=json.dumps(anchor_b))

        count = create_co_anchored_edges(conn, "e1", ["src/a.py", "src/b.py"])
        # e1 shares src/a.py with e2 and src/b.py with e3
        assert count == 2

    def test_co_anchored_skips_self(self) -> None:
        """co_anchored does not create self-edges."""
        conn = _make_conn()
        anchor_data = [{"file": "src/x.py", "symbol_name": "f", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_data))

        count = create_co_anchored_edges(conn, "e1", ["src/x.py"])
        assert count == 0


# ===========================================================================
# FR03: Conflict Resolution
# ===========================================================================


class TestConflictResolution:
    def test_get_conflicts(self) -> None:
        """conflicts_with edge is returned by get_conflicts."""
        conn = _make_conn()
        _insert_memory_row(conn, "e1")
        _insert_memory_row(conn, "e2")
        _insert_edge(
            conn,
            "e1",
            "e2",
            "conflicts_with",
            1.0,
            metadata={"reason": "contradicting"},
        )

        conflicts = get_conflicts(conn, "e1")
        assert len(conflicts) == 1
        assert conflicts[0]["source_id"] == "e1"
        assert conflicts[0]["target_id"] == "e2"

    def test_get_conflicts_bidirectional(self) -> None:
        """get_conflicts finds edges where entry is source OR target."""
        conn = _make_conn()
        _insert_memory_row(conn, "e1")
        _insert_memory_row(conn, "e2")
        _insert_edge(conn, "e2", "e1", "conflicts_with", 1.0)

        conflicts = get_conflicts(conn, "e1")
        assert len(conflicts) == 1
        assert conflicts[0]["source_id"] == "e2"
        assert conflicts[0]["target_id"] == "e1"

    def test_get_conflicts_empty(self) -> None:
        """No conflicts_with edges returns empty list."""
        conn = _make_conn()
        _insert_memory_row(conn, "e1")

        conflicts = get_conflicts(conn, "e1")
        assert conflicts == []

    def test_filter_conflicts_keeps_higher(self) -> None:
        """Higher importance entry is kept, lower is suppressed."""
        conn = _make_conn()
        _insert_memory_row(conn, "e1", importance=0.9)
        _insert_memory_row(conn, "e2", importance=0.3)
        _insert_edge(conn, "e1", "e2", "conflicts_with", 1.0)

        entries = [
            {"id": "e1", "importance": 0.9, "content": "high"},
            {"id": "e2", "importance": 0.3, "content": "low"},
        ]
        result = filter_conflicts(entries, conn)
        ids = [e["id"] for e in result]
        assert "e1" in ids
        assert "e2" not in ids

    def test_filter_conflicts_no_conflicts(self) -> None:
        """Without conflicts_with edges, all entries pass through."""
        conn = _make_conn()
        _insert_memory_row(conn, "e1")
        _insert_memory_row(conn, "e2")

        entries = [
            {"id": "e1", "importance": 0.5, "content": "a"},
            {"id": "e2", "importance": 0.5, "content": "b"},
        ]
        result = filter_conflicts(entries, conn)
        assert len(result) == 2

    def test_filter_conflicts_equal_importance(self) -> None:
        """When importance is equal, both are kept (no suppression)."""
        conn = _make_conn()
        _insert_memory_row(conn, "e1", importance=0.5)
        _insert_memory_row(conn, "e2", importance=0.5)
        _insert_edge(conn, "e1", "e2", "conflicts_with", 1.0)

        entries = [
            {"id": "e1", "importance": 0.5, "content": "a"},
            {"id": "e2", "importance": 0.5, "content": "b"},
        ]
        result = filter_conflicts(entries, conn)
        # Equal importance: keep both (tie-breaking is neutral)
        assert len(result) == 2


# ===========================================================================
# FR04: Cluster Detection
# ===========================================================================


class TestClusterDetection:
    def test_detect_clusters_dense(self) -> None:
        """6 connected nodes with 80% connectivity form a cluster."""
        conn = _make_conn()
        node_ids = [f"n{i}" for i in range(6)]
        for nid in node_ids:
            _insert_memory_row(conn, nid)

        # Create edges for ~80% connectivity
        # 6 nodes: max edges = 6*5/2 = 15; 80% = 12 edges
        now = datetime.now(timezone.utc).isoformat()
        pairs = [(node_ids[i], node_ids[j]) for i in range(6) for j in range(i + 1, 6)]

        # Use 12 of 15 pairs for ~80% connectivity
        for src, tgt in pairs[:12]:
            _upsert_edge(conn, src, tgt, "related_to", 0.7, now)
            _upsert_edge(conn, tgt, src, "related_to", 0.7, now)
        conn.commit()

        clusters = detect_clusters(conn, min_size=5, min_connectivity=0.6)
        assert len(clusters) >= 1
        cluster = clusters[0]
        assert "domain_name" in cluster
        assert "member_ids" in cluster
        assert "connectivity" in cluster
        member_ids = cluster["member_ids"]
        assert isinstance(member_ids, list)
        assert len(member_ids) >= 5
        connectivity = cluster["connectivity"]
        assert isinstance(connectivity, float)
        assert connectivity >= 0.6

    def test_detect_clusters_sparse(self) -> None:
        """4 nodes with 40% connectivity do not form a cluster with min_size=5."""
        conn = _make_conn()
        for i in range(4):
            _insert_memory_row(conn, f"s{i}")

        now = datetime.now(timezone.utc).isoformat()
        # Just 2 edges out of max 6 => ~33% connectivity
        _upsert_edge(conn, "s0", "s1", "related_to", 0.5, now)
        _upsert_edge(conn, "s1", "s0", "related_to", 0.5, now)
        _upsert_edge(conn, "s2", "s3", "related_to", 0.5, now)
        _upsert_edge(conn, "s3", "s2", "related_to", 0.5, now)
        conn.commit()

        clusters = detect_clusters(conn, min_size=5, min_connectivity=0.6)
        assert clusters == []

    def test_detect_clusters_proposes_domain_name(self) -> None:
        """Cluster detection proposes a domain_name string."""
        conn = _make_conn()
        node_ids = [f"c{i}" for i in range(5)]
        for nid in node_ids:
            _insert_memory_row(conn, nid)

        # Fully connected 5-node cluster
        now = datetime.now(timezone.utc).isoformat()
        for i in range(5):
            for j in range(i + 1, 5):
                _upsert_edge(conn, node_ids[i], node_ids[j], "related_to", 0.8, now)
                _upsert_edge(conn, node_ids[j], node_ids[i], "related_to", 0.8, now)
        conn.commit()

        clusters = detect_clusters(conn, min_size=5, min_connectivity=0.6)
        assert len(clusters) >= 1
        assert isinstance(clusters[0]["domain_name"], str)
        assert len(clusters[0]["domain_name"]) > 0


# ===========================================================================
# FR05: Impact Propagation
# ===========================================================================


class TestImpactPropagation:
    def test_propagate_impact_evidence_for(self) -> None:
        """+0.20 delta through evidence_for edge -> 0.3x = +0.06 received."""
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        affected = propagate_impact(conn, "root", 0.20)
        # neighbor should receive 0.20 * 0.3 = 0.06
        neighbor_deltas = [a for a in affected if a[0] == "neighbor"]
        assert len(neighbor_deltas) == 1
        assert abs(neighbor_deltas[0][1] - 0.06) < 1e-6

    def test_propagate_impact_co_anchored(self) -> None:
        """+0.20 delta through co_anchored edge -> 0.2x = +0.04 received."""
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "co_anchored", 0.8)

        affected = propagate_impact(conn, "root", 0.20)
        neighbor_deltas = [a for a in affected if a[0] == "neighbor"]
        assert len(neighbor_deltas) == 1
        assert abs(neighbor_deltas[0][1] - 0.04) < 1e-6

    def test_propagate_impact_max_depth(self) -> None:
        """3-hop chain: node at hop 3 gets nothing (max_depth=2)."""
        conn = _make_conn()
        _insert_memory_row(conn, "a", importance=0.5)
        _insert_memory_row(conn, "b", importance=0.5)
        _insert_memory_row(conn, "c", importance=0.5)
        _insert_memory_row(conn, "d", importance=0.5)

        _insert_edge(conn, "a", "b", "evidence_for", 0.9)
        _insert_edge(conn, "b", "c", "evidence_for", 0.9)
        _insert_edge(conn, "c", "d", "evidence_for", 0.9)

        affected = propagate_impact(conn, "a", 0.20, max_depth=2)
        affected_ids = {a[0] for a in affected}
        assert "b" in affected_ids  # depth 1
        assert "c" in affected_ids  # depth 2
        assert "d" not in affected_ids  # depth 3 — excluded

    def test_propagate_impact_max_affected(self) -> None:
        """Cap at max_affected nodes."""
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)

        # Create 60 neighbors, all connected to root
        for i in range(60):
            nid = f"n{i}"
            _insert_memory_row(conn, nid, importance=0.5)
            _insert_edge(conn, "root", nid, "related_to", 0.7)

        affected = propagate_impact(conn, "root", 0.10, max_affected=50)
        assert len(affected) <= 50

    def test_propagate_impact_negative_delta(self) -> None:
        """Negative delta propagates at 0.5x of positive rate."""
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        affected = propagate_impact(conn, "root", -0.20)
        neighbor_deltas = [a for a in affected if a[0] == "neighbor"]
        assert len(neighbor_deltas) == 1
        # evidence_for rate = 0.3, negative multiplier = 0.5
        # -0.20 * 0.3 * 0.5 = -0.03
        assert abs(neighbor_deltas[0][1] - (-0.03)) < 1e-6

    def test_propagate_impact_records_outcome_history(self) -> None:
        """Impact propagation records changes in outcome_history."""
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        propagate_impact(conn, "root", 0.20)

        row = conn.execute(
            "SELECT outcome_history FROM memories WHERE id = ?",
            ("neighbor",),
        ).fetchone()
        assert row is not None
        history = json.loads(row[0])
        assert len(history) >= 1
        assert "impact_propagation" in history[-1]

    def test_propagate_impact_updates_importance(self) -> None:
        """Impact propagation actually updates the importance field in DB."""
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.5)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        propagate_impact(conn, "root", 0.20)

        row = conn.execute(
            "SELECT importance FROM memories WHERE id = ?",
            ("neighbor",),
        ).fetchone()
        assert row is not None
        # 0.5 + 0.06 = 0.56
        assert abs(row[0] - 0.56) < 1e-6

    def test_propagate_impact_clamps_to_bounds(self) -> None:
        """Importance stays within [0.0, 1.0] after propagation."""
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "neighbor", importance=0.98)
        _insert_edge(conn, "root", "neighbor", "evidence_for", 0.9)

        propagate_impact(conn, "root", 0.50)

        row = conn.execute(
            "SELECT importance FROM memories WHERE id = ?",
            ("neighbor",),
        ).fetchone()
        assert row is not None
        assert row[0] <= 1.0
