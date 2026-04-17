"""Integration tests for bandit and graph primitives working together.

Exercises BanditSelector lifecycle, PageHinkleyDetector change detection,
and KnowledgeGraph operations (co-anchored edges, cluster detection,
impact propagation) with real SQLite state. No mocks.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timezone

import pytest

from trw_memory.bandit.change_detection import PageHinkleyDetector
from trw_memory.bandit.thompson import BanditSelector
from trw_memory.graph import (
    create_co_anchored_edges,
    detect_clusters,
    propagate_impact,
)
from trw_memory.storage._schema import ensure_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the full schema."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def _insert_memory_row(
    conn: sqlite3.Connection,
    entry_id: str,
    *,
    content: str = "content",
    importance: float = 0.5,
    anchors_json: str = "[]",
    tags_json: str = "[]",
    outcome_history_json: str = "[]",
) -> None:
    """Insert a minimal memory row for graph tests."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memories "
        "(id, content, created_at, updated_at, importance, anchors, tags, outcome_history) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (entry_id, content, now, now, importance, anchors_json, tags_json, outcome_history_json),
    )
    conn.commit()


def _insert_edge(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
) -> None:
    """Insert a graph edge directly."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memory_graph_edges "
        "(source_id, target_id, edge_type, weight, created_at, edge_metadata) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source_id, target_id, edge_type, weight, now, "{}"),
    )
    conn.commit()


def _get_importance(conn: sqlite3.Connection, entry_id: str) -> float:
    """Read the importance field for an entry."""
    row = conn.execute("SELECT importance FROM memories WHERE id = ?", (entry_id,)).fetchone()
    assert row is not None, f"Entry {entry_id} not found"
    return float(row[0])


# ===========================================================================
# Thompson Sampling Full Lifecycle
# ===========================================================================


@pytest.mark.unit
class TestThompsonFullLifecycle:
    """BanditSelector add_arm -> select/update cycles -> serialize -> restore."""

    def test_full_lifecycle(self) -> None:
        """20 cycles of select/update, serialize to JSON, restore, verify state."""
        rng = random.Random(12345)
        selector = BanditSelector(tau=25, cold_start_min=2, floor_exploration=0.10)
        arm_ids = ["arm-a", "arm-b", "arm-c", "arm-d", "arm-e"]

        # Run 20 cycles of select + update
        for _ in range(20):
            decision = selector.select(arm_ids)
            reward = rng.random()
            selector.update(decision.selected_id, reward)

        # Serialize
        json_str = selector.to_json()
        parsed = json.loads(json_str)
        assert "arms" in parsed
        assert len(parsed["arms"]) == 5

        # Restore
        restored = BanditSelector.from_json(json_str)

        # Verify arm count and all observations preserved
        assert len(restored._arms) == 5
        for arm_id in arm_ids:
            assert arm_id in restored._arms
            original = selector._arms[arm_id]
            clone = restored._arms[arm_id]
            assert clone.alpha == pytest.approx(original.alpha)
            assert clone.beta == pytest.approx(original.beta)
            assert clone.window == pytest.approx(original.window)
            assert clone.exposure_count == original.exposure_count

        # Verify hyperparameters preserved
        assert restored._tau == selector._tau
        assert restored._cold_start_min == selector._cold_start_min
        assert restored._floor_exploration == pytest.approx(selector._floor_exploration)

    def test_lifecycle_continues_after_restore(self) -> None:
        """Restored selector continues to select and update normally."""
        selector = BanditSelector(tau=10, cold_start_min=1)
        arm_ids = ["x", "y", "z"]

        for _ in range(10):
            d = selector.select(arm_ids)
            selector.update(d.selected_id, 0.5)

        restored = BanditSelector.from_json(selector.to_json())

        # Continue operating -- should not raise
        for _ in range(10):
            d = restored.select(arm_ids)
            assert d.selected_id in arm_ids
            restored.update(d.selected_id, 0.7)


# ===========================================================================
# PageHinkley Change Detection
# ===========================================================================


@pytest.mark.unit
class TestPageHinkleyDetectsShift:
    """PageHinkleyDetector fires alarm on mean shift."""

    def test_no_alarm_stationary(self) -> None:
        """50 observations from N(0,1) with no shift -- no alarm expected."""
        rng = random.Random(42)
        detector = PageHinkleyDetector(delta=0.01, alarm_threshold=20.0)

        alarm_fired = False
        for _ in range(50):
            obs = rng.gauss(0.0, 1.0)
            if detector.update(obs):
                alarm_fired = True
        assert not alarm_fired, "Alarm should not fire on stationary data"

    def test_alarm_after_shift(self) -> None:
        """50 stationary observations then 50 shifted by +3 -- alarm fires."""
        rng = random.Random(42)
        detector = PageHinkleyDetector(delta=0.01, alarm_threshold=20.0)

        # Stationary phase
        for _ in range(50):
            detector.update(rng.gauss(0.0, 1.0))

        # Shifted phase -- alarm should fire at some point
        alarm_fired = False
        for _ in range(50):
            obs = rng.gauss(3.0, 1.0)
            if detector.update(obs):
                alarm_fired = True
                break
        assert alarm_fired, "Alarm should fire after mean shift from 0 to 3"

    def test_serialization_roundtrip(self) -> None:
        """to_dict / from_dict preserves internal state."""
        rng = random.Random(99)
        detector = PageHinkleyDetector(delta=0.02, alarm_threshold=15.0)

        for _ in range(30):
            detector.update(rng.gauss(1.0, 0.5))

        state = detector.to_dict()
        restored = PageHinkleyDetector.from_dict(state)

        assert restored._delta == pytest.approx(detector._delta)
        assert restored._alarm_threshold == pytest.approx(detector._alarm_threshold)
        assert restored._n == detector._n
        assert restored._sum == pytest.approx(detector._sum)
        assert restored._h == pytest.approx(detector._h)


# ===========================================================================
# Co-Anchored Edges from Stored Entries
# ===========================================================================


@pytest.mark.unit
class TestCoAnchoredEdgesFromStoredEntries:
    """create_co_anchored_edges discovers entries sharing anchor files in SQLite."""

    def test_shared_anchor_creates_edge(self) -> None:
        """Two entries sharing anchor file='src/auth.py' get a co_anchored edge."""
        conn = _make_conn()

        anchor_a = [{"file": "src/auth.py", "symbol_name": "login", "symbol_type": "function"}]
        anchor_b = [{"file": "src/auth.py", "symbol_name": "logout", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_a))
        _insert_memory_row(conn, "e2", anchors_json=json.dumps(anchor_b))

        count = create_co_anchored_edges(conn, "e1", ["src/auth.py"])
        assert count >= 1

        # Verify edge exists
        row = conn.execute(
            "SELECT source_id, target_id, edge_type, edge_metadata "
            "FROM memory_graph_edges "
            "WHERE edge_type = 'co_anchored'",
        ).fetchone()
        assert row is not None
        assert row[0] == "e1"
        assert row[1] == "e2"
        meta = json.loads(row[3])
        assert meta["anchor_file"] == "src/auth.py"

    def test_no_shared_anchor_no_edge(self) -> None:
        """Entries with different anchor files produce no co_anchored edge."""
        conn = _make_conn()

        anchor_a = [{"file": "src/auth.py", "symbol_name": "login", "symbol_type": "function"}]
        anchor_b = [{"file": "src/billing.py", "symbol_name": "charge", "symbol_type": "function"}]
        _insert_memory_row(conn, "e1", anchors_json=json.dumps(anchor_a))
        _insert_memory_row(conn, "e2", anchors_json=json.dumps(anchor_b))

        count = create_co_anchored_edges(conn, "e1", ["src/auth.py"])
        assert count == 0


# ===========================================================================
# Cluster Detection from Stored Entries
# ===========================================================================


@pytest.mark.unit
class TestClusterDetectionFromStoredEntries:
    """detect_clusters finds dense subgraphs from pairwise similarity edges."""

    def test_dense_clique_detected(self) -> None:
        """6 entries with all pairwise edges form a cluster with min_size=5."""
        conn = _make_conn()
        node_ids = [f"c{i}" for i in range(6)]
        for nid in node_ids:
            _insert_memory_row(conn, nid, tags_json=json.dumps(["testing"]))

        # Create bidirectional similarity edges for all 15 pairs
        now = datetime.now(timezone.utc).isoformat()
        for i in range(6):
            for j in range(i + 1, 6):
                conn.execute(
                    "INSERT INTO memory_graph_edges "
                    "(source_id, target_id, edge_type, weight, created_at, edge_metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (node_ids[i], node_ids[j], "similarity", 0.9, now, "{}"),
                )
                conn.execute(
                    "INSERT INTO memory_graph_edges "
                    "(source_id, target_id, edge_type, weight, created_at, edge_metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (node_ids[j], node_ids[i], "similarity", 0.9, now, "{}"),
                )
        conn.commit()

        clusters = detect_clusters(conn, min_size=5, min_connectivity=0.6)
        assert len(clusters) >= 1

        # Verify the cluster contains all 6 members
        cluster = clusters[0]
        member_ids = cluster["member_ids"]
        assert isinstance(member_ids, list)
        for nid in node_ids:
            assert nid in member_ids, f"Expected {nid} in cluster members"

        # Verify connectivity is high (should be 1.0 for a complete graph)
        assert cluster["connectivity"] >= 0.9

    def test_sparse_graph_no_cluster(self) -> None:
        """3 disconnected entries do not form a cluster with min_size=5."""
        conn = _make_conn()
        for i in range(3):
            _insert_memory_row(conn, f"sparse{i}")

        clusters = detect_clusters(conn, min_size=5, min_connectivity=0.6)
        assert clusters == []


# ===========================================================================
# Impact Propagation with evidence_for Chains
# ===========================================================================


@pytest.mark.unit
class TestImpactPropagationWithEvidenceFor:
    """propagate_impact spreads delta through evidence_for edge chains."""

    def test_two_hop_propagation(self) -> None:
        """A -> B -> C via evidence_for: B gets more impact than C."""
        conn = _make_conn()
        _insert_memory_row(conn, "A", importance=0.5)
        _insert_memory_row(conn, "B", importance=0.5)
        _insert_memory_row(conn, "C", importance=0.5)

        _insert_edge(conn, "A", "B", "evidence_for", 0.9)
        _insert_edge(conn, "B", "C", "evidence_for", 0.9)

        affected = propagate_impact(conn, "A", importance_delta=0.1, max_depth=2)
        affected_map = {entry_id: delta for entry_id, delta in affected}

        # B receives: 0.1 * 0.3 (evidence_for rate) = 0.03
        assert "B" in affected_map
        assert affected_map["B"] == pytest.approx(0.03, abs=1e-6)

        # C receives: 0.03 * 0.3 = 0.009
        assert "C" in affected_map
        assert affected_map["C"] == pytest.approx(0.009, abs=1e-6)

        # B's delta is larger than C's
        assert abs(affected_map["B"]) > abs(affected_map["C"])

    def test_importance_actually_updated_in_db(self) -> None:
        """propagate_impact writes the updated importance back to SQLite."""
        conn = _make_conn()
        _insert_memory_row(conn, "src", importance=0.5)
        _insert_memory_row(conn, "tgt", importance=0.5)
        _insert_edge(conn, "src", "tgt", "evidence_for", 0.9)

        propagate_impact(conn, "src", importance_delta=0.2, max_depth=2)

        # tgt should have 0.5 + (0.2 * 0.3) = 0.56
        new_importance = _get_importance(conn, "tgt")
        assert new_importance == pytest.approx(0.56, abs=1e-6)

    def test_outcome_history_recorded(self) -> None:
        """propagate_impact records the change in outcome_history."""
        conn = _make_conn()
        _insert_memory_row(conn, "root", importance=0.5)
        _insert_memory_row(conn, "leaf", importance=0.5)
        _insert_edge(conn, "root", "leaf", "evidence_for", 0.9)

        propagate_impact(conn, "root", importance_delta=0.1, max_depth=1)

        row = conn.execute(
            "SELECT outcome_history FROM memories WHERE id = ?",
            ("leaf",),
        ).fetchone()
        assert row is not None
        history = json.loads(row[0])
        assert len(history) >= 1
        assert "impact_propagation" in history[-1]
        assert "from=root" in history[-1]

    def test_depth_limit_respected(self) -> None:
        """Node at depth 3 is not affected when max_depth=2."""
        conn = _make_conn()
        _insert_memory_row(conn, "A", importance=0.5)
        _insert_memory_row(conn, "B", importance=0.5)
        _insert_memory_row(conn, "C", importance=0.5)
        _insert_memory_row(conn, "D", importance=0.5)

        _insert_edge(conn, "A", "B", "evidence_for", 0.9)
        _insert_edge(conn, "B", "C", "evidence_for", 0.9)
        _insert_edge(conn, "C", "D", "evidence_for", 0.9)

        affected = propagate_impact(conn, "A", importance_delta=0.1, max_depth=2)
        affected_ids = {entry_id for entry_id, _ in affected}

        assert "B" in affected_ids
        assert "C" in affected_ids
        assert "D" not in affected_ids

        # D's importance should be unchanged
        assert _get_importance(conn, "D") == pytest.approx(0.5)
