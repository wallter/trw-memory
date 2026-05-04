"""Cluster detection + impact propagation for the graph layer.

Belongs to the ``graph.py`` facade. Re-exported there for back-compat.

3 helpers covering dense-subgraph + propagation operations:

- ``detect_clusters`` — greedy clique-expansion bounded by graph
  connectivity.
- ``propagate_impact`` — BFS importance propagation with edge-type
  rates and 0.5x dampening for negative deltas.
- ``_propose_domain_name`` — pick most-common tag among cluster
  members (private helper).

Plus 2 module constants: `_PROPAGATION_RATES` (per edge-type rate)
and `_NEGATIVE_MULTIPLIER`.

Extracted as PRD-DIST-245 Phase 2 batch 96.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)

_PROPAGATION_RATES: dict[str, float] = {
    "evidence_for": 0.3,
    "co_anchored": 0.2,
    "same_root_cause": 0.15,
    "related_to": 0.1,
    "depends_on": 0.1,
}
_NEGATIVE_MULTIPLIER = 0.5


def detect_clusters(
    conn: sqlite3.Connection,
    min_size: int = 5,
    min_connectivity: float = 0.6,
) -> list[dict[str, object]]:
    """Detect dense subgraphs via greedy clique-expansion bounded by graph size."""
    rows = conn.execute("SELECT DISTINCT source_id, target_id FROM memory_graph_edges").fetchall()

    adj: dict[str, set[str]] = {}
    for src, tgt in rows:
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)

    if len(adj) < min_size:
        return []

    nodes_by_degree = sorted(adj.keys(), key=lambda n: len(adj[n]), reverse=True)
    used: set[str] = set()
    clusters: list[dict[str, object]] = []

    for seed in nodes_by_degree:
        if seed in used:
            continue

        cluster = {seed}
        candidates = sorted(adj[seed] - used, key=lambda n: len(adj.get(n, set())), reverse=True)

        for cand in candidates:
            trial = cluster | {cand}
            n = len(trial)
            max_edges = n * (n - 1) / 2
            if max_edges == 0:
                continue
            actual = sum(
                1 for node in trial for neighbor in adj.get(node, set()) if neighbor in trial and neighbor > node
            )
            connectivity = actual / max_edges
            if connectivity >= min_connectivity:
                cluster.add(cand)

        if len(cluster) >= min_size:
            n = len(cluster)
            max_edges = n * (n - 1) / 2
            actual = sum(
                1 for node in cluster for neighbor in adj.get(node, set()) if neighbor in cluster and neighbor > node
            )
            connectivity = actual / max_edges if max_edges > 0 else 0.0
            domain_name = _propose_domain_name(conn, list(cluster))
            clusters.append(
                {
                    "domain_name": domain_name,
                    "member_ids": sorted(cluster),
                    "connectivity": round(connectivity, 4),
                }
            )
            used.update(cluster)

    logger.debug("clusters_detected", count=len(clusters))
    return clusters


def propagate_impact(
    conn: sqlite3.Connection,
    entry_id: str,
    importance_delta: float,
    max_depth: int = 2,
    max_affected: int = 50,
) -> list[tuple[str, float]]:
    """BFS importance propagation from *entry_id*. Negative deltas at 0.5x rate."""
    affected: list[tuple[str, float]] = []
    visited: set[str] = {entry_id}
    queue: deque[tuple[str, int, float]] = deque()
    queue.append((entry_id, 0, importance_delta))
    now = datetime.now(timezone.utc).isoformat()

    while queue and len(affected) < max_affected:
        node_id, depth, current_delta = queue.popleft()
        if depth >= max_depth:
            continue

        placeholders = ", ".join("?" for _ in _PROPAGATION_RATES)
        rows = conn.execute(
            f"SELECT target_id, edge_type FROM memory_graph_edges "  # noqa: S608
            f"WHERE source_id = ? AND edge_type IN ({placeholders})",
            (node_id, *_PROPAGATION_RATES.keys()),
        ).fetchall()

        for target_id, edge_type in rows:
            if target_id in visited or len(affected) >= max_affected:
                continue

            rate = _PROPAGATION_RATES.get(edge_type, 0.0)
            if rate == 0.0:
                continue

            propagated = current_delta * rate
            if importance_delta < 0:
                propagated *= _NEGATIVE_MULTIPLIER

            conn.execute(
                "UPDATE memories SET "
                "importance = MIN(MAX(importance + ?, 0.0), 1.0), "
                "outcome_history = json_insert(outcome_history, '$[#]', ?) "
                "WHERE id = ?",
                (
                    propagated,
                    f"impact_propagation:delta={propagated:+.4f}:from={entry_id}:via={edge_type}:timestamp={now}",
                    target_id,
                ),
            )

            visited.add(target_id)
            affected.append((target_id, round(propagated, 6)))
            queue.append((target_id, depth + 1, propagated))

    if affected:
        conn.commit()
    logger.debug(
        "impact_propagated",
        entry_id=entry_id,
        delta=importance_delta,
        affected_count=len(affected),
    )
    return affected


def _propose_domain_name(conn: sqlite3.Connection, member_ids: list[str]) -> str:
    """Propose a domain name for a cluster from member tags or IDs.

    Uses the most common tag among cluster members. Falls back to a
    generic label based on cluster size.
    """
    if not member_ids:
        return "unknown"
    placeholders = ", ".join("?" for _ in member_ids)
    rows = conn.execute(
        f"SELECT tags FROM memories WHERE id IN ({placeholders})",  # noqa: S608
        tuple(member_ids),
    ).fetchall()

    tag_counts: dict[str, int] = {}
    for (tags_json,) in rows:
        try:
            tags = json.loads(tags_json) if tags_json else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        for tag in tags:
            if isinstance(tag, str) and tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if tag_counts:
        return max(tag_counts, key=lambda t: tag_counts[t])
    return f"cluster-{len(member_ids)}"
