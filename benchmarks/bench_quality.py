"""Quality benchmarks -- Precision@K, Recall@K, MRR, NDCG@K.

Uses the golden set fixture for ground-truth relevance judgments.
Each golden entry defines queries with boolean relevance labels,
enabling measurement of retrieval quality metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from benchmarks._retrieval import search_backend_entries
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# IR quality metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@K: fraction of top-K results that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved entry IDs.
        relevant_ids: Set of IDs that are relevant to the query.
        k: Number of top results to consider.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for eid in top_k if eid in relevant_ids)
    return relevant_count / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K: fraction of relevant docs found in top-K.

    Args:
        retrieved_ids: Ordered list of retrieved entry IDs.
        relevant_ids: Set of IDs that are relevant to the query.
        k: Number of top results to consider.

    Returns:
        Recall score between 0.0 and 1.0. Returns 0.0 if no relevant docs exist.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for eid in top_k if eid in relevant_ids)
    return found / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank: 1/rank of the first relevant result.

    Args:
        retrieved_ids: Ordered list of retrieved entry IDs.
        relevant_ids: Set of IDs that are relevant to the query.

    Returns:
        1/rank or 0.0 if no relevant result found.
    """
    for i, eid in enumerate(retrieved_ids):
        if eid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(
    queries_results: list[tuple[list[str], set[str]]],
) -> float:
    """Mean Reciprocal Rank across multiple queries.

    Args:
        queries_results: List of (retrieved_ids, relevant_ids) tuples.

    Returns:
        Average reciprocal rank. Returns 0.0 if no queries provided.
    """
    if not queries_results:
        return 0.0
    total = sum(reciprocal_rank(ret, rel) for ret, rel in queries_results)
    return total / len(queries_results)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain.

    Uses binary relevance: 1 if in relevant_ids, 0 otherwise.

    Args:
        retrieved_ids: Ordered list of retrieved entry IDs.
        relevant_ids: Set of IDs that are relevant to the query.
        k: Number of top results to consider.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    # DCG: sum of rel_i / log2(i+2) for i in [0, k)
    dcg = 0.0
    for i, eid in enumerate(top_k):
        if eid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG: all relevant docs ranked first
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Quality Benchmark runner
# ---------------------------------------------------------------------------


class QualityBenchmark:
    """Run quality benchmarks using the golden set fixture.

    Loads the golden set, stores entries in a SQLite backend, then
    runs each golden query and measures precision, recall, MRR, and NDCG.

    Attributes:
        golden_set_path: Path to the golden set JSON fixture.
        db_dir: Directory for temporary benchmark databases.
    """

    def __init__(self, golden_set_path: Path, db_dir: Path) -> None:
        self.golden_set_path = golden_set_path
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, float]:
        """Run all quality benchmarks against the golden set.

        Returns:
            Dict with precision_at_5, recall_at_10, mrr, ndcg_at_10,
            and per-query averages.
        """
        golden_data = json.loads(self.golden_set_path.read_text(encoding="utf-8"))
        golden_entries = golden_data["entries"]

        # Store golden entries in a fresh database
        db_path = self.db_dir / "quality_bench.db"
        backend = SQLiteBackend(db_path=db_path, dim=384)

        try:
            from datetime import datetime, timezone

            for ge in golden_entries:
                now = datetime.now(timezone.utc)
                entry = MemoryEntry(
                    id=str(ge["id"]),
                    content=str(ge["content"]),
                    tags=[str(t) for t in ge["tags"]],
                    importance=float(ge["importance"]),
                    namespace="project:golden",
                    created_at=now,
                    updated_at=now,
                    source="golden-set",
                )
                backend.store(entry)

            # Build query -> relevant IDs mapping from golden set
            all_precisions: list[float] = []
            all_recalls: list[float] = []
            all_ndcgs: list[float] = []
            mrr_data: list[tuple[list[str], set[str]]] = []

            for ge in golden_entries:
                entry_id = str(ge["id"])
                for q_info in ge["queries"]:
                    query_str = str(q_info["query"])
                    is_relevant = bool(q_info["relevant"])

                    if not is_relevant:
                        # Negative queries -- skip for positive metrics
                        continue

                    results = search_backend_entries(
                        backend,
                        query_str,
                        namespace="project:golden",
                        candidate_limit=len(golden_entries),
                        top_k=10,
                    )
                    retrieved_ids = [r.id for r in results]
                    relevant_set = {entry_id}

                    all_precisions.append(precision_at_k(retrieved_ids, relevant_set, 5))
                    all_recalls.append(recall_at_k(retrieved_ids, relevant_set, 10))
                    all_ndcgs.append(ndcg_at_k(retrieved_ids, relevant_set, 10))
                    mrr_data.append((retrieved_ids, relevant_set))

            avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0.0
            avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
            avg_ndcg = sum(all_ndcgs) / len(all_ndcgs) if all_ndcgs else 0.0
            mrr_val = mean_reciprocal_rank(mrr_data)

            return {
                "precision_at_5": round(avg_precision, 4),
                "recall_at_10": round(avg_recall, 4),
                "mrr": round(mrr_val, 4),
                "ndcg_at_10": round(avg_ndcg, 4),
                "total_queries": float(len(all_precisions)),
            }
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

QUALITY_THRESHOLDS: dict[str, float] = {
    # The bundled golden set defines one canonical relevant entry per positive
    # query, so Precision@5 tops out well below 1.0 even when ranking is correct.
    "precision_at_5": 0.60,
    "recall_at_10": 0.70,
    "mrr": 0.60,
    "ndcg_at_10": 0.65,
}
