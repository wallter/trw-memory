"""Recall-time warm discovery: which warm rows tier discovery sees, and their vector relevance.

Split from ``_warm.py`` (PRD-CORE-318 FR02b). With a row bound M the vectored rows come
from a sqlite-vec KNN on the warm tier's own index (``_knn_window``), not a decode and a
Python L2 of every stored vector; ``WarmTierStore.discovery_entries`` states the contract.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory._live_stores import connect_registered
from trw_memory.exceptions import StorageError
from trw_memory.lifecycle.tiers._manager_search import entry_matches_tokens
from trw_memory.lifecycle.tiers._warm_space import WARM_TIER_NAMESPACE, admit_warm_vectors

if TYPE_CHECKING:
    from trw_memory.embeddings.provenance import EmbeddingSpace
    from trw_memory.lifecycle.tiers._manager_search import WindowRank
    from trw_memory.lifecycle.tiers._warm_sidecar_cache import ParsedSidecar, SidecarRows

logger = structlog.get_logger(__name__)

#: sqlite-vec refuses a KNN ``k`` above this.
_KNN_MAX_K = 4096

#: The ``limit`` nearest vectors among those that claim the query's space and that the caller's
#: pool did not already rank; both filters run INSIDE the vec0 scan (``rowid IN``), so neither
#: takes a window slot (PRD-CORE-318 FR02b).
#: Score slack for the stop rule: the KNN ranks on sqlite-vec's float32 distance, recall scores the
#: Python float64 one, so a bound compared at the float32 edge gives way by far more than their gap.
_STOP_SLACK = 1e-6

_KNN_SQL = """SELECT vi.entry_id, vm.distance FROM vec_memories vm JOIN vec_index vi ON vi.rowid = vm.rowid
WHERE vm.embedding MATCH ?1 AND k = ?2 AND vm.rowid IN (SELECT rowid FROM vec_index
    WHERE namespace = ?3 AND space_key = ?4 AND entry_id NOT IN (SELECT value FROM json_each(?5)))
ORDER BY vm.distance"""


class _StopRule:
    """Threshold-algorithm stop rule for the KNN window (PRD-CORE-318 FR02b, review r1 P1).

    A window may stop when its M-th best weighted rank score beats the most any row past it
    could score: that row is no nearer than the window's farthest (so its relevance is at most
    the edge relevance), and every other term is at its maximum over the whole warm tier, cached
    per sidecar version (``ParsedSidecar.maxima``, ``WindowRank.ceiling``; review r2). Rows
    outside recall's best rank class never count toward M.
    """

    def __init__(self, parsed: ParsedSidecar, rank: WindowRank) -> None:
        self._parsed, self._rank = parsed, rank
        self._ceiling: Callable[[float], float] | None = None
        self._scores: dict[str, float | None] = {}  # a row is scored once however far the window widens

    def holds(self, hits: list[tuple[str, float]], limit: int) -> bool:
        if self._ceiling is None:  # the version's cached maxima: O(1) once computed
            self._ceiling = self._rank.ceiling(self._parsed.maxima())
        bar = self._ceiling(_relevance(hits[-1][1])) + _STOP_SLACK
        strong = 0
        for entry_id, distance in hits:
            if entry_id not in self._scores:
                rec = self._parsed.live(entry_id)
                relevance = _relevance(distance)
                self._scores[entry_id] = None if rec is None else self._rank.score(entry_item(entry_id, rec), relevance)
            score = self._scores[entry_id]
            strong += score is not None and score > bar
        return strong >= limit


def _relevance(distance: float) -> float:
    return 1.0 - distance * distance / 2.0


def _knn_window(
    conn: sqlite3.Connection,
    query: list[float],
    space: EmbeddingSpace,
    covered_ids: frozenset[str],
    limit: int,
    stop: _StopRule,
) -> tuple[frozenset[str], list[str] | None]:
    """Every id with a vector claiming *space* (index-only, no blob), and the nearest uncovered ones.

    ONE KNN scan at sqlite-vec's ``k`` cap (vec0 is exhaustive, so a scan costs about the same
    at any ``k``; review r2 measured 27-45 ms per scan at 20k). The window is a prefix of it:
    twice *limit*, widened fourfold until *stop* proves no row past it can reach recall's top
    *limit*, or every candidate. ``None`` when no prefix under the cap is provably complete: the
    caller scans all.
    """
    in_space = frozenset(
        row[0]
        for row in conn.execute(
            "SELECT entry_id FROM vec_index WHERE namespace = ? AND space_key = ?", (WARM_TIER_NAMESPACE, space.key)
        )
    )
    packed = struct.pack(f"{len(query)}f", *query)
    covered = json.dumps(sorted(covered_ids & in_space))
    params = (packed, _KNN_MAX_K, WARM_TIER_NAMESPACE, space.key, covered)
    nearest = [(str(row[0]), float(row[1])) for row in conn.execute(_KNN_SQL, params)]
    # At k = M the M-th row IS the edge row and can never beat it strictly; start one step on.
    k = 2 * limit
    while True:
        if k >= len(nearest):
            complete = len(nearest) < _KNN_MAX_K or stop.holds(nearest, limit)
            return in_space, [entry_id for entry_id, _distance in nearest] if complete else None
        if stop.holds(nearest[:k], limit):
            return in_space, [entry_id for entry_id, _distance in nearest[:k]]
        k *= 4


def entry_item(entry_id: str, rec: dict[str, object]) -> dict[str, object]:
    """One sidecar row's entry payload, as a fresh copy."""
    payload = rec.get("entry")
    if isinstance(payload, dict):
        item = dict(payload)
    else:
        raw_tags = rec.get("tags", [])
        tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
        item = {"id": entry_id, "content": str(rec.get("summary", "")), "tags": tags}
    item.setdefault("id", entry_id)
    return item


def discovery_rows(
    parsed: ParsedSidecar,
    db_path: Path,
    query_embedding: list[float] | None,
    *,
    query_tokens: Sequence[str],
    limit: int | None,
    covered_ids: frozenset[str],
    query_space: EmbeddingSpace | None,
    rank: WindowRank | None,
) -> list[dict[str, object]]:
    """The body of ``WarmTierStore.discovery_entries`` (its docstring is the contract)."""
    rows = parsed.rows_except(covered_ids)
    hints: dict[str, float] = {}
    outside: frozenset[str] = frozenset()
    if query_embedding is not None and rows and db_path.exists():
        # Without a rank there is no sound stop rule, so the window is the full scan.
        bounded = limit if rank is not None else None
        stop = _StopRule(parsed, rank) if rank is not None else None
        hints, outside = _nearest(db_path, rows, query_embedding, query_space, covered_ids, bounded, stop)
    tokens = list(query_tokens)
    found: list[tuple[bool, dict[str, object]]] = []
    for _line, rec in rows:
        entry_id = str(rec.get("id", ""))
        if not entry_id or entry_id in outside:
            continue
        item = entry_item(entry_id, rec)
        if entry_id in hints:
            item["_tier_relevance"] = hints[entry_id]
        found.append((entry_id in hints or entry_matches_tokens(item, tokens), item))
    # FR02 orders hinted and token-matched rows first; with M of them no other row can be kept.
    fill = limit is None or sum(hit for hit, _item in found) < limit
    return [item for hit, item in found if hit or fill]


def _nearest(
    db_path: Path,
    rows: SidecarRows,
    query_embedding: list[float],
    query_space: EmbeddingSpace | None,
    covered_ids: frozenset[str],
    limit: int | None,
    stop: _StopRule | None,
) -> tuple[dict[str, float], frozenset[str]]:
    """``_tier_relevance`` per scored id, and the in-space ids the KNN window left out.

    Any failure degrades to "vectors unavailable" (no hints): this is a
    ranking enhancement, not a data path.
    """
    window: list[str] | None = None
    in_space: frozenset[str] = frozenset()
    try:
        import sqlite_vec

        # connect_registered refuses (StorageError) a store swapped during the open (PRD-SEC-016).
        with closing(connect_registered(db_path, sqlite3, f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            # Warm vectors have one fixed namespace; reject corrupted foreign rows
            # before the existing ID-based bulk decoder sees them.
            foreign = conn.execute(
                "SELECT 1 FROM vec_index WHERE namespace != ? LIMIT 1", (WARM_TIER_NAMESPACE,)
            ).fetchone()
            if foreign:
                from trw_memory.security.namespace_scope import NamespaceScopeError

                raise NamespaceScopeError("warm vector index contains foreign namespace")
            if limit is not None and query_space is not None and stop is not None:
                try:
                    in_space, window = _knn_window(conn, query_embedding, query_space, covered_ids, limit, stop)
                except sqlite3.OperationalError:  # trw-fail-silent-allow: a vec0 without KNN filtering or a dimension mismatch falls back to the FR02 full scan below, logged
                    logger.warning("warm_tier_knn_unavailable_full_scan", exc_info=True)
            if window is None:
                in_space, window = frozenset(), [str(rec.get("id", "")) for _line, rec in rows]
            vectors = admit_warm_vectors(conn, window, query_space)
    except StorageError:
        logger.warning("warm_tier_db_identity_changed_during_open", path=str(db_path))
        return {}, frozenset()
    except (ImportError, sqlite3.Error, OSError, AttributeError):
        logger.debug("warm_tier_discovery_vectors_unavailable", exc_info=True)
        return {}, frozenset()
    hints = {
        entry_id: 1.0 - sum((a - b) ** 2 for a, b in zip(vector, query_embedding, strict=True)) / 2.0
        for entry_id, vector in vectors.items()
        if len(vector) == len(query_embedding)
    }
    return hints, in_space.difference(window)
