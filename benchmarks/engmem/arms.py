"""Retrieval arms under test, all returning a ranked list of learning ids.

The arms exist to answer "compared to what?". A memory engine that cannot beat
recency or grep on engineering recall is not earning its complexity, and the
design requires both baselines in every published table.

Two trw-memory arms, and the difference between them is the point:

* ``trw-hybrid`` -- ``MemoryClient.recall``: BM25 + dense + RRF, then the
  cross-encoder reranker, the adaptive floor and the entity bridge hop.
* ``trw-framework`` -- what ``trw_recall`` and ``trw_session_start`` actually
  execute. Since PRD-DIST-254 the MCP path calls the same ``hybrid_search``
  core, but with a much narrower argument set: ``rerank`` and ``bridge_hop``
  both default to ``False`` (``retrieval/pipeline.py``) and nothing in
  ``trw-mcp/src`` ever passes them, so the reranker, the adaptive floor, the
  bridge hop, recency blending and validity-age tie-breaking are all off.

Every retrieval number measured against LOCOMO describes the first arm. The
framework ships the second. Nobody has measured what that costs, and this is
where that gets measured -- on engineering memory rather than on chat.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

# Roughly 4 characters per token; the absolute value matters less than comparing
# arms at an equal budget, which is the only honest way to compare context.
CHARS_PER_TOKEN = 4


class Arm(Protocol):
    name: str

    async def search(self, query: str, limit: int, as_of: datetime) -> list[tuple[str, int]]:
        """Ranked ``(learning_id, chars)`` best first."""
        ...


@dataclass
class RecencyArm:
    """The null hypothesis: hand back the most recent rows, ignore the query.
    Anything that cannot beat this is a sorted list with extra steps."""

    rows: list[tuple[str, datetime, str]]
    name: str = "recency"

    async def search(self, query: str, limit: int, as_of: datetime) -> list[tuple[str, int]]:
        live = [(rid, at, text) for rid, at, text in self.rows if at <= as_of]
        live.sort(key=lambda r: r[1], reverse=True)
        return [(rid, len(text)) for rid, _at, text in live[:limit]]


@dataclass
class GrepArm:
    """What an engineer does without a memory system: substring-match the corpus.
    Terms are matched case-insensitively and rows are ordered by how many distinct
    query terms they contain, ties by recency."""

    rows: list[tuple[str, datetime, str]]
    name: str = "grep"
    _word = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

    async def search(self, query: str, limit: int, as_of: datetime) -> list[tuple[str, int]]:
        terms = {t.lower() for t in self._word.findall(query)}
        if not terms:
            return []
        hits = []
        for rid, at, text in self.rows:
            if at > as_of:
                continue
            low = text.lower()
            n = sum(1 for t in terms if t in low)
            if n:
                hits.append((n, at, rid, len(text)))
        hits.sort(key=lambda h: (-h[0], -h[1].timestamp()))
        return [(rid, chars) for _n, _at, rid, chars in hits[:limit]]


class _TrwArmBase:
    """Shared plumbing for the two trw-memory arms.

    ``as_of`` is enforced by the replay -- the store only ever contains events up
    to the query's instant -- so these arms do not filter by time themselves. A
    filter here would mask a leak in the replay rather than prevent one.
    """

    def __init__(self, client: Any, namespace: str, name: str) -> None:
        self._client = client
        self._ns = namespace
        self.name = name

    @staticmethod
    def _chars(entry: Any) -> int:
        content = getattr(entry, "content", "") or ""
        detail = getattr(entry, "detail", "") or ""
        return len(content) + len(detail)


class TrwHybridArm(_TrwArmBase):
    """``MemoryClient.recall`` -- the benchmark path, reranker and bridge included."""

    def __init__(self, client: Any, namespace: str) -> None:
        super().__init__(client, namespace, "trw-hybrid")

    async def search(self, query: str, limit: int, as_of: datetime) -> list[tuple[str, int]]:
        rows = await self._client.recall(query, limit=limit, include_org_memories=False)
        out = []
        for r in rows:
            rid = (r.get("metadata") or {}).get("learning_id") or r.get("id") or ""
            out.append((str(rid), len(str(r.get("content") or "")) + len(str(r.get("detail") or ""))))
        return out


class TrwFrameworkArm(_TrwArmBase):
    """What ``trw_recall``/``trw_session_start`` execute since PRD-CORE-292.

    Calls what ``trw-mcp/src/trw_mcp/state/_memory_queries.py`` calls, rather than
    restating it: candidates from ``recall_policy.acquire_candidates`` (the recency
    pool plus the full-text leg, ACTIVE rows only), the query from
    ``resolve_query``, and every ranking argument from ``ranking_arguments``.
    Stored vectors are read for the pool's ids; the MCP path's provenance gate is
    not repeated because every vector in a benchmark store was written by the one
    embedder the arm queries with.

    Before PRD-CORE-298 FR05 this arm restated the pre-CORE-292 path (recency pool
    only, no rerank), so its earlier numbers describe that path, not today's.
    """

    def __init__(self, client: Any, namespace: str, name: str = "trw-framework") -> None:
        super().__init__(client, namespace, name)
        from trw_memory.models.memory import MemoryStatus

        self._status = MemoryStatus.ACTIVE

    def _candidates(self, backend: Any, query: str, limit: int, config: Any) -> list[Any]:
        from trw_memory.retrieval.recall_policy import acquire_candidates

        return acquire_candidates(
            backend, query, namespace=self._ns, limit=limit, config=config, status=self._status
        ).entries

    async def search(self, query: str, limit: int, as_of: datetime) -> list[tuple[str, int]]:
        from trw_memory.embeddings import embed_query
        from trw_memory.models.config import MemoryConfig
        from trw_memory.retrieval.pipeline import hybrid_search
        from trw_memory.retrieval.recall_policy import RECALL_PREFETCH_MULTIPLIER, ranking_arguments, resolve_query
        from trw_memory.security.namespace_scope import authorize_namespaces
        from trw_memory.security.rbac import Permission

        backend = self._client._get_backend()
        cfg = MemoryConfig()
        depth = limit * RECALL_PREFETCH_MULTIPLIER  # trw_recall ranks this deep and caps last
        entries = self._candidates(backend, query, depth, cfg)
        if not entries:
            return []
        retrieval = resolve_query(query, cfg)
        scope = authorize_namespaces(cfg, {self._ns}, Permission.READ, "recall")
        embedder = getattr(self._client, "_get_embedder", lambda: None)()
        query_vec = embed_query(embedder, retrieval.text) if embedder is not None else None
        records = backend.get_vector_records([e.id for e in entries], namespace=self._ns)
        stored = {entry_id: list(record.embedding) for entry_id, record in records.items()}
        ranked = hybrid_search(
            query=retrieval.text,
            entries=entries,
            scope=scope,
            embedder=embedder,
            query_embedding=query_vec,
            stored_embeddings=stored or None,
            **ranking_arguments(cfg, limit=depth, pool_size=len(entries), recency_weight=retrieval.recency_weight),
        )
        return [(str(getattr(e, "id", "")), self._chars(e)) for e in ranked[:limit]]


class TrwQueryPoolArm(TrwFrameworkArm):
    """``TrwFrameworkArm`` with query-driven candidates, and nothing else changed.

    The PRD-CORE-298 FR05 step 3 comparison. Candidates are the per-term
    full-text union ``TrwFtsFirstArm`` uses (recency when full-text finds
    nothing); the query, the vectors and every ranking argument are the
    framework arm's own, so a difference between the two arms is the pool.
    """

    _word = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{2,}")

    def __init__(self, client: Any, namespace: str, per_term: int = 200) -> None:
        super().__init__(client, namespace, "trw-query-pool")
        self._per_term = per_term

    def _candidates(self, backend: Any, query: str, limit: int, config: Any) -> list[Any]:
        seen: dict[str, Any] = {}
        for term in list(dict.fromkeys(self._word.findall(query)))[:12]:
            for e in backend.search_fts(term, top_k=self._per_term, namespace=self._ns, status=self._status):
                seen.setdefault(str(getattr(e, "id", "")), e)
        if seen:
            return list(seen.values())
        pool = max(limit * 5, config.hybrid_search_candidate_pool_size)
        return list(backend.list_entries(namespace=self._ns, status=self._status, limit=pool))


class TrwDaemonToolArm(_TrwArmBase):
    """The daemon's ``memory_recall`` tool as it ships (PRD-CORE-298 FR05).

    The third recall surface: every checkout attached to the daemon recalls
    through it. The arm runs the tool's own body -- ``MemoryConfig()`` from the
    environment, a backend opened per call with ``create_backend_from_config``,
    then ``memory_recall_impl`` with only the arguments a caller sends -- and
    leaves out nothing but the HTTP transport, which ranks nothing.

    The tool resolves its store from the environment, not from ``client``. The
    first search checks both name the same file, so a routing change cannot
    turn this arm into a measurement of an empty or foreign store.
    """

    def __init__(self, client: Any, namespace: str) -> None:
        super().__init__(client, namespace, "trw-daemon-tool")
        self._checked = False

    async def search(self, query: str, limit: int, as_of: datetime) -> list[tuple[str, int]]:
        from trw_memory.integrations._backend import create_backend_from_config, resolve_backend_db_path
        from trw_memory.models.config import MemoryConfig
        from trw_memory.tools.recall import memory_recall_impl

        cfg = MemoryConfig()
        if not self._checked:  # before opening: a wrong path must not be created or migrated
            resolved, expected = resolve_backend_db_path(cfg, self._ns), self._client._get_backend().db_path
            # ASYNC240: one check per run, before any query is timed.
            if os.path.realpath(resolved) != os.path.realpath(expected):  # noqa: ASYNC240
                raise RuntimeError(f"daemon-tool arm would open {resolved}, the replay wrote {expected}")
            self._checked = True
        with create_backend_from_config(cfg, self._ns, check_integrity_once=True) as backend:  # as the tool opens
            result = memory_recall_impl(query, self._ns, backend=backend, limit=limit, config=cfg)
        rows = result.get("memories", [])
        if not isinstance(rows, list):
            raise TypeError(f"memory_recall returned no memories list: {result}")
        return [
            (str(row.get("id", "")), len(str(row.get("content") or "")) + len(str(row.get("detail") or "")))
            for row in rows[:limit]
        ]


class TrwFtsFirstArm(_TrwArmBase):
    """The proposed fix, as an arm: QUERY-DRIVEN candidate generation.

    It ranks exactly as the pre-CORE-292 framework path did (no rerank unless
    ``rerank``) and differs from it only in where candidates come from. That path
    took the 1,000 most-recently-written rows (``list_entries`` ordered by
    ``updated_at DESC``) and ranked those, so anything older was unreachable no
    matter how relevant. This arm asks the FTS5 index instead -- ``memories_fts``
    already exists in the schema -- and ranks what the *query* selected.

    It is NOT a controlled comparison with ``TrwFrameworkArm``, which now measures
    today's path: that arm also reranks, reads stored vectors and resolves the
    query, so the two differ in ranking as well as candidates. The PRD-CORE-298
    FR05 step 3 gate compares ``TrwQueryPoolArm``, which changes only the pool.

    One wrinkle worth naming: ``search_fts`` phrase-quotes the entire query
    (``'"' + query + '"'``), so a natural-language question matches almost
    nothing as a single phrase. That is very likely why FTS augmentation
    contributes so little in the current recall path. This arm therefore queries
    per-term and unions, which is what an FTS-first generator would have to do
    for real.
    """

    _word = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{2,}")

    def __init__(self, client: Any, namespace: str, rrf_k: int = 5, per_term: int = 200, rerank: bool = False) -> None:
        super().__init__(client, namespace, "trw-fts-first" + ("+rerank" if rerank else ""))
        self._rrf_k = rrf_k
        # The proposed product change is BOTH halves: query-driven candidates AND
        # the reranker the MCP path currently leaves off. Measured separately so
        # the contribution of each is visible rather than inferred.
        self._rerank = rerank
        self._per_term = per_term
        from trw_memory.models.memory import MemoryStatus

        self._status = MemoryStatus.ACTIVE

    def _candidates(self, query: str, limit: int) -> list[Any]:
        backend = self._client._get_backend()
        seen: dict[str, Any] = {}
        # Per-term union. Terms are capped so a long question cannot fan out into
        # a full scan -- the point is sublinear candidate generation, not a
        # cleverer way to read the whole table.
        terms = list(dict.fromkeys(self._word.findall(query)))[:12]
        for term in terms:
            for e in backend.search_fts(term, top_k=self._per_term, namespace=self._ns, status=self._status):
                seen.setdefault(str(getattr(e, "id", "")), e)
        if not seen:
            # No lexical hit at all: fall back to the recency window rather than
            # returning nothing, so this arm is never worse than the current one.
            return list(backend.list_entries(namespace=self._ns, status=self._status, limit=max(limit * 5, 1000)))
        return list(seen.values())

    async def search(self, query: str, limit: int, as_of: datetime) -> list[tuple[str, int]]:
        from trw_memory.embeddings import embed_query
        from trw_memory.models.config import MemoryConfig
        from trw_memory.retrieval.pipeline import hybrid_search
        from trw_memory.security.namespace_scope import authorize_namespaces
        from trw_memory.security.rbac import Permission

        entries = self._candidates(query, limit)
        if not entries:
            return []
        scope = authorize_namespaces(MemoryConfig(), {e.namespace for e in entries}, Permission.READ, "recall")
        embedder = getattr(self._client, "_get_embedder", lambda: None)()
        query_vec = embed_query(embedder, query) if embedder is not None else None
        n = len(entries)
        ranked = hybrid_search(
            query=query,
            entries=entries,
            scope=scope,
            embedder=embedder,
            query_embedding=query_vec,
            bm25_candidates=max(50, n),
            vector_candidates=max(50, n),
            rrf_k=self._rrf_k,
            importance_alpha=1.0,
            top_k=limit,
            **({"rerank": True} if self._rerank else {}),
        )
        return [(str(getattr(e, "id", "")), self._chars(e)) for e in ranked[:limit]]


class Bm25Arm(_TrwArmBase):
    """Lexical only, through trw-memory's own BM25, to separate "the index works"
    from "the fusion and reranking add value"."""

    def __init__(self, client: Any, namespace: str, pool: int = 1000) -> None:
        super().__init__(client, namespace, "bm25")
        self._pool = pool
        from trw_memory.models.memory import MemoryStatus

        self._status = MemoryStatus.ACTIVE

    async def search(self, query: str, limit: int, as_of: datetime) -> list[tuple[str, int]]:
        from trw_memory.retrieval.bm25 import bm25_search

        backend = self._client._get_backend()
        # Same bounded pool and status filter as the framework arm, so the only
        # difference between them is fusion and the dense leg -- not how much
        # corpus each one saw, nor whether retired rows were eligible.
        entries = list(backend.list_entries(namespace=self._ns, status=self._status, limit=max(limit * 5, self._pool)))
        if not entries:
            return []
        by_id = {getattr(e, "id", ""): e for e in entries}
        scored = bm25_search(query, entries, top_k=limit)
        return [(str(rid), self._chars(by_id[rid])) for rid, _s in scored[:limit] if rid in by_id]
