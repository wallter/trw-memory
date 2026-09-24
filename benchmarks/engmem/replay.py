"""Time-travel replay: build the store as it stood at each query's ``as_of``.

The one thing this file exists to guarantee is that **a learning written after an
event cannot be retrieved when scoring that event**. On a benchmark mined from
git history that is the whole game: every learning "predicting" a bug it was
written in response to would score perfectly and mean nothing.

Two ways to get that guarantee, and the cheap one is wrong:

* Retrieve from a full store and filter results by timestamp afterwards. This
  leaks: BM25 statistics, the dense candidate pool and any reranking have already
  seen the future rows, so the surviving ranking is not the ranking the system
  would have produced. Filtering hides the leak instead of preventing it.
* Replay events in order into a fresh store and stop at ``as_of``. Slower, and
  correct. This module does the second, amortised by processing queries in
  ascending ``as_of`` so the store is built once and advanced forward.

Queries are therefore sorted by time and the store only ever moves forward. If a
caller needs them in another order, it re-sorts the *results*, never the replay.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .arms import Arm
from .schema import Event, Query
from .score import Scored

CHARS_PER_TOKEN = 4


@dataclass
class ReplayState:
    """What the store holds right now, for the arms that do not use trw-memory."""

    rows: list[tuple[str, datetime, str]]
    # id -> id: the record that replaced a superseded one, for C3 order scoring.
    successor_of: dict[str, str]
    # Rows retired by a `status_changed` event. Retrieval may still surface them;
    # that is exactly what `forbidden@k` is measuring, so they are NOT removed.
    retired: set[str]


async def apply_event(client: Any, ev: Event, state: ReplayState) -> None:
    """Fold one event into both the trw-memory store and the plain-list mirror.

    Creations delegate to :func:`flush_creates` rather than duplicating the write:
    two write paths would eventually disagree about ``entry_id``, and then the
    ranked ids would stop matching the gold ids for reasons nobody could see.
    """
    if ev.kind == "commit":
        return  # commits drive queries, they are not memories
    if ev.kind == "learning_created":
        await flush_creates(client, [ev], state)
    elif ev.kind == "status_changed" and ev.status in ("obsolete", "retired", "superseded"):
        _retire(client, ev.learning_id, state)
    elif ev.kind == "merged":
        for old in ev.merged_from:
            state.successor_of[old] = ev.learning_id
            _retire(client, old, state)


def _retire(client: Any, learning_id: str, state: ReplayState) -> None:
    """Mark a row obsolete in the STORE, not just in the mirror.

    An earlier version of this file recorded retirement only in ``state.retired``
    and never told trw-memory. Every ``forbidden@k`` number measured against that
    version is void: it asked whether ranking happened to avoid a row the store
    had never been told was retired, which is not the product question. Lifecycle
    is the one thing a memory system is supposed to do that a file scan cannot,
    so it has to be exercised for real.

    ``MemoryClient`` exposes no lifecycle method -- no update, supersede, retire
    or delete -- so this reaches the backend directly. That gap is worth fixing in
    the product: a memory engine whose public API cannot retire a memory cannot
    enforce supersession for its callers either.
    """
    state.retired.add(learning_id)
    if client is None:
        return
    from trw_memory.models.memory import MemoryStatus

    backend = client._get_backend()
    backend.update(learning_id, namespace=client.namespace, status=MemoryStatus.OBSOLETE)


async def flush_creates(client: Any, batch: list[Event], state: ReplayState) -> None:
    """Write a run of ``learning_created`` events in one call.

    Identical in effect to applying them one at a time -- they all precede the
    next query -- but at enterprise store sizes the per-row round trip is most of
    the wall clock, and a benchmark nobody can afford to run at 10^6 rows does
    not measure scale.
    """
    if not batch:
        return
    if client is not None:
        from trw_memory._client_bulk_store import BulkStoreRequest

        await client.bulk_store(
            [
                BulkStoreRequest(
                    content=ev.content,
                    detail=ev.detail,
                    tags=list(ev.tags),
                    metadata={"learning_id": ev.learning_id, "engmem_at": ev.at.isoformat()},
                    source="human",
                    entry_id=ev.learning_id,
                )
                for ev in batch
            ]
        )
    for ev in batch:
        state.rows.append((ev.learning_id, ev.at, f"{ev.content}\n{ev.detail}".strip()))
    batch.clear()


async def replay_and_score(
    events: Sequence[Event],
    queries: Sequence[Query],
    arms: Iterable[Arm],
    *,
    client: Any = None,
    state: ReplayState | None = None,
    limit: int = 10,
) -> dict[str, list[Scored]]:
    """Replay once, scoring every arm at each query instant.

    All arms see the identical store at the identical moment, which is what makes
    the comparison paired: any difference is the retrieval policy, never a
    different corpus or a different cutoff.

    ``state`` is the plain-list mirror the non-trw arms read. Callers pass the
    same object they handed those arms, so every arm advances in lockstep; the
    caller owns it because the arms hold a reference to its ``rows`` list.
    """
    arms = list(arms)
    ordered = sorted(queries, key=lambda q: (q.as_of, q.qid))
    if state is None:
        state = ReplayState(rows=[], successor_of={}, retired=set())

    results: dict[str, list[Scored]] = {a.name: [] for a in arms}
    cursor = 0
    for q in ordered:
        # Advance the store to this query's instant -- never past it. Consecutive
        # creations go in one batched write: at 10^5+ rows a per-row round trip
        # dominates the run without changing what any arm sees, since every event
        # in the batch precedes this query anyway.
        batch: list[Event] = []
        while cursor < len(events) and events[cursor].at <= q.as_of:
            ev = events[cursor]
            cursor += 1
            if ev.kind == "learning_created":
                batch.append(ev)
                continue
            await flush_creates(client, batch, state)
            batch = []
            await apply_event(client, ev, state)
        await flush_creates(client, batch, state)

        for arm in arms:
            t0 = time.perf_counter()
            ranked = await arm.search(q.text, limit, q.as_of)
            ms = (time.perf_counter() - t0) * 1000.0
            results[arm.name].append(
                Scored(
                    qid=q.qid,
                    task=q.task,
                    ranked=tuple(rid for rid, _c in ranked),
                    gold=q.gold_ids,
                    forbidden=q.forbidden_ids,
                    tokens=sum(c for _r, c in ranked) // CHARS_PER_TOKEN,
                    ms=ms,
                )
            )
    return results


def leakage_check(events: Sequence[Event], queries: Sequence[Query]) -> list[str]:
    """Fail-closed audit: report any query whose gold was created after its own
    ``as_of``. Such a query is unanswerable by construction and its presence means
    the extractor, not the retriever, is what is being measured."""
    created: dict[str, datetime] = {e.learning_id: e.at for e in events if e.kind == "learning_created"}
    bad: list[str] = []
    for q in queries:
        for g in q.gold_ids:
            at = created.get(g)
            if at is None:
                bad.append(f"{q.qid}: gold {g} has no creation event")
            elif at > q.as_of:
                bad.append(f"{q.qid}: gold {g} created {at.isoformat()} AFTER as_of {q.as_of.isoformat()}")
    return bad
