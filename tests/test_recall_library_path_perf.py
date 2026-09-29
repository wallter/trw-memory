"""PRD-CORE-318 FR02: recall-time tier discovery resolves one bounded batch, not one ``get`` per row.

FR01's profile put tier discovery at 61-89% of ``MemoryClient.recall`` at 5k-20k rows:
every uncovered warm row was resolved with its own ``backend.get``. These tests pin the
bound (one ``get_many`` for a namespace of any size) and the quality contract behind it:
with fresh snapshots the bounded result is exactly the unbounded one.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.lifecycle.tiers import (
    _manager_search,
    _runtime,
    _warm_discovery,
    _warm_sidecar_cache,
    _warm_space,
)
from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recall_selection import LocalCandidate, RecallInvocation
from trw_memory.retrieval.source_policy import SourcePolicy
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.storage._vector_ops import get_vector_records
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _entry(entry_id: str, *, kind: str = "episodic", importance: float = 0.5, content: str = "needle") -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content,
        importance=importance,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        metadata={"source_kind": kind},
    )


def _policy() -> RecallInvocation:
    return RecallInvocation(SourcePolicy.resolve(), TemporalSelection(), "default")


@pytest.fixture
def tiers(tmp_path: Path) -> Iterator[tuple[TierManager, SQLiteBackend]]:
    config = MemoryConfig(storage_path=str(tmp_path), hot_max_entries=5)
    manager = _runtime.get_tier_manager(config, "default")
    with SQLiteBackend(tmp_path / "canonical.db") as backend:
        yield manager, backend
    manager.close()


def _load(manager: TierManager, backend: SQLiteBackend, entries: list[MemoryEntry]) -> None:
    backend.store_many(entries)
    manager.warm_add_many([(e.id, e.model_dump(mode="json"), None) for e in entries])


class _Counting:
    """``resolve_entries`` over the real backend, recording each batch; ``get`` is refused."""

    def __init__(self, backend: SQLiteBackend) -> None:
        self.backend = backend
        self.batches: list[list[str]] = []

    def __call__(self, entry_ids: list[str]) -> Mapping[str, MemoryEntry]:
        self.batches.append(list(entry_ids))
        return self.backend.get_many(entry_ids, namespace="default")


def _discover(manager: TierManager, resolve: _Counting, top_k: int) -> list[LocalCandidate]:
    found = manager.search(["needle"], invocation=_policy(), resolve_entries=resolve, top_k=top_k)
    return [c for c in found if isinstance(c, LocalCandidate)]


@pytest.mark.parametrize("rows", [200, 2000])
def test_one_batched_read_and_a_bound_independent_of_namespace_size(
    tiers: tuple[TierManager, SQLiteBackend], rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, backend = tiers
    _load(manager, backend, [_entry(f"e{i:05}", importance=(i % 97) / 97) for i in range(rows)])
    monkeypatch.setattr(backend, "get", lambda *_a, **_k: pytest.fail("per-row get on the recall path"))
    resolve = _Counting(backend)

    found = _discover(manager, resolve, top_k=5)

    assert len(found) == 5
    assert len(resolve.batches) == 1
    assert len(resolve.batches[0]) <= _manager_search.RESOLVE_MARGIN * 5


def test_a_warm_row_in_the_true_top_k_is_found_under_the_bound(tiers: tuple[TierManager, SQLiteBackend]) -> None:
    """A low-importance durable row outranks every episodic row by source policy.

    It is the last of 1,001 warm rows, 50x the bound, so it is found only because the
    scan orders by the recall rank key, not by arrival or raw score.
    """
    manager, backend = tiers
    episodic = [_entry(f"e{i:05}", importance=0.9) for i in range(1000)]
    durable = _entry("zz-durable", kind="semantic_memory", importance=0.1)
    _load(manager, backend, [*episodic, durable])

    found = _discover(manager, _Counting(backend), top_k=1)

    assert [c.entry.id for c in found] == [durable.id]


def test_bounded_result_equals_unbounded_on_fresh_snapshots(
    tiers: tuple[TierManager, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, backend = tiers
    rng = random.Random(7)
    kinds = ("episodic", "semantic_memory", "procedural")
    words = ("needle", "needle haystack", "haystack")
    _load(
        manager,
        backend,
        [
            _entry(f"e{i:04}", kind=rng.choice(kinds), importance=rng.random(), content=rng.choice(words))
            for i in range(600)
        ],
    )

    bounded = [c.entry.id for c in _discover(manager, _Counting(backend), top_k=10)]
    monkeypatch.setattr(_manager_search, "RESOLVE_MARGIN", 10_000)
    unbounded = [c.entry.id for c in _discover(manager, _Counting(backend), top_k=10)]

    assert bounded == unbounded
    assert len(bounded) == 10


def test_canonical_policy_wins_over_a_stale_snapshot(tiers: tuple[TierManager, SQLiteBackend]) -> None:
    """Admission is judged on the canonical row: a snapshot the policy would reject is still resolved."""
    manager, backend = tiers
    stale = _entry("changed", kind="episodic")
    manager.warm_add_many([(stale.id, stale.model_dump(mode="json"), None)])
    backend.store(_entry("changed", kind="semantic_memory"))
    invocation = RecallInvocation(
        SourcePolicy.resolve(exclude_source_kinds=["episodic"]), TemporalSelection(), "default"
    )

    found = manager.search(["needle"], invocation=invocation, resolve_entries=_Counting(backend), top_k=1)

    assert [c.entry.metadata["source_kind"] for c in found if isinstance(c, LocalCandidate)] == ["semantic_memory"]


def test_admission_rejections_never_fetch_past_the_bound(tiers: tuple[TierManager, SQLiteBackend]) -> None:
    """Review r2 P1-1: one get_many of at most RESOLVE_MARGIN * top_k ids per recall, in total.

    Every matching row is rejected by admission (it lacks the required tag), so recall
    returns fewer candidates; it does not keep fetching.
    """
    manager, backend = tiers
    _load(manager, backend, [_entry(f"t{i}").model_copy(update={"tags": ["other"]}) for i in range(9)])
    invocation = RecallInvocation(SourcePolicy.resolve(), TemporalSelection(), "default", tags=frozenset({"wanted"}))
    resolve = _Counting(backend)

    found = manager.search(["needle"], invocation=invocation, resolve_entries=resolve, top_k=1)

    assert found == []
    assert len(resolve.batches) == 1
    assert sum(len(batch) for batch in resolve.batches) <= _manager_search.RESOLVE_MARGIN * 1


def test_stale_snapshot_text_does_not_exclude_a_canonical_match(tiers: tuple[TierManager, SQLiteBackend]) -> None:
    """Review r2 P1-2: the snapshot's text lacks the query token, the canonical entry has it."""
    manager, backend = tiers
    stale = _entry("edited", content="haystack")
    manager.warm_add_many([(stale.id, stale.model_dump(mode="json"), None)])
    backend.store(_entry("edited", content="needle"))

    found = _discover(manager, _Counting(backend), top_k=1)

    assert [c.entry.id for c in found] == ["edited"]
    assert found[0].entry.content == "needle"


def test_an_id_the_read_layer_withholds_never_returns_through_its_snapshot(
    tiers: tuple[TierManager, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review r2 P1-3 (the CORE-333 guarantee): the store holds ``bad`` but get_many omits it
    (quarantined); a valid tier snapshot of ``bad`` must not stand in for it."""
    manager, backend = tiers
    _load(manager, backend, [_entry("bad"), _entry("good")])
    real = backend.get_many
    monkeypatch.setattr(
        backend,
        "get_many",
        lambda ids, *, namespace: {k: v for k, v in real(ids, namespace=namespace).items() if k != "bad"},
    )

    found = _runtime.tier_candidates(
        manager._config, "default", backend, query="needle", tags=None, limit=5, invocation=_policy()
    )

    assert [c.entry.id for c in found if isinstance(c, LocalCandidate)] == ["good"]


def test_a_row_the_store_never_held_still_surfaces_from_its_snapshot(tiers: tuple[TierManager, SQLiteBackend]) -> None:
    """Cold-archive and primary-less warm rows keep their snapshot fallback."""
    manager, backend = tiers
    orphan = _entry("warm-only")
    manager.warm_add_many([(orphan.id, orphan.model_dump(mode="json"), None)])

    found = _runtime.tier_candidates(
        manager._config, "default", backend, query="needle", tags=None, limit=5, invocation=_policy()
    )

    assert [c.entry.id for c in found if isinstance(c, LocalCandidate)] == ["warm-only"]


def test_a_malformed_unrelated_snapshot_is_skipped_not_fatal() -> None:
    """Review r3: an uncovered snapshot whose ``importance`` does not parse must not abort discovery.

    Its snapshot says nothing usable, so it neither orders nor excludes the row: the canonical
    row decides, and with none (as here) the row is dropped with a logged reason.
    """
    found = _manager_search.discover_candidates(
        [({"id": "unrelated", "content": "haystack", "importance": "invalid"}, False)],
        invocation=_policy(),
        resolve_entries=lambda _ids: {},
        query_tokens=["needle"],
        query_embedding=None,
        config=MemoryConfig(),
        top_k=5,
    )

    assert found == []


# --- FR02b: warm discovery by sqlite-vec KNN, not a full vector scan ---------------------------

SPACE = EmbeddingSpace("a" * 64, "test-encoder:fr02b", 4)
#: A source kind in recall's first bucket; an episodic window widens to the full scan (see the next tests).
DURABLE = "semantic_memory"


def _unit(rng: random.Random) -> list[float]:
    raw = [rng.gauss(0.0, 1.0) for _ in range(4)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _load_vectored(manager: TierManager, backend: SQLiteBackend, entries: list[MemoryEntry], seed: int = 7) -> None:
    """Store *entries* canonically and in the warm tier, each with an in-space unit vector."""
    rng = random.Random(seed)
    backend.store_many(entries)
    for entry in entries:
        vector = _unit(rng)
        proof = VectorProvenance.for_vector(SPACE, f"{entry.content} ", vector)
        manager._warm_store.warm_add(entry.id, entry.model_dump(mode="json"), vector, provenance=proof)


def _vector_discover(manager: TierManager, backend: SQLiteBackend, tokens: list[str], top_k: int) -> list[str]:
    found = manager.search(
        tokens,
        query_embedding=_unit(random.Random(99)),
        query_space=SPACE,
        invocation=_policy(),
        resolve_entries=_Counting(backend),
        top_k=top_k,
    )
    return [c.entry.id for c in found if isinstance(c, LocalCandidate)]


@pytest.mark.parametrize("rows", [100, 400])
def test_warm_rows_scanned_per_recall_are_bounded_independent_of_n(
    tiers: tuple[TierManager, SQLiteBackend], rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR02b: the warm tier decodes and returns a window of at most 2M rows here (M = RESOLVE_MARGIN * top_k), at any N.

    At FR02 (b0bcd76c6) both counts were N: every uncovered warm vector was decoded, L2-scored in
    Python and handed to discovery.
    """
    manager, backend = tiers
    _load_vectored(manager, backend, [_entry(f"v{i:05}", kind=DURABLE, content="haystack") for i in range(rows)])
    decoded: list[int] = []
    returned: list[int] = []
    decode = get_vector_records
    scan = WarmTierStore.discovery_entries

    def counting_decode(*args: Any, **kwargs: Any) -> Any:
        decoded.append(len(kwargs.get("entry_ids") or []))
        return decode(*args, **kwargs)

    def counting_scan(self: WarmTierStore, *args: Any, **kwargs: Any) -> Any:
        found = scan(self, *args, **kwargs)
        returned.append(len(found))
        return found

    monkeypatch.setattr(_warm_space, "get_vector_records", counting_decode)
    monkeypatch.setattr(WarmTierStore, "discovery_entries", counting_scan)
    top_k = 2
    bound = _manager_search.RESOLVE_MARGIN * top_k

    found = _vector_discover(manager, backend, ["needle"], top_k)

    assert len(found) == top_k
    # The window is the first one the stop rule accepts: a function of the query and the data
    # near it, not of N (review r1 P1: it starts at 2M and widens x4 until provably complete).
    assert sum(decoded) <= 2 * bound, f"decoded {sum(decoded)} warm vectors for N={rows}, bound {bound}"
    assert returned == [sum(decoded)], f"warm discovery returned {returned} rows for N={rows}"


def test_bounded_knn_discovery_equals_the_unbounded_full_scan(
    tiers: tuple[TierManager, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a store where the composite score orders like the vector distance (equal importance and
    age), the M nearest hold the whole kept set, so the bounded result is the full scan's.

    The reference is FR02's unbounded warm scan (``limit=None``), with everything else unchanged.
    """
    manager, backend = tiers
    words = ("needle", "haystack")
    _load_vectored(manager, backend, [_entry(f"v{i:03}", kind=DURABLE, content=words[i % 2]) for i in range(60)])
    orphan = _entry("tokens-only", kind=DURABLE, content="needle")
    backend.store(orphan)
    manager.warm_add_many([(orphan.id, orphan.model_dump(mode="json"), None)])

    bounded = _vector_discover(manager, backend, ["needle"], top_k=3)
    scan = WarmTierStore.discovery_entries

    def full_scan(self: WarmTierStore, *args: Any, **kwargs: Any) -> Any:
        return scan(self, *args, **{**kwargs, "limit": None})

    monkeypatch.setattr(WarmTierStore, "discovery_entries", full_scan)
    unbounded = _vector_discover(manager, backend, ["needle"], top_k=3)

    assert bounded == unbounded
    assert len(bounded) == 3


def test_a_vectorless_warm_row_is_still_found_by_its_token(tiers: tuple[TierManager, SQLiteBackend]) -> None:
    """A warm row with no vector is outside the KNN; its snapshot's token match still brings it in."""
    manager, backend = tiers
    _load_vectored(manager, backend, [_entry(f"v{i:03}", kind=DURABLE, content="haystack") for i in range(50)])
    plain = _entry("no-vector", kind=DURABLE, content="needle")
    backend.store(plain)
    manager.warm_add_many([(plain.id, plain.model_dump(mode="json"), None)])

    found = _vector_discover(manager, backend, ["needle"], top_k=2)

    assert "no-vector" in found


def test_a_better_ranked_row_past_the_window_widens_it(tiers: tuple[TierManager, SQLiteBackend]) -> None:
    """Source policy ranks any durable row above every episodic one, whatever the distance, so a
    window of nearer episodic rows must widen until it reaches the far durable row."""
    manager, backend = tiers
    _load_vectored(manager, backend, [_entry(f"e{i:03}", content="needle") for i in range(40)])
    durable = _entry("far-durable", kind=DURABLE, content="haystack")
    backend.store(durable)
    far = [-x for x in _unit(random.Random(99))]
    proof = VectorProvenance.for_vector(SPACE, f"{durable.content} ", far)
    manager._warm_store.warm_add(durable.id, durable.model_dump(mode="json"), far, provenance=proof)

    assert _vector_discover(manager, backend, ["needle"], top_k=1) == ["far-durable"]


def test_a_window_of_superseded_rows_widens_to_an_eligible_one(tiers: tuple[TierManager, SQLiteBackend]) -> None:
    """Temporal eligibility ranks before distance too: superseded near rows do not fill the window."""
    manager, backend = tiers
    closed = datetime(2021, 1, 1, tzinfo=timezone.utc)
    stale = [
        _entry(f"old{i:03}", kind=DURABLE).model_copy(update={"invalid_from": closed, "invalidated_by": "new"})
        for i in range(40)
    ]
    _load_vectored(manager, backend, stale)
    current = _entry("current", kind=DURABLE, content="haystack")
    backend.store(current)
    far = [-x for x in _unit(random.Random(99))]
    proof = VectorProvenance.for_vector(SPACE, f"{current.content} ", far)
    manager._warm_store.warm_add(current.id, current.model_dump(mode="json"), far, provenance=proof)

    assert _vector_discover(manager, backend, ["needle"], top_k=1) == ["current"]


# --- FR02b review r1 P1: the window stops only when no row past it can outrank its M-th row ---------


def _put_warm(manager: TierManager, backend: SQLiteBackend, entry: MemoryEntry, vector: list[float]) -> None:
    backend.store(entry)
    proof = VectorProvenance.for_vector(SPACE, f"{entry.content} ", vector)
    manager._warm_store.warm_add(entry.id, entry.model_dump(mode="json"), vector, provenance=proof)


def _search(manager: TierManager, backend: SQLiteBackend, query: list[float], top_k: int) -> list[str]:
    found = manager.search(
        ["needle"],
        query_embedding=query,
        query_space=SPACE,
        invocation=_policy(),
        resolve_entries=_Counting(backend),
        top_k=top_k,
    )
    return [c.entry.id for c in found if isinstance(c, LocalCandidate)]


def _bounded_and_unbounded(
    manager: TierManager, backend: SQLiteBackend, query: list[float], top_k: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], list[str]]:
    bounded = _search(manager, backend, query, top_k)
    scan = WarmTierStore.discovery_entries
    with monkeypatch.context() as patch:
        patch.setattr(WarmTierStore, "discovery_entries", lambda self, *a, **k: scan(self, *a, **{**k, "limit": None}))
        unbounded = _search(manager, backend, query, top_k)
    return bounded, unbounded


def _at_cosine(cos: float, axis: int, sign: float = 1.0) -> list[float]:
    """A unit vector at cosine *cos* to (1, 0, 0, 0), leaning along *axis*."""
    vector = [cos, 0.0, 0.0, 0.0]
    vector[axis] = sign * math.sqrt(1.0 - cos * cos)
    return vector


def test_a_row_past_the_distance_window_that_wins_on_importance_is_kept(
    tiers: tuple[TierManager, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review r1 P1 repro: four nearer rows (relevance 0.99, importance 0) score ~0.696; the fifth
    (relevance 0.98, importance 1) scores ~0.992 and is the true top-1, just past a window of M=4."""
    manager, backend = tiers
    stamp = datetime.now(timezone.utc)
    near = [(1, 1.0), (2, 1.0), (3, 1.0), (1, -1.0)]
    for i, (axis, sign) in enumerate(near):
        entry = _entry(f"near{i}", kind=DURABLE, importance=0.0).model_copy(update={"created_at": stamp})
        _put_warm(manager, backend, entry, _at_cosine(0.99, axis, sign))
    winner = _entry("far-important", kind=DURABLE, importance=1.0).model_copy(update={"created_at": stamp})
    _put_warm(manager, backend, winner, _at_cosine(0.98, 2, -1.0))

    bounded, unbounded = _bounded_and_unbounded(manager, backend, [1.0, 0.0, 0.0, 0.0], 1, monkeypatch)

    assert unbounded == ["far-important"]
    assert bounded == unbounded


_SEED_CHUNKS = 8
_SEEDS_PER_CHUNK = 25  # 8 x 25 = the 200 seeded stores, split so xdist can spread them


@pytest.mark.parametrize("chunk", range(_SEED_CHUNKS))
def test_bounded_equals_unbounded_over_random_small_stores(
    chunk: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Property check over 200 seeded stores: bounded top-k == unbounded top-k, again after a write on every third.

    Half the stores draw every term at random (relevance, importance, age, source kind), so the
    window usually has to widen to the whole store. The other half share one importance and age
    except for a few random outliers (the review repro's shape), so the window can stop early and
    the stop rule itself is exercised; the test requires that it did, many times.
    Each chunk covers 25 consecutive seeds; the seeds are deterministic, so so is each
    chunk's early-stop count.
    """
    now = datetime.now(timezone.utc)
    kinds = (DURABLE, "episodic", "procedural")
    early = 0
    real_window = _warm_discovery._knn_window

    def window(*args: Any) -> Any:
        nonlocal early
        in_space, found = real_window(*args)
        early += found is not None and len(found) < len(in_space)
        return in_space, found

    monkeypatch.setattr(_warm_discovery, "_knn_window", window)
    for seed in range(chunk * _SEEDS_PER_CHUNK, (chunk + 1) * _SEEDS_PER_CHUNK):
        rng = random.Random(seed)
        uniform = seed % 2 == 0
        config = MemoryConfig(storage_path=str(tmp_path / f"s{seed}"), hot_max_entries=5)
        manager = _runtime.get_tier_manager(config, "default")
        with SQLiteBackend(tmp_path / f"s{seed}" / "canonical.db") as backend:
            for i in range(rng.randint(20, 40)):
                outlier = not uniform or rng.random() < 0.1
                age = timedelta(days=rng.choice((0, 0, 1, 3, 20, 90)) if outlier else 0)
                importance = round(rng.random(), 2) if outlier else 0.5
                kind = rng.choice(kinds) if outlier else DURABLE
                entry = _entry(f"r{i:02}", kind=kind, importance=importance).model_copy(
                    update={"created_at": now - age}
                )
                _put_warm(manager, backend, entry, _unit(rng))
            top_k = rng.randint(1, 2)
            query = _unit(rng)
            bounded, unbounded = _bounded_and_unbounded(manager, backend, query, top_k, monkeypatch)
            assert bounded == unbounded, f"seed {seed}"
            if seed % 3 == 0:  # a write between recalls: a new version, possibly a new maximum
                late = _entry("late", kind=DURABLE, importance=round(rng.random(), 2))
                _put_warm(manager, backend, late.model_copy(update={"created_at": now}), _unit(rng))
                bounded, unbounded = _bounded_and_unbounded(manager, backend, query, top_k, monkeypatch)
                assert bounded == unbounded, f"seed {seed} after a write"
        manager.close()
        _runtime._TIER_MANAGER_CACHE.clear()
    # The unsplit run required >= 20 of 200 seeds; 3 per chunk keeps that total (8 x 3 = 24).
    # Observed per chunk: 6-14 (86 in all), deterministic by seed.
    assert early >= 3, f"chunk {chunk}: the stop rule cut the window early only {early} times"


def test_uniform_importance_and_age_lets_the_window_stop_early(
    tiers: tuple[TierManager, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no row past the window can beat it on the other terms, relevance alone decides and the
    window stops long before N."""
    manager, backend = tiers
    stamp = datetime.now(timezone.utc)
    rng = random.Random(3)
    for i in range(400):
        entry = _entry(f"u{i:03}", kind=DURABLE, importance=0.5).model_copy(update={"created_at": stamp})
        _put_warm(manager, backend, entry, _unit(rng))
    decoded: list[int] = []
    decode = get_vector_records

    def counting_decode(*args: Any, **kwargs: Any) -> Any:
        decoded.append(len(kwargs.get("entry_ids") or []))
        return decode(*args, **kwargs)

    query = _unit(random.Random(4))
    monkeypatch.setattr(_warm_space, "get_vector_records", counting_decode)
    bounded = _search(manager, backend, query, 1)
    monkeypatch.undo()
    bounded_again, unbounded = _bounded_and_unbounded(manager, backend, query, 1, monkeypatch)

    assert bounded == bounded_again == unbounded
    assert sum(decoded) <= 64, f"decoded {sum(decoded)} of 400 warm vectors"


# --- FR02b review r2: the ceiling's maxima are cached per sidecar version, O(1) per recall -------


@pytest.mark.parametrize("rows", [100, 400])
def test_a_warm_recall_does_no_per_row_ceiling_work(
    tiers: tuple[TierManager, SQLiteBackend], rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a sidecar version's maxima are known, a recall reads them; no per-row access-date pass."""
    manager, backend = tiers
    stamp = datetime.now(timezone.utc)
    rng = random.Random(5)
    for i in range(rows):
        entry = _entry(f"w{i:03}", kind=DURABLE).model_copy(update={"created_at": stamp})
        _put_warm(manager, backend, entry, _unit(rng))
    query = _unit(random.Random(6))
    _search(manager, backend, query, 1)  # first recall of this version may fill the cache
    calls = 0

    def counting(real: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return real(*args, **kwargs)

        return wrapped

    for module in (_manager_search, _warm_discovery, _warm_sidecar_cache):
        if hasattr(module, "days_since_access"):
            monkeypatch.setattr(module, "days_since_access", counting(module.days_since_access))

    _search(manager, backend, query, 1)

    assert calls == 0, f"{calls} per-row access-date reads on a warm recall over {rows} rows"


def test_a_write_raising_importance_past_the_cached_maximum_is_seen(
    tiers: tuple[TierManager, SQLiteBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row written after the maxima were cached, with a higher importance than any before it, far
    from the query: a stale ceiling would stop the window before it."""
    manager, backend = tiers
    stamp = datetime.now(timezone.utc)
    rng = random.Random(8)
    for i in range(60):
        entry = _entry(f"low{i:02}", kind=DURABLE, importance=0.1).model_copy(update={"created_at": stamp})
        _put_warm(manager, backend, entry, _unit(rng))
    query = [1.0, 0.0, 0.0, 0.0]
    first, _ = _bounded_and_unbounded(manager, backend, query, 1, monkeypatch)
    heavy = _entry("heavy", kind=DURABLE, importance=1.0).model_copy(update={"created_at": stamp})
    _put_warm(manager, backend, heavy, _at_cosine(0.5, 2))

    bounded, unbounded = _bounded_and_unbounded(manager, backend, query, 1, monkeypatch)

    assert unbounded == ["heavy"] != first
    assert bounded == unbounded
