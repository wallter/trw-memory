"""PRD-CORE-279 FR01-FR03/NFR01: the process-level embedding-provider cache.

The daemon built a provider -- and loaded the model -- once per request. These
tests pin the cache's identity rules rather than its speed: same key reuses,
policy change evicts, failure is not remembered, and a security refusal is
still a refusal.
"""

from __future__ import annotations

import threading

import pytest

import trw_memory.embeddings as embeddings_pkg
from trw_memory.embeddings import get_local_embedder, reset_provider_cache
from trw_memory.exceptions import LocalOnlyViolationError, RemoteCodeNotPermittedError


class _FakeProvider:
    """A provider that records how many times a model "load" happened."""

    loads = 0

    def __init__(self, model_name: str = "m", dim: int = 4, *, available: bool = True) -> None:
        self._model_name = model_name
        self._dim = dim
        self._available = available

    def available(self) -> bool:
        type(self).loads += 1
        return self._available

    def dim(self) -> int:
        return self._dim


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_provider_cache()
    _FakeProvider.loads = 0
    yield
    reset_provider_cache()


def _install(monkeypatch, factory) -> None:
    monkeypatch.setattr(embeddings_pkg, "LocalEmbeddingProvider", factory)


def test_repeated_calls_reuse_one_provider(monkeypatch):
    """FR01: 40 calls, one construction and one load."""
    built: list[_FakeProvider] = []

    def _factory(*, model_name: str, dim: int) -> _FakeProvider:
        provider = _FakeProvider(model_name, dim)
        built.append(provider)
        return provider

    _install(monkeypatch, _factory)

    first = get_local_embedder()
    for _ in range(39):
        assert get_local_embedder() is first
    assert len(built) == 1
    assert _FakeProvider.loads == 1


def test_cache_holds_one_provider_per_key(monkeypatch):
    """NFR04: a different model or dimension is a different provider."""
    _install(monkeypatch, lambda *, model_name, dim: _FakeProvider(model_name, dim))

    a = get_local_embedder(model_name="alpha", dim=8)
    b = get_local_embedder(model_name="beta", dim=8)
    c = get_local_embedder(model_name="alpha", dim=16)

    assert a is not b
    assert a is not c
    assert get_local_embedder(model_name="alpha", dim=8) is a


def test_concurrent_first_calls_construct_one_provider(monkeypatch):
    """FR02: initialisation is serialised, so N first callers load once."""
    barrier = threading.Barrier(8)
    built: list[_FakeProvider] = []

    def _slow_factory(*, model_name: str, dim: int) -> _FakeProvider:
        provider = _FakeProvider(model_name, dim)
        built.append(provider)
        return provider

    _install(monkeypatch, _slow_factory)

    results: list[object] = []
    results_lock = threading.Lock()

    def _call() -> None:
        barrier.wait(timeout=10)
        provider = get_local_embedder()
        with results_lock:
            results.append(provider)

    threads = [threading.Thread(target=_call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive(), "a caller blocked: the init lock is not being released"

    assert len(built) == 1, f"expected one construction, got {len(built)}"
    assert len(results) == 8
    assert all(item is results[0] for item in results)


def test_transient_failure_is_not_cached(monkeypatch):
    """FR02 negative: a provider that could not load is retried, not remembered."""
    attempts: list[int] = []

    def _flaky(*, model_name: str, dim: int) -> _FakeProvider:
        attempts.append(1)
        return _FakeProvider(model_name, dim, available=len(attempts) > 1)

    _install(monkeypatch, _flaky)

    assert get_local_embedder() is None
    second = get_local_embedder()
    assert second is not None
    assert len(attempts) == 2
    # And the retry's success IS cached, so the distinction is failure-only.
    assert get_local_embedder() is second
    assert len(attempts) == 2


def test_policy_change_invalidates_the_cache(monkeypatch):
    """FR03: a tightened offline policy is never served by the old provider."""
    _install(monkeypatch, lambda *, model_name, dim: _FakeProvider(model_name, dim))
    monkeypatch.delenv("TRW_OFFLINE", raising=False)

    first = get_local_embedder()
    assert get_local_embedder() is first, "nothing was cached, so this proves nothing"
    monkeypatch.setenv("TRW_OFFLINE", "1")
    second = get_local_embedder()

    assert second is not first, "the provider survived a policy change"
    assert get_local_embedder() is second, "the new policy's provider was not cached"


def test_pid_change_invalidates_the_cache(monkeypatch):
    """FR03: a forked child rebuilds rather than serving the parent's provider."""
    _install(monkeypatch, lambda *, model_name, dim: _FakeProvider(model_name, dim))

    first = get_local_embedder()
    assert get_local_embedder() is first, "nothing was cached, so this proves nothing"

    import trw_memory.embeddings._provider_cache as cache_mod

    real_pid = cache_mod.os.getpid()
    monkeypatch.setattr(cache_mod.os, "getpid", lambda: real_pid + 1)
    second = get_local_embedder()

    assert second is not first


@pytest.mark.parametrize(
    "error",
    [
        LocalOnlyViolationError("model not in the local cache"),
        RemoteCodeNotPermittedError("repository ships python modules"),
    ],
)
def test_security_refusal_still_raises_and_is_not_cached(monkeypatch, error):
    """NFR01: caching must not turn a fail-closed refusal into a cached answer."""
    calls: list[int] = []

    def _refusing(*, model_name: str, dim: int) -> _FakeProvider:
        calls.append(1)
        raise error

    _install(monkeypatch, _refusing)

    with pytest.raises(type(error)):
        get_local_embedder()
    with pytest.raises(type(error)):
        get_local_embedder()
    assert len(calls) == 2, "the refusal was cached instead of re-evaluated"


def test_provider_serialises_its_own_encode_calls():
    """NFR01: sentence-transformers encode is not re-entrant, so it is locked."""
    from trw_memory.embeddings.local import LocalEmbeddingProvider

    provider = LocalEmbeddingProvider(model_name="x", dim=4)
    assert hasattr(provider, "_encode_lock")

    observed: list[int] = []
    in_flight = 0
    ready = threading.Barrier(4)

    class _Model:
        def encode(self, text, **kwargs):
            nonlocal in_flight
            in_flight += 1
            observed.append(in_flight)
            # Long enough that an unsynchronised second caller would overlap.
            threading.Event().wait(0.02)
            in_flight -= 1
            return [0.0, 0.0, 0.0, 0.0]

    provider._model = _Model()
    provider._load_attempted = True

    def _embed() -> None:
        ready.wait(timeout=10)
        provider.embed("hello world")

    threads = [threading.Thread(target=_embed) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()

    assert observed and max(observed) == 1, f"concurrent encode observed: {observed}"


def _run_with_deadline(fn, seconds: float = 5.0) -> object:
    """Run *fn* on a thread and fail the test if it does not finish in time."""
    outcome: dict[str, object] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    assert not thread.is_alive(), f"call did not finish within {seconds}s -- the cache deadlocked"
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome.get("value")


def test_a_factory_that_re_enters_the_cache_does_not_deadlock(monkeypatch):
    """FR02 regression: the load must never run under the shared lock.

    The first version of this module called the factory inside the global
    ``threading.Lock``. Any re-entry from the load path -- and the model load
    has several -- then hung the process forever; three full-suite runs on this
    box sat at 0% CPU until they were killed.
    """
    seen: list[object] = []

    def _reentrant(*, model_name: str, dim: int) -> _FakeProvider:
        seen.append(get_local_embedder(model_name=model_name, dim=dim))
        return _FakeProvider(model_name, dim)

    _install(monkeypatch, _reentrant)

    provider = _run_with_deadline(lambda: get_local_embedder())
    assert provider is not None
    assert seen == [None], "the re-entrant call should be answered 'nothing loaded yet'"
    # And the outer call's provider is the one that got cached.
    assert get_local_embedder() is provider


def test_a_slow_load_does_not_block_a_different_model(monkeypatch):
    """FR02: the lock guards bookkeeping, so two models can load at once."""
    started = threading.Event()
    release = threading.Event()

    def _factory(*, model_name: str, dim: int) -> _FakeProvider:
        if model_name == "slow":
            started.set()
            assert release.wait(timeout=10), "the fast load never ran"
        return _FakeProvider(model_name, dim)

    _install(monkeypatch, _factory)

    slow_result: list[object] = []
    slow = threading.Thread(target=lambda: slow_result.append(get_local_embedder(model_name="slow", dim=4)))
    slow.start()
    assert started.wait(timeout=10)

    fast = _run_with_deadline(lambda: get_local_embedder(model_name="fast", dim=4))
    assert fast is not None

    release.set()
    slow.join(timeout=10)
    assert not slow.is_alive()
    assert slow_result and slow_result[0] is not None


def test_two_threads_building_each_others_models_do_not_deadlock(monkeypatch):
    """FR02 regression: a builder must never block on another builder.

    A builds X and needs Y; B builds Y and needs X. If a builder is allowed to
    wait on another builder's holder, both park forever -- which is the second
    shape of the deadlock that hung this box.
    """
    ready = threading.Barrier(2)

    def _factory(*, model_name: str, dim: int) -> _FakeProvider:
        other = "y" if model_name == "x" else "x"
        ready.wait(timeout=10)
        get_local_embedder(model_name=other, dim=4)
        return _FakeProvider(model_name, dim)

    _install(monkeypatch, _factory)

    results: dict[str, object] = {}

    def _call(name: str) -> None:
        results[name] = get_local_embedder(model_name=name, dim=4)

    threads = [threading.Thread(target=_call, args=(name,)) for name in ("x", "y")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "two builders deadlocked on each other"
    assert results["x"] is not None
    assert results["y"] is not None


def test_a_reset_during_a_build_is_not_undone_by_it(monkeypatch):
    """FR03: a provider built before a reset must not be published after it."""
    building = threading.Event()
    release = threading.Event()

    def _slow(*, model_name: str, dim: int) -> _FakeProvider:
        building.set()
        assert release.wait(timeout=10)
        return _FakeProvider(model_name, dim)

    _install(monkeypatch, _slow)

    built: list[object] = []
    worker = threading.Thread(target=lambda: built.append(get_local_embedder()))
    worker.start()
    assert building.wait(timeout=10)

    reset_provider_cache()
    release.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert built and built[0] is not None, "the builder still gets its own result"

    # But the reset stands: the next caller builds again rather than being
    # served the provider that was in flight when the cache was cleared.
    _install(monkeypatch, lambda *, model_name, dim: _FakeProvider(model_name, dim))
    assert get_local_embedder() is not built[0]


def test_a_waiter_gives_up_on_a_builder_that_never_finishes(monkeypatch):
    """FR02: an unbounded wait would park every server worker behind one load."""
    import trw_memory.embeddings._provider_cache as cache_mod

    monkeypatch.setattr(cache_mod, "BUILD_WAIT_TIMEOUT_SECONDS", 0.2)
    started = threading.Event()
    release = threading.Event()

    def _never_finishes(*, model_name: str, dim: int) -> _FakeProvider:
        started.set()
        release.wait(timeout=30)
        return _FakeProvider(model_name, dim)

    _install(monkeypatch, _never_finishes)

    builder = threading.Thread(target=get_local_embedder, daemon=True)
    builder.start()
    assert started.wait(timeout=10)

    assert _run_with_deadline(lambda: get_local_embedder(), seconds=5.0) is None
    release.set()
    builder.join(timeout=10)
