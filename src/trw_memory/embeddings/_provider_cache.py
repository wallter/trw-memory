"""Process-level embedding-provider cache -- PRD-CORE-279 FR01/FR02/FR03.

``LocalEmbeddingProvider`` deliberately keeps its loaded model as an instance
attribute, which is right for a process that builds one provider and wrong for
the loopback daemon, which built one per request: 40 recalls produced 40 model
loads (measured 2026-09-17; the field report sub_MpOjKl-FCnHldYZs measured 143
loads in 40 recalls and p95 1105 ms). This module adds the missing identity --
one provider per process, per model, per dimension, per policy -- without
touching the loader or any of its refusals.

**The model load NEVER happens under the shared lock.** The first version of
this module called the factory inside the global lock, which deadlocked every
parallel test run on the box: a lock held across a multi-second model load is
already bad, and the load path can re-enter this function, which a plain
``threading.Lock`` answers by hanging forever. So the lock only guards the
bookkeeping. A caller that finds no provider installs a ``_Pending`` holder,
releases the lock, builds outside it, and then publishes; concurrent callers
wait on that holder's event instead of starting a second load. A re-entrant
call on the SAME thread cannot wait for itself -- it would be waiting for work
it is itself supposed to finish -- so it is answered ``None``.

Three further properties the cache must not trade away:

**Only a working provider is cached.** ``available()`` is what loads the
weights, so a provider that fails to load is a *transient* answer: the library
may be installed a minute later, or the model pulled into the cache. Caching
``None`` would make the first failure permanent for the life of the process.

**Policy decides identity, not just the model name.** ``local_only``,
``embedding_trust_remote_code`` and the offline switches (PRD-SEC-014,
PRD-QUAL-110) select *what may be loaded and from where*; the HuggingFace cache
roots select *which snapshot*. All of them are part of the fingerprint, and a
fingerprint change DROPS the cache rather than adding a second entry -- a
provider built under a looser policy must not survive a tightening.

**A forked child is not this process.** The fingerprint carries the pid, so a
child rebuilds instead of serving a provider whose model was loaded before the
fork. That is cache hygiene, not a fork-safety guarantee: torch itself does not
support re-initialising CUDA in a forked child, so a process that loads a model
and then forks is outside what this package supports either way.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from trw_memory.embeddings.interface import EmbeddingProvider

logger = structlog.get_logger(__name__)

__all__ = ["BUILD_WAIT_TIMEOUT_SECONDS", "cached_local_embedder", "reset_provider_cache"]

#: Environment variables that select what may be loaded and from where. The
#: offline pair is PRD-QUAL-110-FR04; the cache roots are what
#: ``embeddings/_hf_cache.py`` resolves dynamically, so a caller that repoints
#: HF_HOME is asking for a different snapshot, not the cached one.
_POLICY_ENV_VARS = (
    "TRW_OFFLINE",
    "HF_HUB_OFFLINE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
)

#: How long a caller waits for someone else's in-flight load before giving up
#: and answering "no embedder right now". A first load of all-MiniLM-L6-v2 is
#: seconds, not minutes; this is the bound that keeps a pathological build from
#: holding every worker in a server process.
BUILD_WAIT_TIMEOUT_SECONDS = 120.0

#: Guards the bookkeeping ONLY: the cache dict, the pending map and the
#: fingerprint. Never held across a factory call.
_CACHE_LOCK = threading.Lock()

#: (model_name, dim) -> provider. Only populated with providers that loaded.
_PROVIDER_CACHE: dict[tuple[str, int], EmbeddingProvider] = {}

#: The fingerprint every cached provider was built under. A mismatch empties
#: the cache; it never selects a second entry.
_CACHE_FINGERPRINT: tuple[object, ...] | None = None


class _Pending:
    """One in-flight construction: who owns it, and what it is building FOR.

    The identity matters as much as the result. A provider built under one
    policy generation must not be handed to a caller that arrived under a
    different one, so the waiter compares before accepting.
    """

    __slots__ = ("error", "event", "generation", "owner", "provider", "stamp")

    def __init__(self, stamp: tuple[object, ...], generation: int) -> None:
        self.event = threading.Event()
        self.provider: EmbeddingProvider | None = None
        self.error: BaseException | None = None
        self.owner = threading.get_ident()
        self.stamp = stamp
        self.generation = generation


#: (model_name, dim) -> the construction another caller is already doing.
_PENDING: dict[tuple[str, int], _Pending] = {}

#: Bumped by every reset and every policy change. A pending build started under
#: an older generation is not accepted by a caller in a newer one, which is what
#: closes the ABA hole: a reset followed by the same fingerprint returning is a
#: NEW generation, not the old one.
_GENERATION = 0

#: Per-thread marker: this thread is inside a factory call. A thread that is
#: itself building must never block on another thread's build, or two builders
#: needing each other's model deadlock (A builds X and wants Y; B builds Y and
#: wants X). It builds its own unshared instance instead.
_LOCAL = threading.local()


def _policy_fingerprint() -> tuple[object, ...]:
    """Collapse the security, offline, snapshot-source and process inputs.

    Built from ``MemoryConfig`` plus the environment on every call. That is a
    config construction per call, not a model load, and it is what makes the
    cache honour a policy that changed since the last one.
    """
    from trw_memory.models.config import MemoryConfig

    config = MemoryConfig()
    return (
        os.getpid(),
        bool(config.local_only),
        bool(config.embedding_trust_remote_code),
        tuple(os.environ.get(name, "") for name in _POLICY_ENV_VARS),
    )


def reset_provider_cache() -> None:
    """Drop every cached provider. Test seam and policy-change path.

    In-flight constructions are left to finish; their results are neither
    published nor handed to a waiter, because the generation they were started
    under is gone.
    """
    global _CACHE_FINGERPRINT, _GENERATION
    with _CACHE_LOCK:
        _PROVIDER_CACHE.clear()
        _PENDING.clear()
        _CACHE_FINGERPRINT = None
        _GENERATION += 1


def cached_local_embedder(
    key: tuple[str, int],
    factory: Callable[[], EmbeddingProvider | None],
) -> EmbeddingProvider | None:
    """Return the cached provider for *key*, or build one exactly once.

    Args:
        key: ``(model_name, dim)`` -- the provider's own identity.
        factory: Builds and probes a provider. Returning ``None`` means "not
            available right now"; raising means the caller must see the
            exception (a security refusal is a raise, by design).

    Returns:
        The provider, or ``None`` when no provider could be produced. A
        ``None`` answer is never cached, and neither is a raise.
    """
    global _CACHE_FINGERPRINT, _GENERATION
    while True:
        # Re-read the policy on every pass: a retry after a failed or discarded
        # build must decide under the CURRENT policy, not the one it started in.
        fingerprint = _policy_fingerprint()
        with _CACHE_LOCK:
            if fingerprint != _CACHE_FINGERPRINT:
                if _PROVIDER_CACHE:
                    logger.debug("embedder_cache_invalidated", reason="policy_or_process_changed")
                _PROVIDER_CACHE.clear()
                _CACHE_FINGERPRINT = fingerprint
                _GENERATION += 1
            generation = _GENERATION
            cached = _PROVIDER_CACHE.get(key)
            if cached is not None:
                return cached
            pending = _PENDING.get(key)
            if pending is not None and pending.owner == threading.get_ident():
                # Re-entered from inside our own factory. Waiting here would be
                # waiting on ourselves, so answer "nothing loaded yet" instead.
                logger.debug("embedder_cache_reentrant_call", model=key[0])
                return None
            if pending is not None and getattr(_LOCAL, "building", False):
                # We are a builder ourselves. Blocking on another builder is how
                # two of them deadlock, so build an unshared instance instead.
                pending = None
                share = False
            else:
                share = True
            mine = pending is None
            if mine and share:
                pending = _Pending(fingerprint, generation)
                _PENDING[key] = pending
        if not mine and pending is not None:
            # Someone else is building it: wait for their result rather than
            # starting a second load of the same model. The wait is BOUNDED --
            # a builder that never finishes must not be able to park every
            # server worker behind it, which on a loaded machine is the
            # difference between a slow request and a daemon that answers
            # nothing.
            if not pending.event.wait(timeout=BUILD_WAIT_TIMEOUT_SECONDS):
                logger.warning("embedder_build_wait_timed_out", model=key[0], seconds=BUILD_WAIT_TIMEOUT_SECONDS)
                return None
            if pending.error is None and pending.generation == generation and pending.stamp == fingerprint:
                return pending.provider
            # Either the builder failed, or it was building for a policy that is
            # no longer ours. Nothing usable was published, so go round again
            # and decide under the current policy.
            continue
        if not share:
            # Unshared build: no holder to publish to, and nothing cached, so a
            # nested construction cannot poison the shared entry either.
            return factory()
        if pending is None:  # pragma: no cover - unreachable; ``mine`` implies a holder
            return factory()
        return _build_and_publish(key, factory, pending, fingerprint, generation)


def _build_and_publish(
    key: tuple[str, int],
    factory: Callable[[], EmbeddingProvider | None],
    pending: _Pending,
    fingerprint: tuple[object, ...],
    generation: int,
) -> EmbeddingProvider | None:
    """Run *factory* outside the lock and publish the outcome to waiters."""
    _LOCAL.building = True
    try:
        provider = factory()
    except BaseException as exc:
        pending.error = exc
        with _CACHE_LOCK:
            if _PENDING.get(key) is pending:
                del _PENDING[key]
        pending.event.set()
        raise
    finally:
        _LOCAL.building = False
    with _CACHE_LOCK:
        if _PENDING.get(key) is pending:
            del _PENDING[key]
        # A policy change or a reset during the load means this provider was
        # built for a generation that is no longer current: hand it to this
        # caller, but do not install it for the next one.
        if provider is not None and fingerprint == _CACHE_FINGERPRINT and generation == _GENERATION:
            _PROVIDER_CACHE[key] = provider
    pending.provider = provider
    pending.event.set()
    return provider
