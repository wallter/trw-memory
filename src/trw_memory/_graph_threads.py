"""Background graph-update thread registry.

``schedule_graph_update`` (in :mod:`trw_memory.graph`) dispatches best-effort
graph enrichment onto daemon threads off the write critical path. This module
owns the registry of those in-flight threads plus the join primitive that tests
and process teardown use to wait for them to finish.

Extracted from ``graph.py`` so the mutable registry lives behind a small class
interface (one module-level singleton) instead of bare module globals — this
prevents thread state from leaking across the module surface.
``wait_for_graph_updates`` is re-exported from ``trw_memory.graph``.
"""

from __future__ import annotations

import os
import threading
from time import monotonic, sleep


class _GraphThreadRegistry:
    """Thread-safe registry of in-flight background graph-update threads.

    Encapsulates the ``set`` of live threads plus the guard lock protecting it.
    """

    def __init__(self) -> None:
        self._threads: set[threading.Thread] = set()
        self._owners: dict[threading.Thread, object | None] = {}
        self._guard = threading.Lock()

    def track(self, thread: threading.Thread, owner: object | None = None) -> None:
        """Register *thread* as an in-flight background graph update."""
        with self._guard:
            self._threads.add(thread)
            self._owners[thread] = owner

    def after_fork_in_child(self) -> None:
        """A forked child has none of the parent's threads, and may inherit ``_guard`` held by one of them."""
        self._threads = set()
        self._owners = {}
        self._guard = threading.Lock()

    def untrack(self, thread: threading.Thread) -> None:
        """Remove *thread* once it has finished (called from the worker finally)."""
        with self._guard:
            self._threads.discard(thread)
            self._owners.pop(thread, None)

    def wait(self, timeout: float = 5.0, *, owner: object | None = None) -> None:
        """Block until matching registered threads finish or *timeout* elapses.

        An explicit owner is compared by identity, never by shared storage path.
        Raises ``TimeoutError`` if live threads remain past the deadline.
        """
        deadline = monotonic() + timeout
        while True:
            with self._guard:
                threads = [
                    thread
                    for thread in self._threads
                    if (owner is None or self._owners.get(thread) is owner)
                    and (thread.is_alive() or thread.ident is None)
                ]
            if not threads:
                return
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for background graph updates")
            if threads[0].ident is None:
                # Registered before start(): never mistake pending work for
                # completed work, or attempt to join an unstarted thread.
                sleep(min(0.01, remaining))
            else:
                threads[0].join(min(0.05, remaining))


# Module-level singleton — one registry for the process, matching the previous
# module-global semantics.
GRAPH_THREADS = _GraphThreadRegistry()
os.register_at_fork(after_in_child=GRAPH_THREADS.after_fork_in_child)


#: How long a drain waits for background graph work: close()'s drain, and interpreter exit's.
GRAPH_DRAIN_SECONDS = 5.0


def wait_for_graph_updates(timeout: float = GRAPH_DRAIN_SECONDS, *, owner: object | None = None) -> None:
    """Drain one backend owner's workers, or all workers when owner is omitted."""
    GRAPH_THREADS.wait(timeout, owner=owner)
