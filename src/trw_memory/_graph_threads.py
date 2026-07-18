"""Background graph-update thread registry.

``schedule_graph_update`` (in :mod:`trw_memory.graph`) dispatches best-effort
graph enrichment onto daemon threads off the write critical path. This module
owns the registry of those in-flight threads plus the join primitive that tests
and process teardown use to wait for them to finish.

Extracted from ``graph.py`` so the mutable registry lives behind a small class
interface (one module-level singleton) instead of bare module globals — this
prevents thread state from leaking across the module surface. The back-compat
names ``_track_graph_thread`` / ``_untrack_graph_thread`` /
``wait_for_graph_updates`` (and the ``_BACKGROUND_GRAPH_THREADS`` /
``_BACKGROUND_GRAPH_THREADS_GUARD`` aliases) are re-exported from
``trw_memory.graph``.
"""

from __future__ import annotations

import threading
from time import monotonic


class _GraphThreadRegistry:
    """Thread-safe registry of in-flight background graph-update threads.

    Encapsulates the ``set`` of live threads plus the guard lock protecting it.
    ``track`` / ``untrack`` mutate the set in place so external aliases bound to
    :attr:`_threads` keep observing the live registry state.
    """

    def __init__(self) -> None:
        self._threads: set[threading.Thread] = set()
        self._guard = threading.Lock()

    def track(self, thread: threading.Thread) -> None:
        """Register *thread* as an in-flight background graph update."""
        with self._guard:
            self._threads.add(thread)

    def untrack(self, thread: threading.Thread) -> None:
        """Remove *thread* once it has finished (called from the worker finally)."""
        with self._guard:
            self._threads.discard(thread)

    def alive(self) -> list[threading.Thread]:
        """Snapshot the currently-alive registered threads under the guard."""
        with self._guard:
            return [thread for thread in self._threads if thread.is_alive()]

    def wait(self, timeout: float = 5.0) -> None:
        """Block until all registered threads finish or *timeout* elapses.

        Raises ``TimeoutError`` if live threads remain past the deadline.
        """
        deadline = monotonic() + timeout
        while True:
            threads = self.alive()
            if not threads:
                return
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for background graph updates")
            threads[0].join(min(0.05, remaining))


# Module-level singleton — one registry for the process, matching the previous
# module-global semantics.
_REGISTRY = _GraphThreadRegistry()


def _track_graph_thread(thread: threading.Thread) -> None:
    """Back-compat shim: register *thread* on the process registry."""
    _REGISTRY.track(thread)


def _untrack_graph_thread(thread: threading.Thread) -> None:
    """Back-compat shim: unregister *thread* from the process registry."""
    _REGISTRY.untrack(thread)


def wait_for_graph_updates(timeout: float = 5.0) -> None:
    """Block until scheduled graph-update threads finish or *timeout* elapses."""
    _REGISTRY.wait(timeout)
