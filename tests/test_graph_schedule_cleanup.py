"""Graph scheduling cleanup regressions."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from trw_memory import graph
from trw_memory._graph_threads import GRAPH_THREADS
from trw_memory.models.config import MemoryConfig

from .conftest import make_entry


def test_schedule_failure_does_not_retain_unstarted_thread() -> None:
    entry = make_entry(entry_id="M-schedule")
    with (
        patch.object(graph, "_derive_graph_config", return_value=MemoryConfig()),
        patch.object(threading.Thread, "start", side_effect=RuntimeError("thread unavailable")),
    ):
        assert graph.schedule_graph_update(entry, MagicMock()) is False

    assert all(thread.name != "trw-memory-graph-M-schedule" for thread in GRAPH_THREADS._threads)
