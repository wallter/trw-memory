"""Remote sync -- publish, fetch, conflict resolution, SSE subscription.

Exports resolve on first access (PEP 562), so importing one submodule -- the daemon
client's ``_remote_common.build_platform_headers`` on every hook read -- does not
load the rest (``delta`` alone cost ~0.2 s; PRD-CORE-333 S3b). Nothing here registers
anything at import time, so deferring the imports changes no behaviour.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trw_memory.sync._remote_admission import AdmissionOutcome, Gate, store_gate
    from trw_memory.sync.conflict import (
        compare_clocks,
        increment_clock,
        init_clock,
        merge_clocks,
        resolve_conflict,
    )
    from trw_memory.sync.delta import DeltaTracker
    from trw_memory.sync.remote import SharedFetchResult, fetch_shared_memories
    from trw_memory.sync.retry_queue import RetryQueue
    from trw_memory.sync.subscriber import SSESubscriber

#: Each export and the submodule that defines it.
_EXPORTS: dict[str, str] = {
    "AdmissionOutcome": "_remote_admission",
    "Gate": "_remote_admission",
    "store_gate": "_remote_admission",
    "compare_clocks": "conflict",
    "increment_clock": "conflict",
    "init_clock": "conflict",
    "merge_clocks": "conflict",
    "resolve_conflict": "conflict",
    "DeltaTracker": "delta",
    "SharedFetchResult": "remote",
    "fetch_shared_memories": "remote",
    "RetryQueue": "retry_queue",
    "SSESubscriber": "subscriber",
}

__all__ = [
    "AdmissionOutcome",
    "DeltaTracker",
    "Gate",
    "RetryQueue",
    "SSESubscriber",
    "SharedFetchResult",
    "compare_clocks",
    "fetch_shared_memories",
    "increment_clock",
    "init_clock",
    "merge_clocks",
    "resolve_conflict",
    "store_gate",
]


def __getattr__(name: str) -> object:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # resolve once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
