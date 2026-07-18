"""TypedDict shapes + agent protocols for ``MemoryClient``.

Belongs to ``client.py``. Re-exported there for back-compat.

3 TypedDicts + 2 Protocols + 1 type alias:

- ``MemoryResultDict`` — shape of a recall/search result.
- ``StoreResultDict`` — shape of a store(...) return value.
- ``ForgetResultDict`` — shape of a forget(...) return value.
- ``AgentWithRegisterTool`` — agents exposing ``register_tool``.
- ``AgentWithToolDecorator`` — agents exposing ``tool()`` decorator.
- ``_ToolFn`` — agent tool function type alias.

Extracted as PRD-DIST-246 batch 113.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Protocol, runtime_checkable

from typing_extensions import NotRequired, TypedDict


class MemoryResultDict(TypedDict):
    """Shape of a single result dict returned by recall/search."""

    memory_id: str
    content: str
    detail: str
    tags: list[str]
    importance: float
    score: float
    created_at: str
    updated_at: str
    namespace: str
    source: str
    last_accessed_at: NotRequired[str]
    q_value: NotRequired[float]
    q_observations: NotRequired[int]
    recurrence: NotRequired[int]
    access_count: NotRequired[int]
    expires: NotRequired[str]
    metadata: NotRequired[dict[str, str]]
    anomaly_dimension: NotRequired[str]
    z_score: NotRequired[float]
    _relevance_hint: NotRequired[float]


class StoreResultDict(TypedDict):
    """Shape of the dict returned by MemoryClient.store()."""

    memory_id: str
    namespace: str
    status: str
    timestamp: str
    quarantined: NotRequired[bool]
    stored: NotRequired[bool]
    anomaly_dimension: NotRequired[str]
    z_score: NotRequired[float]


class ForgetResultDict(TypedDict):
    """Shape of the dict returned by MemoryClient.forget()."""

    memory_id: str
    status: str
    namespace: str
    entries_deleted: NotRequired[int]


@runtime_checkable
class AgentWithRegisterTool(Protocol):
    """Protocol for agent objects that expose a ``register_tool`` method."""

    def register_tool(self, name: str, fn: Callable[..., Coroutine[object, object, object]]) -> None: ...


@runtime_checkable
class AgentWithToolDecorator(Protocol):
    """Protocol for agent objects that expose a ``tool()`` decorator factory."""

    def tool(self) -> Callable[[Callable[..., Coroutine[object, object, object]]], None]: ...


_ToolFn = Callable[..., Coroutine[object, object, object]]
