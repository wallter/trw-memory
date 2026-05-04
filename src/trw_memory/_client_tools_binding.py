"""Tools-binding cluster — agent tool fns + auto_recall decorator.

Belongs to ``client.py``. Re-exported there for back-compat.

3 helpers covering the agent-framework integration surface:

- ``make_tool_functions`` — produce a dict of async tool functions
  (memory_store / memory_recall / memory_forget / memory_search) that
  close over a ``MemoryClient`` instance.
- ``register_tools`` — register the tool dict with an agent that
  exposes either ``register_tool(name, fn)`` or ``tool()`` decorator.
- ``auto_recall`` — decorator factory that injects recalled memories
  into the decorated function as a ``recalled_memories`` kwarg.

Logger lookup goes through the parent module so test patches on
``trw_memory.client.logger`` propagate.

Extracted as PRD-DIST-246 batch 108.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import structlog

from trw_memory.exceptions import (
    MemoryConnectionError,
    StorageError,
    ToolAlreadyRegisteredError,
)

if TYPE_CHECKING:
    from trw_memory.client import (
        AgentWithRegisterTool,
        AgentWithToolDecorator,
        ForgetResultDict,
        MemoryClient,
        MemoryResultDict,
        StoreResultDict,
        _ToolFn,
    )

logger = structlog.get_logger(__name__)


def _client_logger() -> Any:
    """Parent-module logger lookup so test patches on ``trw_memory.client.logger`` propagate."""
    from trw_memory import client as _c
    return _c.logger


def make_tool_functions(client: "MemoryClient") -> "dict[str, _ToolFn]":
    """Create the shared tool functions for agent registration."""

    async def memory_store(
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> "StoreResultDict":
        return await client.store(content, tags=tags, importance=importance)

    async def memory_recall(
        query: str,
        limit: int = 10,
        include_org_memories: bool = True,
        include_shared: bool = False,
        include_distilled: bool = True,
        distilled_weight: float | None = None,
        include_source_kinds: list[str] | None = None,
        exclude_source_kinds: list[str] | None = None,
        source_weights: dict[str, float] | None = None,
        exclude_expired: bool = True,
    ) -> "list[MemoryResultDict]":
        return await client.recall(
            query,
            limit=limit,
            include_org_memories=include_org_memories,
            include_shared=include_shared,
            include_distilled=include_distilled,
            distilled_weight=distilled_weight,
            include_source_kinds=include_source_kinds,
            exclude_source_kinds=exclude_source_kinds,
            source_weights=source_weights,
            exclude_expired=exclude_expired,
        )

    async def memory_forget(memory_id: str | None = None, actor: str | None = None) -> "ForgetResultDict":
        return await client.forget(memory_id, actor=actor)

    async def memory_search(
        tags: list[str] | None = None,
        min_importance: float = 0.0,
        limit: int = 50,
        actor: str | None = None,
        status: str | None = None,
    ) -> "list[MemoryResultDict]":
        return await client.search(
            tags=tags,
            min_importance=min_importance,
            limit=limit,
            actor=actor,
            status=status,
        )

    return {
        "memory_store": memory_store,
        "memory_recall": memory_recall,
        "memory_forget": memory_forget,
        "memory_search": memory_search,
    }


def register_tools(
    client: "MemoryClient",
    agent: "AgentWithRegisterTool | AgentWithToolDecorator",
) -> None:
    """Register memory tools with an agent framework."""
    if client._tools_registered:
        raise ToolAlreadyRegisteredError("register_tools() has already been called on this client")

    tools = client._make_tool_functions()

    register_fn = getattr(agent, "register_tool", None)
    tool_decorator = getattr(agent, "tool", None)

    if register_fn is not None and callable(register_fn):
        for name, fn in tools.items():
            register_fn(name, fn)
    elif tool_decorator is not None and callable(tool_decorator):
        dec = tool_decorator()
        for fn in tools.values():
            dec(fn)
    else:
        raise TypeError("Agent must have a 'register_tool()' method or 'tool()' decorator")

    client._tools_registered = True


def auto_recall(
    client: "MemoryClient",
    query_from: str,
    limit: int = 10,
    min_score: float = 0.0,
) -> Callable[[Callable[..., Coroutine[object, object, object]]], Callable[..., Coroutine[object, object, object]]]:
    """Decorator factory that injects recalled memories into the decorated function."""

    def decorator(
        fn: Callable[..., Coroutine[object, object, object]],
    ) -> Callable[..., Coroutine[object, object, object]]:
        sig = inspect.signature(fn)
        for name, param in sig.parameters.items():
            if (
                name == "recalled_memories"
                and param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                and param.default is inspect.Parameter.empty
            ):
                raise TypeError(
                    "Decorated function must not have 'recalled_memories' as a required positional parameter"
                )

        @functools.wraps(fn)
        async def wrapper(*args: object, **kwargs: object) -> object:
            memories: list[MemoryResultDict] = []
            try:
                query = kwargs.get(query_from, "")
                if query:
                    raw = await client.recall(str(query), limit=limit)
                    memories = [m for m in raw if float(m["score"]) >= min_score]
            except (OSError, ValueError, StorageError, MemoryConnectionError):
                _client_logger().debug("auto_recall_failed", op="auto_recall", outcome="failure", exc_info=True)
                memories = []

            kwargs["recalled_memories"] = memories
            return await fn(*args, **kwargs)

        return wrapper

    return decorator
