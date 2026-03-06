"""MemoryClient — Python SDK for trw-memory.

Provides a high-level async API for storing, recalling, searching, and
forgetting memories.  Supports local (SQLite/YAML) backends with stubs
for future MCP and REST transport modes.

Usage::

    async with MemoryClient(namespace="project:my-app") as client:
        await client.store("Pydantic v2 requires strict=True", tags=["pydantic"])
        results = await client.recall("pydantic validation")
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from trw_memory.exceptions import (
    MemoryConnectionError,
    MemoryNotFoundError,
    ToolAlreadyRegisteredError,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.interface import StorageBackend


def _make_id() -> str:
    """Generate a unique memory ID with M- prefix."""
    return f"M-{uuid.uuid4().hex[:8]}"


def _entry_to_result(entry: MemoryEntry, score: float = 0.0) -> dict[str, Any]:
    """Convert a MemoryEntry to a result dict."""
    return {
        "memory_id": entry.id,
        "content": entry.content,
        "detail": entry.detail,
        "tags": list(entry.tags),
        "importance": entry.importance,
        "score": score,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "namespace": entry.namespace,
    }


def _create_local_backend(config: MemoryConfig, namespace: str) -> StorageBackend:
    """Create a storage backend from config, isolated by namespace.

    Args:
        config: Memory configuration.
        namespace: Namespace for path isolation.

    Returns:
        Configured StorageBackend instance.
    """
    base = Path(config.storage_path)

    if config.storage_backend == "sqlite":
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        db_path = base / namespace.replace(":", "_") / config.sqlite_db_name
        return SQLiteBackend(db_path=db_path, dim=config.embedding_dim)

    # YAML fallback
    from trw_memory.storage.yaml_backend import YAMLBackend

    entries_dir = base / namespace.replace(":", "_") / "entries"
    return YAMLBackend(entries_dir=entries_dir)


class MemoryClient:
    """High-level async client for the trw-memory system.

    Args:
        namespace: Isolation scope (e.g. ``"project:my-app"``, ``"default"``).
        mode: Transport mode — ``"local"`` (SQLite/YAML), ``"mcp"`` (stdio),
            ``"rest"`` (HTTP), or ``"auto"`` (try local first).
        base_url: Base URL for REST mode.  Ignored for other modes.
        timeout: Timeout in seconds for remote operations.
    """

    def __init__(
        self,
        namespace: str,
        mode: Literal["local", "mcp", "rest", "auto"] = "auto",
        base_url: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        validate_namespace(namespace)
        self._namespace = namespace
        self._timeout = timeout
        self._base_url = base_url
        self._lock = asyncio.Lock()
        self._tools_registered = False
        self._backend: StorageBackend | None = None
        self._resolved_mode: str = ""

        if mode in ("mcp", "rest"):
            raise NotImplementedError("MCP/REST modes are not yet implemented")

        if mode in ("local", "auto"):
            try:
                config = MemoryConfig()
                self._backend = _create_local_backend(config, namespace)
                self._resolved_mode = "local"
            except (OSError, ValueError, ImportError) as exc:
                if mode == "local":
                    raise MemoryConnectionError(
                        f"Failed to create local backend: {exc}"
                    ) from exc

        if mode == "auto" and self._backend is None:
            raise MemoryConnectionError(
                "No connection mode available. Tried: local."
            )

    @property
    def resolved_mode(self) -> str:
        """The actual transport mode in use."""
        return self._resolved_mode

    @property
    def namespace(self) -> str:
        """The namespace this client operates in."""
        return self._namespace

    def _get_backend(self) -> StorageBackend:
        """Return the active backend, raising if the client is closed."""
        if self._backend is None:
            raise MemoryConnectionError("Client is closed or has no backend")
        return self._backend

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def store(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        detail: str = "",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Store a new memory entry.

        Args:
            content: Core knowledge statement (must not be empty).
            tags: Categorisation tags.
            importance: Importance score in ``[0.0, 1.0]``.
            detail: Extended explanation.
            metadata: Arbitrary key-value pairs.

        Returns:
            Dict with ``memory_id``, ``namespace``, ``status``, ``timestamp``.

        Raises:
            ValueError: If *content* is empty or *importance* out of range.
        """
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        if not 0.0 <= importance <= 1.0:
            raise ValueError(
                f"importance must be in [0.0, 1.0], got {importance}"
            )

        memory_id = _make_id()
        now = datetime.now(timezone.utc)

        entry = MemoryEntry(
            id=memory_id,
            content=content.strip(),
            detail=detail,
            tags=tags or [],
            importance=importance,
            namespace=self._namespace,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            source="agent",
        )

        async with self._lock:
            self._get_backend().store(entry)

        return {
            "memory_id": memory_id,
            "namespace": self._namespace,
            "status": "stored",
            "timestamp": now.isoformat(),
        }

    async def recall(
        self,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search memories by keyword query.

        Args:
            query: Free-text search term.
            limit: Maximum number of results (must be >= 1).
            tags: If provided, results must contain all listed tags.
            min_score: Minimum score threshold for results.

        Returns:
            List of result dicts ordered by score descending.

        Raises:
            ValueError: If *limit* < 1.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        async with self._lock:
            entries = self._get_backend().search(
                query,
                top_k=limit,
                tags=tags,
                namespace=self._namespace,
            )

        # Score by importance (keyword match already filtered by backend)
        results: list[dict[str, Any]] = [
            _entry_to_result(entry, score=entry.importance)
            for entry in entries
            if entry.importance >= min_score
        ]

        results.sort(key=lambda r: float(r["score"]), reverse=True)
        return results[:limit]

    async def forget(self, memory_id: str) -> dict[str, str]:
        """Delete a memory entry.

        Args:
            memory_id: ID of the memory to delete.

        Returns:
            Dict with ``memory_id``, ``status``, ``namespace``.

        Raises:
            MemoryNotFoundError: If *memory_id* does not exist or belongs
                to a different namespace.
        """
        async with self._lock:
            backend = self._get_backend()
            existing = backend.get(memory_id)
            if existing is None:
                raise MemoryNotFoundError(
                    f"Memory entry {memory_id!r} not found"
                )
            if existing.namespace != self._namespace:
                raise MemoryNotFoundError(
                    f"Memory entry {memory_id!r} not found in namespace {self._namespace!r}"
                )
            backend.delete(memory_id)

        return {
            "memory_id": memory_id,
            "status": "deleted",
            "namespace": self._namespace,
        }

    async def search(
        self,
        tags: list[str] | None = None,
        min_importance: float = 0.0,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search memories with filters.

        Args:
            tags: If provided, results must contain all listed tags.
            min_importance: Lower bound on importance (inclusive, ``[0.0, 1.0]``).
            since: If provided, only return entries created after this time.
            limit: Maximum number of results (must be >= 1).

        Returns:
            List of result dicts ordered by importance descending.

        Raises:
            ValueError: If *limit* < 1 or *min_importance* out of range.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if not 0.0 <= min_importance <= 1.0:
            raise ValueError(
                f"min_importance must be in [0.0, 1.0], got {min_importance}"
            )

        async with self._lock:
            entries = self._get_backend().list_entries(
                namespace=self._namespace,
                limit=limit * 5,  # over-fetch to allow for post-filtering
            )

        # Post-filter
        tag_set = set(tags) if tags else None
        results: list[dict[str, Any]] = []
        for entry in entries:
            if entry.importance < min_importance:
                continue
            if tag_set and not tag_set.issubset(set(entry.tags)):
                continue
            if since is not None and entry.created_at < since:
                continue
            results.append(_entry_to_result(entry, score=entry.importance))

        results.sort(key=lambda r: float(r["score"]), reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Tool registration (FR09)
    # ------------------------------------------------------------------

    def _make_tool_functions(self) -> dict[str, Callable[..., Any]]:
        """Create the shared tool functions for agent registration."""
        client = self

        async def memory_store(
            content: str,
            tags: list[str] | None = None,
            importance: float = 0.5,
        ) -> dict[str, str]:
            return await client.store(content, tags=tags, importance=importance)

        async def memory_recall(
            query: str, limit: int = 10
        ) -> list[dict[str, Any]]:
            return await client.recall(query, limit=limit)

        async def memory_forget(memory_id: str) -> dict[str, str]:
            return await client.forget(memory_id)

        async def memory_search(
            tags: list[str] | None = None,
            min_importance: float = 0.0,
        ) -> list[dict[str, Any]]:
            return await client.search(tags=tags, min_importance=min_importance)

        return {
            "memory_store": memory_store,
            "memory_recall": memory_recall,
            "memory_forget": memory_forget,
            "memory_search": memory_search,
        }

    def register_tools(self, agent: Any) -> None:
        """Register memory tools with an agent framework.

        Attempts to register ``memory_store``, ``memory_recall``,
        ``memory_forget``, and ``memory_search`` tools on the agent object.

        Args:
            agent: An agent-like object with a ``tool()`` decorator method
                or ``register_tool()`` method.

        Raises:
            ToolAlreadyRegisteredError: If called more than once.
            TypeError: If *agent* does not have a compatible registration API.
        """
        if self._tools_registered:
            raise ToolAlreadyRegisteredError(
                "register_tools() has already been called on this client"
            )

        tools = self._make_tool_functions()

        # Detect registration API
        register_fn = getattr(agent, "register_tool", None)
        tool_decorator = getattr(agent, "tool", None)

        if register_fn is not None and callable(register_fn):
            for name, fn in tools.items():
                register_fn(name, fn)
        elif tool_decorator is not None and callable(tool_decorator):
            dec: Any = tool_decorator()
            for fn in tools.values():
                dec(fn)
        else:
            raise TypeError(
                "Agent must have a 'register_tool()' method or 'tool()' decorator"
            )

        self._tools_registered = True

    # ------------------------------------------------------------------
    # auto_recall decorator (FR06)
    # ------------------------------------------------------------------

    def auto_recall(
        self,
        query_from: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> Callable[..., Any]:
        """Decorator that injects recalled memories into a function.

        Before calling the decorated function, extracts a query string from
        the function's keyword argument named *query_from*, performs a recall,
        and injects the results as a ``recalled_memories`` keyword argument.

        Fail-open: if the backend is unreachable or the query_from key is
        absent, an empty list is injected.

        Args:
            query_from: Name of the kwarg to extract the query from.
            limit: Maximum number of recalled entries.
            min_score: Minimum score threshold.

        Returns:
            A decorator function.

        Raises:
            TypeError: If the decorated function has ``recalled_memories``
                as a positional parameter.
        """
        client = self

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            # Check if recalled_memories is a positional arg
            sig = inspect.signature(fn)
            for name, param in sig.parameters.items():
                if (
                    name == "recalled_memories"
                    and param.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    and param.default is inspect.Parameter.empty
                ):
                    raise TypeError(
                        "Decorated function must not have 'recalled_memories' "
                        "as a required positional parameter"
                    )

            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                memories: list[dict[str, Any]] = []
                try:
                    query = kwargs.get(query_from, "")
                    if query:
                        raw = await client.recall(str(query), limit=limit)
                        memories = [
                            m for m in raw if float(m["score"]) >= min_score
                        ]
                except Exception:  # broad catch: fail-open recall decorator
                    memories = []

                kwargs["recalled_memories"] = memories
                return await fn(*args, **kwargs)

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MemoryClient:
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the async context manager, closing the backend."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying backend and release resources."""
        if self._backend is not None:
            self._backend.close()
            self._backend = None
