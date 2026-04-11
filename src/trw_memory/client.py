"""MemoryClient — Python SDK for trw-memory.

Provides a high-level async API for storing, recalling, searching, and
forgetting memories.  Supports local (SQLite/YAML) backends with a stub
for future MCP transport mode.

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
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, TypedDict, runtime_checkable

import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.exceptions import (
    MemoryConnectionError,
    MemoryNotFoundError,
    StorageError,
    ToolAlreadyRegisteredError,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

__all__ = [
    "ForgetResultDict",
    "MemoryClient",
    "MemoryResultDict",
    "StoreResultDict",
]


# Fallback recall scoring (used when hybrid retrieval pipeline is unavailable).
# Blends term-frequency relevance with stored importance.
_FALLBACK_TF_WEIGHT: float = 0.7
_FALLBACK_IMPORTANCE_WEIGHT: float = 0.3
_FALLBACK_TF_SCALE: float = 10.0  # amplify raw TF ratio to [0, 1] range

# Re-export old names for backward compatibility (internal only).
_TF_WEIGHT = _FALLBACK_TF_WEIGHT
_IMPORTANCE_WEIGHT = _FALLBACK_IMPORTANCE_WEIGHT
_TF_SCALE = _FALLBACK_TF_SCALE


def _make_id() -> str:
    """Generate a unique memory ID with ``M-`` prefix and 16 hex characters.

    Uses 16 hex characters (64 bits of entropy) from a UUID4 to minimise
    collision probability.  At 10k entries the birthday-paradox collision
    chance is ~2.7e-12 (vs ~1.2e-5 with the previous 8-char / 32-bit scheme).

    Returns:
        A string matching ``M-[0-9a-f]{16}``, e.g. ``"M-a1b2c3d4e5f6a7b8"``.
    """
    return f"M-{uuid.uuid4().hex[:16]}"


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


class StoreResultDict(TypedDict):
    """Shape of the dict returned by MemoryClient.store()."""

    memory_id: str
    namespace: str
    status: str
    timestamp: str


class ForgetResultDict(TypedDict):
    """Shape of the dict returned by MemoryClient.forget()."""

    memory_id: str
    status: str
    namespace: str


@runtime_checkable
class AgentWithRegisterTool(Protocol):
    """Protocol for agent objects that expose a ``register_tool`` method."""

    def register_tool(self, name: str, fn: Callable[..., Coroutine[object, object, object]]) -> None: ...


@runtime_checkable
class AgentWithToolDecorator(Protocol):
    """Protocol for agent objects that expose a ``tool()`` decorator factory."""

    def tool(self) -> Callable[[Callable[..., Coroutine[object, object, object]]], None]: ...


def _entry_to_result(entry: MemoryEntry, score: float = 0.0) -> MemoryResultDict:
    """Convert a MemoryEntry to a result dict."""
    return MemoryResultDict(
        memory_id=entry.id,
        content=entry.content,
        detail=entry.detail,
        tags=list(entry.tags),
        importance=entry.importance,
        score=score,
        created_at=entry.created_at.isoformat(),
        updated_at=entry.updated_at.isoformat(),
        namespace=entry.namespace,
    )


def _create_local_backend(config: MemoryConfig, namespace: str) -> StorageBackend:
    """Create a storage backend from config, isolated by namespace.

    Args:
        config: Memory configuration.
        namespace: Namespace for path isolation.

    Returns:
        Configured StorageBackend instance.
    """
    from trw_memory.integrations._backend import create_backend_from_config

    return create_backend_from_config(config, namespace)


# Type alias for the async tool functions produced by _make_tool_functions.
_ToolFn = Callable[..., Coroutine[object, object, object]]


class MemoryClient:
    """High-level async client for the trw-memory system.

    Args:
        namespace: Isolation scope (e.g. ``"project:my-app"``, ``"default"``).
        mode: Transport mode — ``"local"`` (SQLite/YAML), ``"mcp"`` (stdio),
            or ``"auto"`` (try local first).
        timeout: Timeout in seconds for remote operations.
    """

    def __init__(
        self,
        namespace: str,
        mode: Literal["local", "mcp", "auto"] = "auto",
        timeout: float = 5.0,
    ) -> None:
        """Initialise a MemoryClient with namespace isolation and mode selection.

        Mode selection logic:

        - ``"local"`` — create a SQLite or YAML backend directly (controlled
          by ``MEMORY_STORAGE_BACKEND`` env var).  Raises
          :class:`MemoryConnectionError` if backend creation fails.
        - ``"mcp"`` — reserved for future MCP stdio transport (currently
          raises :class:`NotImplementedError`).
        - ``"auto"`` (default) — attempt ``"local"`` first; if that fails,
          raise :class:`MemoryConnectionError` (MCP fallback not yet available).

        Args:
            namespace: Isolation scope (e.g. ``"project:my-app"``, ``"default"``).
                Must pass :func:`~trw_memory.namespaces.validation.validate_namespace`.
            mode: Transport mode — ``"local"``, ``"mcp"``, or ``"auto"``.
            timeout: Timeout in seconds for future remote operations.

        Raises:
            ValueError: If *namespace* fails validation.
            NotImplementedError: If *mode* is ``"mcp"``.
            MemoryConnectionError: If no backend can be established.
        """
        validate_namespace(namespace)
        self._namespace = namespace
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._tools_registered = False
        self._backend: StorageBackend | None = None
        self._resolved_mode: str = ""

        if mode == "mcp":
            raise NotImplementedError("MCP mode is not yet implemented")

        if mode in ("local", "auto"):
            try:
                config = MemoryConfig()
                self._backend = _create_local_backend(config, namespace)
                self._resolved_mode = "local"
                logger.debug(
                    "client_initialized",
                    op="init",
                    namespace=namespace,
                    mode=self._resolved_mode,
                    backend=config.storage_backend,
                )
            except (OSError, ValueError, ImportError) as exc:
                if mode == "local":
                    raise MemoryConnectionError(f"Failed to create local backend: {exc}") from exc

        if mode == "auto" and self._backend is None:
            raise MemoryConnectionError("No connection mode available. Tried: local.")

    def __repr__(self) -> str:
        return f"MemoryClient(namespace={self._namespace!r}, mode={self._resolved_mode!r})"

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
    ) -> StoreResultDict:
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
            raise ValueError(f"importance must be in [0.0, 1.0], got {importance}")

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

        logger.debug(
            "memory_stored",
            op="store",
            outcome="success",
            memory_id=memory_id,
            namespace=self._namespace,
            content_len=len(content),
            tag_count=len(tags or []),
            importance=importance,
        )
        return StoreResultDict(
            memory_id=memory_id,
            namespace=self._namespace,
            status="stored",
            timestamp=now.isoformat(),
        )

    async def recall(
        self,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        min_score: float = 0.0,
        *,
        token_budget: int | None = None,
    ) -> list[MemoryResultDict]:
        """Search memories by keyword query using hybrid retrieval.

        Uses a two-tier strategy:

        1. **Hybrid search** (preferred): Fetches all namespace entries and runs
           them through the retrieval pipeline (``hybrid_search``) which combines
           BM25 sparse retrieval with dense vector similarity via Reciprocal Rank
           Fusion (RRF).  This produces substantially better ranking than simple
           keyword matching.

        2. **Fallback TF scoring**: When the hybrid pipeline is unavailable
           (missing optional deps, import errors, or empty results), falls back
           to the original LIKE-based ``backend.search()`` with a blended
           term-frequency + importance score.

        Both paths apply tag filtering, min_score thresholds, limit capping,
        and optional token budget fitting.

        Args:
            query: Free-text search term.
            limit: Maximum number of results (must be >= 1).
            tags: If provided, results must contain all listed tags.
            min_score: Minimum score threshold for results.
            token_budget: If provided, truncate results to fit within this
                token budget.  Must be a positive integer.  When the budget
                is too small for all results, at least one result is always
                returned (minimum-one guarantee).

        Returns:
            List of result dicts ordered by score descending.

        Raises:
            ValueError: If *limit* < 1 or *token_budget* <= 0.
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if token_budget is not None and token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")

        # --- Tier 1: Hybrid retrieval pipeline (BM25 + dense + RRF) ----------
        hybrid_results = await self._try_hybrid_recall(query, limit, tags)
        if hybrid_results is not None:
            # Apply min_score filter and limit
            filtered = [r for r in hybrid_results if r["score"] >= min_score]
            final = filtered[:limit]
            final = self._apply_budget(final, token_budget)
            logger.debug(
                "memory_recalled",
                op="recall",
                outcome="success",
                query=query[:80],
                namespace=self._namespace,
                result_count=len(final),
                search_path="hybrid",
            )
            return final

        # --- Tier 2: Fallback LIKE + TF scoring ------------------------------
        results = await self._fallback_recall(query, limit, tags, min_score)
        return self._apply_budget(results, token_budget)

    @staticmethod
    def _apply_budget(
        results: list[MemoryResultDict],
        token_budget: int | None,
    ) -> list[MemoryResultDict]:
        """Apply token budget filtering to recall results.

        When *token_budget* is ``None``, returns *results* unchanged.

        Args:
            results: Ordered list of result dicts.
            token_budget: Maximum token budget, or ``None`` to skip.

        Returns:
            Filtered list of results that fit within the budget.
        """
        if token_budget is None or not results:
            return results

        from trw_memory.retrieval.token_budget import apply_token_budget

        # MemoryResultDict is a TypedDict — cast to dict[str, object] for the
        # budget function, then cast back.  The underlying dicts are the same
        # objects so no copy overhead.
        raw: list[dict[str, object]] = list(results)  # type: ignore[arg-type]
        filtered, _used, _truncated = apply_token_budget(raw, token_budget)
        return filtered  # type: ignore[return-value]

    async def _try_hybrid_recall(
        self,
        query: str,
        limit: int,
        tags: list[str] | None,
    ) -> list[MemoryResultDict] | None:
        """Attempt hybrid retrieval; return None to signal fallback.

        Fetches all entries for the namespace, runs them through
        ``hybrid_search`` (BM25 + optional dense vectors + RRF fusion),
        applies tag filtering, and converts to result dicts with
        positional scoring.

        Returns:
            A list of result dicts on success, or ``None`` when the hybrid
            pipeline is unavailable or produces no candidates.
        """
        try:
            from trw_memory.retrieval.pipeline import hybrid_search
        except ImportError:
            return None

        # Fetch candidate entries under lock (consistent with forget/store pattern)
        async with self._lock:
            backend = self._get_backend()
            all_entries = backend.list_entries(
                namespace=self._namespace,
                limit=limit * 5,
            )

        if not all_entries:
            return None

        # Optionally obtain an embedding provider for dense retrieval
        embedder = self._get_embedder()

        try:
            ranked = hybrid_search(
                query=query,
                entries=all_entries,
                embedder=embedder,
                top_k=limit * 3,
            )
        except Exception:
            logger.debug(
                "hybrid_search_failed",
                op="recall",
                outcome="failure",
                exc_info=True,
            )
            return None

        if not ranked:
            return None

        # Apply tag filter
        if tags:
            tag_set = set(tags)
            ranked = [e for e in ranked if tag_set.issubset(set(e.tags))]

        # Convert to result dicts with RRF-style positional scoring
        results: list[MemoryResultDict] = []
        for rank, entry in enumerate(ranked):
            score = round(1.0 / (1 + rank), 4)
            results.append(_entry_to_result(entry, score=score))

        return results

    async def _fallback_recall(
        self,
        query: str,
        limit: int,
        tags: list[str] | None,
        min_score: float,
    ) -> list[MemoryResultDict]:
        """Original LIKE-based search with TF + importance scoring.

        Uses ``backend.search()`` for keyword matching and blends a
        term-frequency relevance score (weight ``_FALLBACK_TF_WEIGHT``)
        with the entry's stored importance (weight
        ``_FALLBACK_IMPORTANCE_WEIGHT``).
        """
        async with self._lock:
            entries = self._get_backend().search(
                query,
                top_k=limit * 3,
                tags=tags,
                namespace=self._namespace,
            )

        query_terms = set(query.lower().split())
        results: list[MemoryResultDict] = []
        for entry in entries:
            if not query_terms:
                tf_score = entry.importance
            else:
                text_tokens = (
                    f"{entry.content} {entry.detail} {' '.join(entry.tags)}"
                    .lower()
                    .split()
                )
                matches = sum(1 for t in text_tokens if t in query_terms)
                tf_score = (
                    min(1.0, matches / max(len(text_tokens), 1) * _FALLBACK_TF_SCALE)
                    * _FALLBACK_TF_WEIGHT
                    + entry.importance * _FALLBACK_IMPORTANCE_WEIGHT
                )
            if tf_score >= min_score:
                results.append(_entry_to_result(entry, score=round(tf_score, 4)))

        results.sort(key=lambda r: float(r["score"]), reverse=True)
        final = results[:limit]
        logger.debug(
            "memory_recalled",
            op="recall",
            outcome="success",
            query=query[:80],
            namespace=self._namespace,
            result_count=len(final),
            search_path="fallback",
        )
        return final

    @staticmethod
    def _get_embedder() -> EmbeddingProvider | None:
        """Try to obtain a local embedding provider; return None on failure.

        This is a best-effort helper: when sentence-transformers is not
        installed or the provider reports itself as unavailable, dense
        retrieval is silently skipped and the hybrid pipeline degrades
        to BM25-only mode.
        """
        try:
            from trw_memory.embeddings.local import LocalEmbeddingProvider

            provider = LocalEmbeddingProvider()
            if provider.available():
                return provider
        except Exception:
            logger.debug("embedder_init_failed", op="recall", exc_info=True)
        return None

    async def forget(self, memory_id: str) -> ForgetResultDict:
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
                raise MemoryNotFoundError(f"Memory entry {memory_id!r} not found")
            if existing.namespace != self._namespace:
                raise MemoryNotFoundError(f"Memory entry {memory_id!r} not found in namespace {self._namespace!r}")
            backend.delete(memory_id)

        logger.debug(
            "memory_forgotten",
            op="forget",
            outcome="success",
            memory_id=memory_id,
            namespace=self._namespace,
        )
        return ForgetResultDict(
            memory_id=memory_id,
            status="deleted",
            namespace=self._namespace,
        )

    async def search(
        self,
        tags: list[str] | None = None,
        min_importance: float = 0.0,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[MemoryResultDict]:
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
            raise ValueError(f"min_importance must be in [0.0, 1.0], got {min_importance}")

        async with self._lock:
            entries = self._get_backend().list_entries(
                namespace=self._namespace,
                limit=limit * 5,  # over-fetch to allow for post-filtering
            )

        # Post-filter
        tag_set: set[str] = set(tags) if tags else set()
        results: list[MemoryResultDict] = []
        for entry in entries:
            if entry.importance < min_importance:
                continue
            if tag_set and not tag_set.issubset(set(entry.tags)):
                continue
            if since is not None and entry.created_at < since:
                continue
            results.append(_entry_to_result(entry, score=entry.importance))

        results.sort(key=lambda r: float(r["score"]), reverse=True)
        final = results[:limit]
        logger.debug(
            "memory_searched",
            op="search",
            outcome="success",
            namespace=self._namespace,
            tag_filter=tags,
            min_importance=min_importance,
            result_count=len(final),
        )
        return final

    # ------------------------------------------------------------------
    # Tool registration (FR09)
    # ------------------------------------------------------------------

    def _make_tool_functions(self) -> dict[str, _ToolFn]:
        """Create the shared tool functions for agent registration."""
        client = self

        async def memory_store(
            content: str,
            tags: list[str] | None = None,
            importance: float = 0.5,
        ) -> StoreResultDict:
            return await client.store(content, tags=tags, importance=importance)

        async def memory_recall(query: str, limit: int = 10) -> list[MemoryResultDict]:
            return await client.recall(query, limit=limit)

        async def memory_forget(memory_id: str) -> ForgetResultDict:
            return await client.forget(memory_id)

        async def memory_search(
            tags: list[str] | None = None,
            min_importance: float = 0.0,
        ) -> list[MemoryResultDict]:
            return await client.search(tags=tags, min_importance=min_importance)

        return {
            "memory_store": memory_store,
            "memory_recall": memory_recall,
            "memory_forget": memory_forget,
            "memory_search": memory_search,
        }

    def register_tools(self, agent: AgentWithRegisterTool | AgentWithToolDecorator) -> None:
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
            raise ToolAlreadyRegisteredError("register_tools() has already been called on this client")

        tools = self._make_tool_functions()

        # Detect registration API
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

        self._tools_registered = True

    # ------------------------------------------------------------------
    # auto_recall decorator (FR06)
    # ------------------------------------------------------------------

    def auto_recall(
        self,
        query_from: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> Callable[[Callable[..., Coroutine[object, object, object]]], Callable[..., Coroutine[object, object, object]]]:
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

        def decorator(
            fn: Callable[..., Coroutine[object, object, object]],
        ) -> Callable[..., Coroutine[object, object, object]]:
            # Check if recalled_memories is a positional arg
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
                except (OSError, ValueError, StorageError, MemoryConnectionError):  # fail-open: recall failures
                    logger.debug("auto_recall_failed", op="auto_recall", outcome="failure", exc_info=True)
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
            logger.debug("client_closed", op="close", namespace=self._namespace)
