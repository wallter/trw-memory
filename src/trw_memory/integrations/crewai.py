"""CrewAI integration — RAGStorage-compatible adapter.

Provides a storage backend for CrewAI's memory system using trw-memory.
Implements the duck-typed RAGStorage interface (``save``, ``search``,
``reset``) used by CrewAI's ``ShortTermMemory`` and ``EntityMemory``.

Usage::

    from crewai.memory.short_term.short_term_memory import ShortTermMemory
    from trw_memory.integrations.crewai import TRWCrewStorage

    storage = TRWCrewStorage(namespace="project:my-crew")
    memory = ShortTermMemory(storage=storage)

Requires ``crewai >= 0.74.0``::

    pip install "trw-memory[crewai]"
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import TYPE_CHECKING

from typing_extensions import TypedDict

# Verify crewai is available (but don't import heavy modules)
try:
    _crewai_spec = importlib.util.find_spec("crewai")
except (ValueError, ModuleNotFoundError):
    _crewai_spec = None

if _crewai_spec is None:
    raise ImportError('crewai is required for the CrewAI adapter. Install it with: pip install "trw-memory[crewai]"')

_MIN_CREWAI_VERSION = (0, 74, 0)


def _parse_version(version: str) -> tuple[int, ...]:
    """Extract a comparable numeric prefix from a version string."""
    numbers: list[int] = []
    for part in version.replace("-", ".").split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


try:
    _crewai_version = importlib.metadata.version("crewai")
except importlib.metadata.PackageNotFoundError as exc:
    raise ImportError(
        'crewai metadata is required for the CrewAI adapter. Install it with: pip install "trw-memory[crewai]"'
    ) from exc

if _parse_version(_crewai_version) < _MIN_CREWAI_VERSION:
    raise ImportError(
        f"crewai>={_MIN_CREWAI_VERSION[0]}.{_MIN_CREWAI_VERSION[1]}.{_MIN_CREWAI_VERSION[2]} "
        f"is required for the CrewAI adapter (found {_crewai_version}). "
        'Install it with: pip install "trw-memory[crewai]"'
    )

from trw_memory.integrations._mixin import BackendOwnerMixin  # noqa: E402 — import after crewai version check

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

_TAG_PREFIX = "crewai"


class CrewSearchResult(TypedDict):
    """Typed structure for a single CrewAI search result entry."""

    id: str
    metadata: dict[str, str]
    context: str
    score: float


class TRWCrewStorage(BackendOwnerMixin):
    """RAGStorage-compatible adapter backed by trw-memory.

    Implements the duck-typed interface expected by CrewAI's memory classes:
    ``save()``, ``search()``, and ``reset()``.

    Args:
        namespace: trw-memory namespace for storage isolation.
        storage_path: Override for the storage directory.
        search_limit: Maximum number of results from ``search()``.
        backend: Pre-existing backend (for testing).
    """

    def __init__(
        self,
        namespace: str = "default",
        *,
        storage_path: str | None = None,
        search_limit: int = 10,
        backend: StorageBackend | None = None,
    ) -> None:
        from trw_memory.integrations._backend import config_for_storage_path, resolve_backend

        self._namespace = namespace
        self._search_limit = search_limit
        self._config = config_for_storage_path(storage_path)
        self._backend, self._owns_backend = resolve_backend(
            namespace,
            storage_path,
            backend,
        )

    @property
    def namespace(self) -> str:
        """The namespace this storage operates in."""
        return self._namespace

    def save(
        self,
        value: object,
        metadata: dict[str, object] | None = None,
        agent: str | None = None,
    ) -> None:
        """Persist a memory entry.

        The write goes through :func:`guarded_store_or_raise`, so crew output
        carrying an injection payload is rejected instead of persisted and
        replayed on every later ``search``. The rejection RAISES rather than
        skipping: ``save`` is a single-item call whose only return channel is
        ``None``, so a skip would be indistinguishable from success to CrewAI's
        memory layer. A QUARANTINE raises for the same reason — until 2026-07-30
        this discarded ``guarded_store``'s result, so a held entry was dropped
        while ``save`` returned normally.

        Args:
            value: Content to store (converted to string).
            metadata: Arbitrary key-value pairs.
            agent: If provided, tags the entry with the agent name.

        Raises:
            PoisoningError, PIIBlockError, RateLimitError: from the store gate.
        """
        from trw_memory.integrations._backend import make_entry
        from trw_memory.security.write_gate import guarded_store_or_raise

        tags: list[str] = [_TAG_PREFIX]
        if agent:
            tags.append(f"agent:{agent}")

        str_metadata: dict[str, str] = {}
        if metadata:
            str_metadata = {str(k): str(v) for k, v in metadata.items()}

        entry = make_entry(
            content=str(value),
            namespace=self._namespace,
            tags=tags,
            importance=0.5,
            metadata=str_metadata,
            source="agent",
        )
        guarded_store_or_raise(self._backend, entry, config=self._config)

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: dict[str, object] | None = None,
        score_threshold: float = 0.0,
    ) -> list[CrewSearchResult]:
        """Search stored memories by query.

        Args:
            query: Free-text search string.
            limit: Maximum number of results.
            filter: Optional exact-match metadata filters.
            score_threshold: Minimum importance for results.

        Returns:
            List of dicts with ``id``, ``metadata``, ``context``, ``score``.
        """
        effective_limit = min(limit, self._search_limit)
        entries = self._backend.search(
            query,
            top_k=effective_limit,
            namespace=self._namespace,
        )

        str_filter = {str(key): str(value) for key, value in filter.items()} if filter else None
        results = [
            CrewSearchResult(
                id=entry.id,
                metadata=dict(entry.metadata),
                context=entry.content,
                score=entry.importance,
            )
            for entry in entries
            if entry.importance >= score_threshold
            and (str_filter is None or all(entry.metadata.get(key) == value for key, value in str_filter.items()))
        ]
        return results

    def reset(self) -> None:
        """Clear all stored memories in this namespace."""
        deleted = self._backend.delete_by_namespace(self._namespace)
        if deleted == 0:
            return

    # Resource management inherited from BackendOwnerMixin.
