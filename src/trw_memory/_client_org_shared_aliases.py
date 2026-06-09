"""Org-shared recall alias seam for :class:`MemoryClient`.

Split from ``client.py`` (PRD-DIST-246 effective-LOC ratchet). The
implementations live in ``_client_org_shared.py``; the thin wrappers here
exist purely to preserve the documented test seam — ``monkeypatch.setattr(
client, "_merge_shared_results", ...)`` instance patches and
``MemoryClient._coerce_float(...)`` static call sites — after the bodies moved
out of the facade. ``MemoryClient`` mixes this in, so attribute resolution
(``self._X`` / ``MemoryClient._X``) is unchanged via the MRO.

Instance wrappers ``cast`` ``self`` to ``MemoryClient`` because the
``_client_org_shared`` implementations type their first parameter as the
concrete client; the cast is a no-op at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from trw_memory._client_models import MemoryResultDict
from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient


class OrgSharedAliasMixin:
    """Thin org-shared recall delegators preserving the monkeypatch seam."""

    async def _merge_shared_results(
        self, query: str, local_results: list[MemoryResultDict], limit: int
    ) -> list[MemoryResultDict]:
        from trw_memory._client_org_shared import merge_shared_results as _impl

        return await _impl(cast("MemoryClient", self), query, local_results, limit)

    async def _load_entries_for_results(self, results: list[MemoryResultDict]) -> list[MemoryEntry]:
        from trw_memory._client_org_shared import load_entries_for_results as _impl

        return await _impl(cast("MemoryClient", self), results)

    @staticmethod
    def _shared_result_to_result(result: dict[str, object]) -> MemoryResultDict:
        from trw_memory._client_org_shared import shared_result_to_result as _impl

        return _impl(result)

    @staticmethod
    def _coerce_float(value: object) -> float:
        from trw_memory._client_org_shared import coerce_float as _impl

        return _impl(value)

    @staticmethod
    def _is_retired_shared_result(result: dict[str, object]) -> bool:
        from trw_memory._client_org_shared import is_retired_shared_result as _impl

        return _impl(result)

    def _merge_shared_candidates(
        self, local_results: list[MemoryResultDict], shared_results: list[MemoryResultDict]
    ) -> list[MemoryResultDict]:
        from trw_memory._client_org_shared import merge_shared_candidates as _impl

        return _impl(local_results, shared_results)

    def _snapshot_cached_shared_results(self, query: str) -> list[MemoryResultDict]:
        from trw_memory._client_org_shared import snapshot_cached_shared_results as _impl

        return _impl(cast("MemoryClient", self), query)

    @staticmethod
    def _matches_query(result: MemoryResultDict, query: str) -> bool:
        from trw_memory._client_org_shared import matches_query as _impl

        return _impl(result, query)

    async def _dedupe_cached_shared_results(
        self,
        cached_results: list[MemoryResultDict],
        *,
        local_entries: list[MemoryEntry],
        embedder: EmbeddingProvider | None,
        dedup_threshold: float = 0.92,
    ) -> list[MemoryResultDict]:
        from trw_memory._client_org_shared import dedupe_cached_shared_results as _impl

        return await _impl(
            cast("MemoryClient", self),
            cached_results,
            local_entries=local_entries,
            embedder=embedder,
            dedup_threshold=dedup_threshold,
        )

    @staticmethod
    def _strip_shared_prefix(content: str) -> str:
        from trw_memory._client_org_shared import strip_shared_prefix as _impl

        return _impl(content)

    async def _mark_fetch_retirements(self, shared_results: list[dict[str, object]]) -> None:
        from trw_memory._client_org_shared import mark_fetch_retirements as _impl

        await _impl(cast("MemoryClient", self), shared_results)
