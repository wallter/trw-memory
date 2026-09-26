"""Org-shared recall alias seam for :class:`MemoryClient`.

Split from ``client.py`` (PRD-DIST-246 effective-LOC ratchet). The
implementations live in ``_client_org_shared.py``; the thin wrappers here
exist purely to preserve the three names recall still calls through the client
-- ``_merge_shared_results`` (also the ``monkeypatch.setattr`` test seam),
``_matches_query`` and ``MemoryClient._coerce_float`` -- after the bodies moved
out of the facade. ``MemoryClient`` mixes this in, so attribute resolution
(``self._X`` / ``MemoryClient._X``) is unchanged via the MRO.

Instance wrappers ``cast`` ``self`` to ``MemoryClient`` because the
``_client_org_shared`` implementations type their first parameter as the
concrete client; the cast is a no-op at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from trw_memory._client_models import MemoryResultDict
from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
    from trw_memory.client import MemoryClient


class OrgSharedAliasMixin:
    """Thin org-shared recall delegators preserving the monkeypatch seam."""

    async def _merge_shared_results(
        self,
        query: str,
        local_results: list[MemoryResultDict],
        limit: int,
        *,
        local_entries: list[MemoryEntry] | None = None,
    ) -> list[MemoryResultDict]:
        from trw_memory._client_org_shared import merge_shared_results as _impl

        if local_entries is not None:
            return await _impl(cast("MemoryClient", self), query, local_results, limit, local_entries=local_entries)
        return await _impl(cast("MemoryClient", self), query, local_results, limit)

    @staticmethod
    def _coerce_float(value: object) -> float:
        from trw_memory._client_org_shared import coerce_float as _impl

        return _impl(value)

    @staticmethod
    def _matches_query(result: MemoryResultDict, query: str) -> bool:
        from trw_memory._client_org_shared import matches_query as _impl

        return _impl(result, query)
