"""Shared mixin for integration adapter resource management."""
from __future__ import annotations

from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend


class BackendOwnerMixin:
    """Mixin providing close() + context manager for adapters that own a backend."""

    _backend: StorageBackend
    _owns_backend: bool

    def close(self) -> None:
        """Release backend resources if this instance owns them."""
        if self._owns_backend:
            self._backend.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
