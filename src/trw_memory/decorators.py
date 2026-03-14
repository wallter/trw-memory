"""Decorators for automatic memory recall injection.

The primary decorator is :meth:`MemoryClient.auto_recall`, which is a
bound method on the client.  This module provides standalone documentation
and re-exports for convenience.

Usage::

    client = MemoryClient(namespace="project:my-app")

    @client.auto_recall(query_from="prompt", limit=5)
    async def handle(prompt: str, *, recalled_memories: list[dict[str, object]] | None = None):
        ...
"""

from __future__ import annotations

# The auto_recall decorator is implemented as a method on MemoryClient.
# This module exists for import convenience and documentation.
from trw_memory.client import MemoryClient

__all__ = ["MemoryClient"]
