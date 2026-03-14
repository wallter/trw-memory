"""Shared type definitions for MCP tool registration helpers.

Defines the ``McpServer`` Protocol so that ``register_*_tool(mcp: McpServer)``
is fully typed without importing FastMCP at type-check time.  Individual tool
modules import from here rather than from the package ``__init__`` to avoid
circular imports (``__init__`` itself re-exports ``McpServer`` for convenience).
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class McpServer(Protocol):
    """Structural Protocol capturing the subset of FastMCP used by register_*_tool helpers.

    Only the ``.tool()`` decorator method is required.  All ``register_*_tool``
    functions accept ``mcp: McpServer`` instead of ``mcp: Any`` so mypy can
    verify call sites without importing fastmcp at type-check time.

    The return type of ``.tool()`` is typed as ``Callable[..., object]`` so
    that ``@mcp.tool()`` is recognised as a decorator call by mypy without
    requiring a type-ignore comment.
    """

    def tool(self) -> Callable[..., object]:  # noqa: D102 — Protocol stub
        ...  # pragma: no cover
