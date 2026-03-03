"""FastMCP server entry point for trw-memory.

Registers all 6 MCP tools and exposes a ``main()`` callable for the
``trw-memory`` console script defined in pyproject.toml.

fastmcp is an optional dependency (``pip install trw-memory[mcp]``).
Importing this module without fastmcp installed will raise ImportError.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("trw-memory")


def _register_tools() -> None:
    from trw_memory.tools.consolidate import register_consolidate_tool
    from trw_memory.tools.forget import register_forget_tool
    from trw_memory.tools.recall import register_recall_tool
    from trw_memory.tools.search import register_search_tool
    from trw_memory.tools.status import register_status_tool
    from trw_memory.tools.store import register_store_tool

    register_store_tool(mcp)
    register_recall_tool(mcp)
    register_forget_tool(mcp)
    register_consolidate_tool(mcp)
    register_search_tool(mcp)
    register_status_tool(mcp)


_register_tools()


def main() -> None:
    mcp.run()
