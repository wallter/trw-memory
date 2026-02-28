"""MCP tools for trw-memory — 6 tools registered via FastMCP."""

from trw_memory.tools.consolidate import (
    memory_consolidate_impl,
    register_consolidate_tool,
)
from trw_memory.tools.forget import memory_forget_impl, register_forget_tool
from trw_memory.tools.recall import memory_recall_impl, register_recall_tool
from trw_memory.tools.search import memory_search_impl, register_search_tool
from trw_memory.tools.status import memory_status_impl, register_status_tool
from trw_memory.tools.store import memory_store_impl, register_store_tool

__all__ = [
    "memory_consolidate_impl",
    "memory_forget_impl",
    "memory_recall_impl",
    "memory_search_impl",
    "memory_status_impl",
    "memory_store_impl",
    "register_consolidate_tool",
    "register_forget_tool",
    "register_recall_tool",
    "register_search_tool",
    "register_status_tool",
    "register_store_tool",
]
