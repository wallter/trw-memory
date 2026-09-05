"""MCP tools for trw-memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trw_memory.tools._contract import (
    AuditImpl,
    ConsolidateImpl,
    ForgetImpl,
    MemoryToolSurface,
    RecallImpl,
    ReviewImpl,
    SearchImpl,
    StatusImpl,
    StoreImpl,
)
from trw_memory.tools._types import McpServer
from trw_memory.tools.audit import memory_audit_impl, register_audit_tool
from trw_memory.tools.consolidate import (
    memory_consolidate_impl,
    register_consolidate_tool,
)
from trw_memory.tools.forget import memory_forget_impl, register_forget_tool
from trw_memory.tools.recall import memory_recall_impl, register_recall_tool
from trw_memory.tools.review import memory_review_impl, register_review_tool
from trw_memory.tools.search import memory_search_impl, register_search_tool
from trw_memory.tools.status import memory_status_impl, register_status_tool
from trw_memory.tools.store import memory_store_impl, register_store_tool

__all__ = [
    "McpServer",
    "MemoryToolSurface",
    "memory_audit_impl",
    "memory_consolidate_impl",
    "memory_forget_impl",
    "memory_recall_impl",
    "memory_review_impl",
    "memory_search_impl",
    "memory_status_impl",
    "memory_store_impl",
    "register_audit_tool",
    "register_consolidate_tool",
    "register_forget_tool",
    "register_recall_tool",
    "register_review_tool",
    "register_search_tool",
    "register_status_tool",
    "register_store_tool",
]

if TYPE_CHECKING:  # pragma: no cover - static conformance proof (PRD-CORE-251 FR01)
    # Binding each impl to its contract member is what makes MemoryToolSurface a
    # gate rather than documentation: `mypy --strict` fails here the moment an
    # implementation's signature drifts from the shape this package publishes.
    _store_conforms: StoreImpl = memory_store_impl
    _recall_conforms: RecallImpl = memory_recall_impl
    _search_conforms: SearchImpl = memory_search_impl
    _forget_conforms: ForgetImpl = memory_forget_impl
    _consolidate_conforms: ConsolidateImpl = memory_consolidate_impl
    _status_conforms: StatusImpl = memory_status_impl
    _review_conforms: ReviewImpl = memory_review_impl
    _audit_conforms: AuditImpl = memory_audit_impl
