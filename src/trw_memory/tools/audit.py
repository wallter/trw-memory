"""MCP tool: memory_audit — SEC-001 provenance audit surface."""

from __future__ import annotations

from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.security.runtime import audit_entry
from trw_memory.tools._types import McpServer


def memory_audit_impl(
    learning_id: str,
    *,
    namespace: str = "default",
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    cfg = config or MemoryConfig()
    require_namespace_permission(cfg, namespace, Permission.READ, "audit")
    from trw_memory.integrations._backend import create_backend_from_config

    with create_backend_from_config(cfg, namespace) as backend:
        return audit_entry(cfg, learning_id=learning_id, active_backend=backend, namespace=namespace)


def register_audit_tool(mcp: McpServer) -> None:
    @mcp.tool()
    async def memory_audit(learning_id: str, namespace: str = "default") -> dict[str, object]:
        """Return provenance + lifecycle audit data for one memory row."""

        return memory_audit_impl(learning_id, namespace=namespace)
