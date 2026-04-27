"""MCP tool: memory_review — SEC-001 quarantine review surface."""

from __future__ import annotations

from typing import Literal

from trw_memory.models.config import MemoryConfig
from trw_memory.security.runtime import review_quarantined_entry
from trw_memory.tools._types import McpServer


def memory_review_impl(
    learning_id: str,
    *,
    decision: Literal["approve", "reject"],
    reviewer_id: str,
    config: MemoryConfig | None = None,
) -> dict[str, str]:
    cfg = config or MemoryConfig()
    from trw_memory.integrations._backend import create_backend_from_config

    with create_backend_from_config(cfg, "default") as backend:
        return review_quarantined_entry(
            cfg,
            active_backend=backend,
            learning_id=learning_id,
            decision=decision,
            reviewer_id=reviewer_id,
        )


def register_review_tool(mcp: McpServer) -> None:
    @mcp.tool()
    async def memory_review(
        learning_id: str,
        decision: Literal["approve", "reject"],
        reviewer_id: str,
    ) -> dict[str, str]:
        """Resolve a quarantined memory row once, immutably."""

        return memory_review_impl(
            learning_id,
            decision=decision,
            reviewer_id=reviewer_id,
        )
