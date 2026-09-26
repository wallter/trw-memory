"""MCP tool: memory_verify -- the maintain-verify sweep over one granted namespace (PRD-CORE-280 FR01).

``memory_maintain`` runs the same sweep against the daemon's own configured
``project_root``; a migrated checkout's sweep runs against that checkout, since it
is the one whose files its learnings' assertions and anchors point at. Over the
transport the root is the one the token's grant records: a caller cannot aim the
sweep's file reads anywhere else. The sweep persists verdicts, so this is a write
(not replayed after a lost response).

``memory_assertion_health`` is the read-only side: the session-start summary of
the verdicts the sweep has cached, counted where the rows live.
"""

from __future__ import annotations

from trw_memory.lifecycle import verification_pass
from trw_memory.security.rbac import Permission
from trw_memory.tools import _maintain_sweep
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import serve_namespace

#: The longest stale window ``memory_assertion_health`` accepts (a century); beyond it the cutoff date overflows.
_STALE_DAYS_MAX = 36_500


def register_verify_tool(mcp: McpServer) -> None:
    """Register memory_verify with a FastMCP server instance."""

    async def memory_verify(
        namespace: str,
        project_root: str | None = None,
        settings: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Re-verify *namespace*'s assertion/anchor learnings against the files of the granted checkout.

        One call verifies a bounded part (about a minute or 10,000 rows); while the reply carries
        "next", call again: it resumes where the namespace's last maintain or verify stopped. A
        maintain or verify of the namespace already running makes this one {"status": "busy"}.
        """
        return await _maintain_sweep.serve_verify(namespace, project_root, settings)

    async def memory_assertion_health(namespace: str, stale_days: int) -> dict[str, object]:
        """Count *namespace*'s assertions as passing, failing, stale or unverifiable from their cached verdicts."""
        if isinstance(stale_days, bool) or not isinstance(stale_days, int) or not 1 <= stale_days <= _STALE_DAYS_MAX:
            return {
                "error": f"stale_days must be an int in [1, {_STALE_DAYS_MAX}], not {stale_days!r}",
                "status": "invalid",
            }
        return await serve_namespace(
            namespace,
            Permission.READ,
            "assertion_health",
            lambda backend, _config: {
                "status": "ok",
                "health": verification_pass.assertion_health(backend, namespace=namespace, stale_days=stale_days),
            },
        )

    mcp.tool()(memory_verify)
    mcp.tool()(memory_assertion_health)
