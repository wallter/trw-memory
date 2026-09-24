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

from dataclasses import asdict
from pathlib import Path

from trw_memory.lifecycle import verification_pass
from trw_memory.lifecycle.verification_pass import VerifySettings
from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import checkout_path, in_namespace, refused_namespace

#: The longest stale window ``memory_assertion_health`` accepts (a century); beyond it the cutoff date overflows.
_STALE_DAYS_MAX = 36_500


def memory_verify_impl(
    namespace: str, project_root: str | None, settings: dict[str, object] | None, *, backend: StorageBackend
) -> dict[str, object]:
    """``{"status": "ok", "summary": {...}}`` -- see ``MaintainVerifySummary``."""
    try:
        knobs = VerifySettings(**(settings or {}))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        return {"error": f"invalid verify settings: {exc}", "status": "invalid"}
    summary = verification_pass.run_maintain_verify(
        backend, project_root=Path(project_root) if project_root else None, namespace=namespace, **asdict(knobs)
    )
    return {"status": "ok", "summary": summary.as_dict()}


def register_verify_tool(mcp: McpServer) -> None:
    """Register memory_verify with a FastMCP server instance."""

    async def memory_verify(
        namespace: str, project_root: str | None = None, settings: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Re-verify *namespace*'s assertion/anchor learnings against the files of the granted checkout."""
        if refused := refused_namespace(namespace, Permission.WRITE, "verify", MemoryConfig()):
            return refused
        root = checkout_path(project_root, "memory_verify", within=False)
        if isinstance(root, dict):
            return root
        return in_namespace(
            namespace,
            Permission.WRITE,
            "verify",
            lambda backend, _config: memory_verify_impl(namespace, root, settings, backend=backend),
        )

    async def memory_assertion_health(namespace: str, stale_days: int) -> dict[str, object]:
        """Count *namespace*'s assertions as passing, failing, stale or unverifiable from their cached verdicts."""
        if isinstance(stale_days, bool) or not isinstance(stale_days, int) or not 1 <= stale_days <= _STALE_DAYS_MAX:
            return {
                "error": f"stale_days must be an int in [1, {_STALE_DAYS_MAX}], not {stale_days!r}",
                "status": "invalid",
            }
        return in_namespace(
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
