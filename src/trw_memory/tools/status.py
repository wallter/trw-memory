"""MCP tool: memory_status — report on the current state of the memory store.

Returns total entry count, namespace breakdown, and a summary of the
active configuration. Namespace is optional — when None, reports globally.
"""

from __future__ import annotations

import structlog

from trw_memory.exceptions import ConfigError
from trw_memory.integrations._backend import discover_namespace_backends
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.runtime import list_quarantined_entries, security_maintenance_status
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer

logger = structlog.get_logger(__name__)

# Canonical namespaces to include in per-namespace breakdown when no filter applied
_COMMON_NAMESPACES = ["project:default", "global"]


def _quarantine_count(cfg: MemoryConfig) -> int | None:
    try:
        return len(list_quarantined_entries(cfg, limit=10_000))
    except Exception:  # broad catch: posture should never make status unusable
        logger.debug("memory_status_quarantine_count_failed", exc_info=True)
        return None


def _security_posture(cfg: MemoryConfig) -> dict[str, object] | None:
    """Return compact posture only when security is degraded or non-default."""
    quarantine_count = _quarantine_count(cfg)
    maintenance = security_maintenance_status()
    signals: dict[str, object] = {
        "recall_filter_mode": "disabled" if not cfg.enable_recall_filter else cfg.recall_filter_mode,
        "trust_scoring_mode": "disabled" if not cfg.enable_trust_scoring else cfg.trust_scoring_mode,
        "quarantine_count": quarantine_count,
        "canary_status": "halt" if cfg.canary_fail_mode == "halt" else f"degraded:{cfg.canary_fail_mode}",
        "provenance_mode": "required" if cfg.provenance_required else "optional",
        "pii_mode": "disabled" if not cfg.pii_enabled else cfg.pii_action,
        "maintenance": maintenance,
    }
    queued_raw = maintenance.get("queued", 0)
    queued_count = queued_raw if isinstance(queued_raw, int) else 0
    quarantine_degraded = quarantine_count is None or quarantine_count > 0
    degraded = (
        not cfg.enable_recall_filter
        or not cfg.enable_trust_scoring
        or not cfg.provenance_required
        or not cfg.pii_enabled
        or cfg.canary_fail_mode != "halt"
        or queued_count > 0
        or quarantine_degraded
    )
    non_default = (
        cfg.recall_filter_mode != "redact"
        or cfg.trust_scoring_mode != "observe"
        or cfg.pii_action != "warn"
        or not cfg.security_maintenance_inline
    )
    if not degraded and not non_default:
        return None
    return {"status": "degraded" if degraded else "non_default", **signals}


def _status_introspection(cfg: MemoryConfig) -> dict[str, object]:
    """Expose live tool/config inventory for generated docs checks."""
    security_fields = sorted(
        field_name
        for field_name in MemoryConfig.model_fields
        if field_name.startswith(("audit_", "pii_", "poisoning_", "quarantine_", "provenance_", "canary_"))
        or field_name
        in {
            "enable_recall_filter",
            "enable_trust_scoring",
            "recall_filter_mode",
            "trust_scoring_mode",
            "security_maintenance_inline",
        }
    )
    recovery_fields = sorted(field_name for field_name in MemoryConfig.model_fields if "recovery" in field_name)
    return {
        "tool": "memory_status",
        "tool_version": 1,
        "storage_backend": cfg.storage_backend,
        "security_config_fields": security_fields,
        "recovery_config_fields": recovery_fields,
    }


def memory_status_impl(
    namespace: str | None,
    *,
    backend: StorageBackend,
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    """Core implementation of memory_status (callable without MCP).

    Args:
        namespace: If provided, scope the count to this namespace.
            Must be a valid namespace if given.
        backend: Storage backend instance.
        config: MemoryConfig for the configuration summary. Uses defaults if None.

    Returns:
        {"total_entries": int, "namespaces": dict, "config": dict}
        or {"error": str, "status": "invalid"} on validation failure.
    """
    cfg = config or MemoryConfig()

    # Validate namespace if provided
    if namespace is not None:
        try:
            validate_namespace(namespace)
        except ConfigError as exc:
            return {"error": str(exc), "status": "invalid"}

    # Build namespace breakdown
    namespaces: dict[str, int] = {}
    if namespace is not None:
        try:
            total_entries = backend.count(namespace=namespace)
        except Exception as exc:  # broad catch: tool error boundary
            logger.exception("memory_status_count_failed", error=str(exc))
            return {"error": f"storage error: {exc}", "status": "error"}
        namespaces[namespace] = total_entries
    elif config is None:
        try:
            total_entries = backend.count(namespace=None)
        except Exception as exc:  # broad catch: tool error boundary
            logger.exception("memory_status_count_failed", error=str(exc))
            return {"error": f"storage error: {exc}", "status": "error"}

        # Try to count a few common namespaces
        for ns in _COMMON_NAMESPACES:
            try:
                ns_count = backend.count(namespace=ns)
                namespaces[ns] = ns_count
            except Exception:  # broad catch: best-effort namespace count, per-item skip intentional
                logger.debug("memory_status_ns_count_failed", namespace=ns, exc_info=True)

        # Also include active entry count
        try:
            active_entries = backend.list_entries(
                status=MemoryStatus.ACTIVE,
                limit=10_000,
            )
            namespaces["__active__"] = len(active_entries)
        except Exception:  # broad catch: best-effort active count
            logger.debug("memory_status_active_count_failed", exc_info=True)
    else:
        try:
            total_entries = 0
            active_entry_count = 0
            with discover_namespace_backends(cfg) as stores:
                for store_namespaces, store_backend in stores:
                    for store_namespace in store_namespaces:
                        ns_count = store_backend.count(namespace=store_namespace)
                        total_entries += ns_count
                        existing = namespaces.get(store_namespace, 0)
                        namespaces[store_namespace] = int(existing) + ns_count

                    active_entry_count += len(
                        store_backend.list_entries(
                            status=MemoryStatus.ACTIVE,
                            limit=10_000,
                        )
                    )
        except Exception as exc:  # broad catch: tool error boundary
            logger.exception("memory_status_count_failed", error=str(exc))
            return {"error": f"storage error: {exc}", "status": "error"}

        namespaces["__active__"] = active_entry_count

    config_summary: dict[str, object] = {
        "storage_backend": cfg.storage_backend,
        "dedup_enabled": cfg.dedup_enabled,
        "consolidation_enabled": cfg.consolidation_enabled,
        "hot_max_entries": cfg.hot_max_entries,
        "retention_days": cfg.retention_days,
        "security_maintenance_inline": cfg.security_maintenance_inline,
        "memory_recovery_inline_max_bytes": cfg.memory_recovery_inline_max_bytes,
    }
    security_posture = _security_posture(cfg)

    logger.debug(
        "memory_status",
        namespace=namespace,
        total_entries=total_entries,
    )

    result: dict[str, object] = {
        "total_entries": total_entries,
        "namespaces": namespaces,
        "config": config_summary,
        "introspection": _status_introspection(cfg),
    }
    if security_posture is not None:
        result["security_posture"] = security_posture
    return result


def register_status_tool(mcp: McpServer) -> None:
    """Register memory_status with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from trw_memory.integrations._backend import create_backend_from_config

    @mcp.tool()
    async def memory_status(
        namespace: str | None = None,
    ) -> dict[str, object]:
        """Report the current state of the memory store.

        Args:
            namespace: If provided, scope to this namespace (e.g., 'project:default').
                When None, reports across all namespaces.

        Returns:
            {"total_entries": int, "namespaces": dict, "config": dict}
        """
        cfg = MemoryConfig()
        backend_namespace = namespace or "default"
        with create_backend_from_config(cfg, backend_namespace) as backend:
            return memory_status_impl(
                namespace,
                backend=backend,
                config=cfg,
            )
