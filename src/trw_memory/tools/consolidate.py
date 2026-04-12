"""MCP tool: memory_consolidate — cluster and merge similar memory entries.

Thin wrapper around lifecycle.consolidation.consolidate_cycle. Validates
namespace, runs the consolidation cycle (or a dry-run preview), and returns
a structured result dict.

Also supports team namespace promotion: when namespace starts with "team:",
high-importance entries are copied to the project namespace.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import structlog

from trw_memory.integrations._backend import discover_namespace_backends
from trw_memory.embeddings import get_local_embedder
from trw_memory.exceptions import ConfigError, StorageError
from trw_memory.lifecycle.consolidation import consolidate_cycle
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools._types import McpServer

logger = structlog.get_logger(__name__)
TEAM_NAMESPACE_WILDCARD = "team:*"


def _promote_team_memories(
    namespace: str,
    source_backend: StorageBackend,
    *,
    target_backend: StorageBackend | None = None,
    promotion_threshold: float = 0.7,
) -> dict[str, object]:
    """Promote high-impact team memories to the project namespace.

    Entries with importance >= promotion_threshold are copied to
    "project:default" with provenance tracking. Lower-importance
    entries are counted but not promoted.

    Args:
        namespace: Team namespace (e.g., "team:sprint-37").
        source_backend: Backend that owns the team namespace entries.
        target_backend: Backend that should receive promoted project entries.
            Defaults to ``source_backend`` for tests or shared-store backends.
        promotion_threshold: Minimum importance to promote (default 0.7).

    Returns:
        {"promoted_count": int, "discarded_count": int, "namespace_id": str,
         "completed_at": str}
    """
    entries = source_backend.list_entries(
        status=MemoryStatus.ACTIVE,
        namespace=namespace,
        limit=10_000,
    )

    project_backend = target_backend or source_backend
    promoted_count = 0
    discarded_count = 0
    now = datetime.now(timezone.utc)

    for entry in entries:
        if entry.importance >= promotion_threshold:
            outcome = f"promoted_from:{namespace}:timestamp={now.isoformat()}"
            promoted = entry.model_copy(
                update={
                    "id": f"promoted-{entry.id}",
                    "namespace": "project:default",
                    "source_identity": namespace,
                    "outcome_history": [*entry.outcome_history, outcome],
                    "updated_at": now,
                }
            )
            project_backend.store(promoted)
            promoted_count += 1
        else:
            discarded_count += 1

    logger.info(
        "team_memories_promoted",
        namespace=namespace,
        promoted=promoted_count,
        discarded=discarded_count,
    )

    if isinstance(source_backend, SQLiteBackend):
        NamespaceManager(source_backend).mark_team_namespace_completed(namespace, completed_at=now)

    return {
        "promoted_count": promoted_count,
        "discarded_count": discarded_count,
        "namespace_id": namespace,
        "completed_at": now.isoformat(),
    }


def _promote_all_team_namespaces(
    cfg: MemoryConfig,
    *,
    namespace_backend_factory: Callable[[str], StorageBackend] | None = None,
) -> dict[str, object]:
    """Promote all discovered team namespaces and aggregate their summaries."""
    namespace_results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    seen_namespaces: set[str] = set()

    with discover_namespace_backends(cfg) as stores:
        for namespaces, store_backend in stores:
            for namespace in namespaces:
                if not namespace.startswith("team:") or namespace in seen_namespaces:
                    continue
                seen_namespaces.add(namespace)

                if isinstance(store_backend, SQLiteBackend):
                    if NamespaceManager(store_backend).team_namespace_expired(namespace):
                        logger.debug("team_namespace_wildcard_skip_expired", namespace=namespace)
                        continue

                project_backend = namespace_backend_factory("project:default") if namespace_backend_factory else store_backend
                try:
                    namespace_results.append(
                        _promote_team_memories(
                            namespace,
                            store_backend,
                            target_backend=project_backend,
                        )
                    )
                except (StorageError, ValueError) as exc:
                    logger.exception(
                        "team_namespace_wildcard_namespace_failed",
                        namespace=namespace,
                        error=str(exc),
                    )
                    errors.append({"namespace": namespace, "error": str(exc)})
                finally:
                    if project_backend is not store_backend:
                        project_backend.close()

    if not namespace_results and not errors:
        logger.debug("team_namespace_wildcard_skipped", reason="no_team_namespaces")
        return {
            "promoted_count": 0,
            "discarded_count": 0,
            "namespace_id": TEAM_NAMESPACE_WILDCARD,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "namespaces": [],
            "status": "skipped",
            "skipped_reason": "no_team_namespaces",
        }

    return {
        "promoted_count": sum(int(str(result["promoted_count"])) for result in namespace_results),
        "discarded_count": sum(int(str(result["discarded_count"])) for result in namespace_results),
        "namespace_id": TEAM_NAMESPACE_WILDCARD,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "namespaces": namespace_results,
        **({"errors": errors} if errors else {}),
        **({"status": "error" if not namespace_results else "partial"} if errors else {}),
    }


def memory_consolidate_impl(
    namespace: str,
    *,
    backend: StorageBackend,
    dry_run: bool = False,
    config: MemoryConfig | None = None,
    namespace_backend_factory: Callable[[str], StorageBackend] | None = None,
) -> dict[str, object]:
    """Core implementation of memory_consolidate (callable without MCP).

    Args:
        namespace: Namespace to consolidate within (e.g., "project:default").
        backend: Storage backend instance.
        dry_run: If True, preview clusters without modifying storage.
        config: Optional MemoryConfig. When omitted, the default config is loaded.
        namespace_backend_factory: Optional backend factory used when team namespace
            promotion must write into a different namespace store.

    Returns:
        {"clusters_found": int, "entries_consolidated": int, "dry_run": bool}
        or {"error": str, "status": "invalid"} on validation failure.
    """
    if namespace == TEAM_NAMESPACE_WILDCARD:
        cfg = config or MemoryConfig()
        return _promote_all_team_namespaces(
            cfg,
            namespace_backend_factory=namespace_backend_factory,
        )

    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}

    # Team namespace promotion: copy high-impact entries to project namespace
    if namespace.startswith("team:"):
        project_backend = namespace_backend_factory("project:default") if namespace_backend_factory else backend
        try:
            return _promote_team_memories(
                namespace,
                backend,
                target_backend=project_backend,
            )
        finally:
            if project_backend is not backend:
                project_backend.close()

    cfg = config or MemoryConfig()
    embedder = get_local_embedder(model_name=cfg.embedding_model, dim=cfg.embedding_dim)

    try:
        result = consolidate_cycle(
            backend,
            # Public consolidate entrypoints must resolve the embedder here; the
            # lifecycle engine only clusters when an embedder is explicitly present.
            embedder=embedder,
            dry_run=dry_run,
            namespace=namespace,
            config=cfg,
        )
    except (StorageError, ValueError) as exc:
        logger.exception("memory_consolidate_failed", namespace=namespace, error=str(exc))
        return {"error": f"consolidation error: {exc}", "status": "error"}

    # Normalise result keys for the MCP contract
    clusters_found = int(str(result.get("clusters_found", 0)))
    consolidated_count = int(str(result.get("consolidated_count", 0)))

    logger.info(
        "memory_consolidate",
        namespace=namespace,
        dry_run=dry_run,
        clusters_found=clusters_found,
        entries_consolidated=consolidated_count,
    )

    return {
        "clusters_found": clusters_found,
        "entries_consolidated": consolidated_count,
        "dry_run": bool(result.get("dry_run", dry_run)),
        **({"clusters": result["clusters"]} if "clusters" in result else {}),
        **({"status": str(result["status"])} if "status" in result else {}),
        **({"skipped_reason": str(result["skipped_reason"])} if "skipped_reason" in result else {}),
        **({"errors": result["errors"]} if "errors" in result else {}),
    }


def register_consolidate_tool(mcp: McpServer) -> None:
    """Register memory_consolidate with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from trw_memory.integrations._backend import create_backend_from_config

    @mcp.tool()
    async def memory_consolidate(
        namespace: str = "project:default",
        dry_run: bool = False,
    ) -> dict[str, object]:
        """Consolidate similar memory entries by clustering and merging.

        Uses embedding-based clustering to find semantically similar entries,
        then merges each cluster into a single consolidated entry. Originals
        are archived.

        Args:
            namespace: Namespace to consolidate (e.g., 'project:default').
            dry_run: If True, preview clusters without writing changes.

        Returns:
            {"clusters_found": int, "entries_consolidated": int, "dry_run": bool}
        """
        cfg = MemoryConfig()
        backend_namespace = "default" if namespace == TEAM_NAMESPACE_WILDCARD else namespace
        with create_backend_from_config(cfg, backend_namespace) as backend:
            return memory_consolidate_impl(
                namespace,
                backend=backend,
                dry_run=dry_run,
                config=cfg,
                namespace_backend_factory=lambda extra_ns: create_backend_from_config(cfg, extra_ns),
            )
