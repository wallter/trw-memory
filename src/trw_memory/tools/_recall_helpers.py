"""Shared recall helper functions used by the MCP recall surface."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack

import structlog

from trw_memory.graph import graph_query, list_org_shared_entries
from trw_memory.lifecycle._recall import rank_by_utility, record_recall_access
from trw_memory.lifecycle.scoring import entry_utility
from trw_memory.lifecycle.tiers._scoring import compute_importance_score
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security.recall_filter import filter_recall_window
from trw_memory.security.telemetry_emit import build_security_traceability, emit_security_event
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)


def _apply_sec001_recall_policy(
    results: list[dict[str, object]],
    *,
    config: MemoryConfig,
    namespace: str = "default",
) -> list[dict[str, object]]:
    if not config.enable_recall_filter:
        return results
    result_by_id: dict[str, dict[str, object]] = {}
    entries: list[MemoryEntry] = []
    for idx, result in enumerate(results):
        synthetic_id = f"{result['id']}::{idx}"
        result_by_id[synthetic_id] = result
        entries.append(MemoryEntry.model_validate({**result, "id": synthetic_id}))
    filtered = filter_recall_window(entries, mode=config.recall_filter_mode)
    session_id = os.environ.get("TRW_SESSION_ID", "").strip() or namespace
    run_id = os.environ.get("TRW_RUN_ID", "").strip() or None
    emit_security_event(
        config,
        emitter="recall_filter",
        session_id=session_id,
        run_id=run_id,
        payload={
            "event_name": "recall_filter_outcome",
            "path": "tool_recall",
            "namespace": namespace,
            "mode": config.recall_filter_mode,
            "window_size": len(entries),
            "accepted_count": len(filtered.accepted),
            "would_reject_count": len(filtered.would_reject),
            "actions": dict(filtered.actions),
            "traceability": build_security_traceability(
                live_path="tools.recall.memory_recall_impl",
                requirement_ids=["FR-003", "NFR-010", "NFR-011"],
            ),
        },
    )
    secured: list[dict[str, object]] = []
    for entry in filtered.accepted:
        if entry.metadata.get("system_canary") == "true":
            continue
        source_result = result_by_id.get(entry.id)
        if source_result is None:
            source = entry.model_dump(mode="json")
            if "::" in entry.id:
                source["id"] = entry.id.rsplit("::", 1)[0]
        else:
            source = dict(source_result)
        source["content"] = entry.content
        source["detail"] = entry.detail
        source["metadata"] = dict(entry.metadata)
        secured.append(source)
    return secured


def _merge_tier_entries(
    ranked_dicts: list[dict[str, object]],
    tier_dicts: list[dict[str, object]],
    query_tokens: list[str],
    config: MemoryConfig,
    query_embedding: list[float] | None,
) -> list[dict[str, object]]:
    """Merge tier-only matches into the main recall candidate set."""
    merged: list[dict[str, object]] = list(ranked_dicts)
    seen_keys = {(str(item.get("namespace", "project:default")), str(item.get("id", ""))) for item in ranked_dicts}
    for item in tier_dicts:
        key = (str(item.get("namespace", "project:default")), str(item.get("id", "")))
        if key in seen_keys:
            continue
        merged.append(item)
        seen_keys.add(key)
    for item in merged:
        relevance_hint = item.get("_tier_relevance")
        item["score"] = compute_importance_score(
            item,
            query_tokens,
            query_embedding=query_embedding,
            config=config,
            relevance_hint=float(str(relevance_hint)) if relevance_hint is not None else None,
        )
    merged.sort(key=lambda entry: float(str(entry.get("score", 0.0))), reverse=True)
    return merged


def _org_memory_results(
    config: MemoryConfig,
    namespace: str,
    query: str,
    tags: list[str] | None,
    min_score: float,
    *,
    exclude_keys: set[tuple[str, str]],
    limit: int,
) -> list[dict[str, object]]:
    """Build additive org-wide recall results from sibling project stores."""
    org_entries = list_org_shared_entries(
        config,
        namespace,
        exclude_keys=exclude_keys,
        limit=max(limit, 25),
    )
    if not org_entries:
        return []

    query_tokens = query.lower().split() if query else []
    tag_set = set(tags or [])
    org_results = []
    for entry in org_entries:
        if tag_set and not tag_set.issubset(set(entry.tags)):
            continue
        if query_tokens and not _entry_matches_query(entry.model_dump(mode="json"), query_tokens):
            continue

        item = entry.model_dump(mode="json")
        item["scope"] = "org"
        if min_score > 0.0 and entry_utility(item, config=config) < min_score:
            continue
        org_results.append(item)

    return rank_by_utility(org_results, query_tokens, lambda_weight=0.4, config=config)


def _entry_matches_query(entry: dict[str, object], query_tokens: list[str]) -> bool:
    """Return whether any query token appears in content, detail, or tags."""
    if not query_tokens:
        return True
    content = str(entry.get("content", "")).lower()
    detail = str(entry.get("detail", "")).lower()
    raw_tags = entry.get("tags", [])
    tag_text = " ".join(str(tag).lower() for tag in raw_tags) if isinstance(raw_tags, list) else ""
    return any(token in f"{content} {detail} {tag_text}" for token in query_tokens)


def _graph_related(
    result_dicts: list[dict[str, object]],
    depth: int,
    backend: StorageBackend,
    conn: sqlite3.Connection | None,
    namespace: str | None = None,
) -> list[dict[str, object]]:
    """Query the knowledge graph for entries related to the recall results.

    ``namespace`` scopes BFS traversal so related entries never leak across
    namespaces (memory_graph_edges has no namespace column). ``None`` keeps
    the legacy unscoped behaviour for back-compat.
    """
    effective_conn = conn
    if effective_conn is None:
        effective_conn = getattr(backend, "_conn", None)
    if effective_conn is None:
        logger.debug("graph_related_skip", reason="no_sqlite_connection")
        return []

    root_ids = [str(d["id"]) for d in result_dicts if "id" in d]

    try:
        related_nodes = graph_query(effective_conn, root_ids, depth=depth, namespace=namespace)
    except (sqlite3.Error, ValueError, KeyError):
        logger.debug("graph_related_error", exc_info=True)
        return []

    hydrated: list[dict[str, object]] = []
    for node in related_nodes:
        entry = backend.get(str(node["id"]))
        if entry is None:
            # Dangling edge -- the target row was deleted; skip without crashing.
            continue
        if entry.status != MemoryStatus.ACTIVE:
            # The primary recall path lists only ACTIVE entries; the graph
            # related path must not re-surface obsolete / archived / poisoned /
            # resolved learnings (same obsolete-leak class fixed in recall).
            continue
        item = entry.model_dump(mode="json")
        item.update(node)
        hydrated.append(item)
    return hydrated


def _record_access_by_namespace(
    result_dicts: list[dict[str, object]],
    backend: StorageBackend,
    namespace: str,
    namespace_backend_factory: Callable[[str], StorageBackend] | None,
) -> None:
    """Persist access metadata for returned entries across namespace stores."""
    grouped: dict[str, list[str]] = {}
    for result in result_dicts:
        if "id" not in result:
            continue
        result_namespace = str(result.get("namespace", namespace))
        grouped.setdefault(result_namespace, []).append(str(result["id"]))

    if not grouped:
        return

    with ExitStack() as stack:
        for result_namespace, ids in grouped.items():
            target_backend = backend
            if result_namespace != namespace:
                if namespace_backend_factory is None:
                    continue
                target_backend = stack.enter_context(namespace_backend_factory(result_namespace))
            record_recall_access(target_backend, ids)
