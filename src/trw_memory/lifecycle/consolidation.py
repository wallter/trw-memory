"""Memory consolidation engine for trw-memory.

Clusters semantically similar memory entries using embeddings and
complete-linkage agglomerative clustering, then consolidates each cluster
into a single entry via LLM summarization (with a longest-entry fallback).
Original entries are archived after consolidation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar
from uuid import uuid4

import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.exceptions import StorageError
from trw_memory.graph import schedule_graph_update
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

# NFR06 — Path redaction pattern for LLM prompts
_PATH_RE = re.compile(
    r"(?:/home/|/Users/|/mnt/|/tmp/|/var/|[A-Z]:\\)[^\s,;\"')\]}>]*",
)


def _redact_paths(text: str) -> str:
    """Replace filesystem paths with [REDACTED_PATH] before sending to LLM."""
    return _PATH_RE.sub("[REDACTED_PATH]", text)


# ---------------------------------------------------------------------------
# Shared clustering algorithm — used by both trw-memory and trw-mcp
# ---------------------------------------------------------------------------

T = TypeVar("T")


def complete_linkage_cluster(
    items: list[tuple[T, list[float]]],
    similarity_threshold: float,
    min_cluster_size: int,
    similarity_fn: Callable[[list[float], list[float]], float] | None = None,
) -> list[list[T]]:
    """Complete-linkage agglomerative clustering on (item, vector) pairs.

    Two items belong to the same cluster when every pair in the group has
    cosine similarity >= *similarity_threshold*.

    This is the shared algorithm extracted so both trw-memory and trw-mcp
    use a single canonical implementation.

    Args:
        items: List of (item, embedding_vector) tuples.
        similarity_threshold: Minimum pairwise similarity to merge into cluster.
        min_cluster_size: Clusters smaller than this are discarded.
        similarity_fn: Cosine similarity function. Defaults to
            :func:`trw_memory.retrieval.dense.cosine_similarity`.

    Returns:
        List of clusters; each cluster is a list of items (first element of
        each tuple). Clusters smaller than *min_cluster_size* are excluded.
    """
    if similarity_fn is None:
        similarity_fn = cosine_similarity

    n = len(items)
    cluster_id: list[int] = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_fn(items[i][1], items[j][1])
            if sim >= similarity_threshold:
                cid_i = cluster_id[i]
                cid_j = cluster_id[j]
                if cid_i == cid_j:
                    continue
                # Check that ALL pairs between the two clusters satisfy threshold
                i_members = [k for k in range(n) if cluster_id[k] == cid_i]
                j_members = [k for k in range(n) if cluster_id[k] == cid_j]
                can_merge = all(
                    similarity_fn(items[a][1], items[b][1]) >= similarity_threshold
                    for a in i_members
                    for b in j_members
                )
                if can_merge:
                    for k in range(n):
                        if cluster_id[k] == cid_j:
                            cluster_id[k] = cid_i

    # Collect clusters by cluster_id
    clusters_map: dict[int, list[T]] = {}
    for idx, cid in enumerate(cluster_id):
        clusters_map.setdefault(cid, []).append(items[idx][0])

    return [cluster for cluster in clusters_map.values() if len(cluster) >= min_cluster_size]


# ---------------------------------------------------------------------------
# FR01 — Embedding-Based Cluster Detection
# ---------------------------------------------------------------------------


def find_clusters(
    storage: StorageBackend,
    embedder: EmbeddingProvider | None = None,
    *,
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 3,
    max_entries: int = 50,
    namespace: str | None = None,
) -> list[list[MemoryEntry]]:
    """Detect clusters of semantically similar active memory entries.

    Loads up to *max_entries* active entries from *storage*, generates
    embeddings in a single batch call, then applies complete-linkage
    agglomerative clustering: two entries belong to the same cluster when
    every pair in the group has cosine similarity >= *similarity_threshold*.

    Args:
        storage: StorageBackend to load entries from.
        embedder: EmbeddingProvider for generating vectors.
        similarity_threshold: Minimum pairwise similarity to merge into cluster.
        min_cluster_size: Clusters smaller than this are discarded.
        max_entries: Cap on number of entries loaded.
        namespace: If provided, restrict to this namespace.

    Returns:
        List of clusters; each cluster is a list of MemoryEntry objects.
        Returns [] when embeddings are unavailable.
    """
    if embedder is None or not embedder.available():
        logger.debug("consolidation_embed_unavailable")
        return []

    # Load active entries (capped)
    entries = storage.list_entries(
        status=MemoryStatus.ACTIVE,
        namespace=namespace,
        limit=max_entries,
    )

    # Filter out already-consolidated entries and entries already archived
    entries = [e for e in entries if e.source != "consolidated" and e.consolidated_into is None]

    if len(entries) < min_cluster_size:
        return []

    # Batch embed all entries in one call (FR01 requirement)
    texts = [e.content + " " + e.detail for e in entries]
    vectors = embedder.embed_batch(texts)

    # Build (entry, vector) pairs, dropping entries with no embedding
    indexed: list[tuple[MemoryEntry, list[float]]] = []
    for i, vec in enumerate(vectors):
        if vec is not None:
            indexed.append((entries[i], vec))

    if len(indexed) < min_cluster_size:
        return []

    return complete_linkage_cluster(
        indexed,
        similarity_threshold,
        min_cluster_size,
    )


# ---------------------------------------------------------------------------
# FR02/FR05 — Cluster Summarization (longest-entry selection)
# Future: LLM summarization hook point — see consolidation design docs
# ---------------------------------------------------------------------------


def _summarize_cluster_fallback(
    cluster: list[MemoryEntry],
) -> dict[str, str]:
    """Select the longest-content entry as the consolidated summary/detail.

    Used when LLM is unavailable or summarization fails.
    Logs at INFO level with cluster_size.

    Args:
        cluster: List of MemoryEntry objects in the cluster.

    Returns:
        Dict with "summary" (content) and "detail" from the best entry.
    """
    best = max(
        cluster,
        key=lambda e: len(e.content) + len(e.detail),
    )
    logger.info(
        "consolidation_llm_fallback",
        cluster_size=len(cluster),
        selected_id=best.id,
    )
    return {
        "summary": best.content,
        "detail": best.detail,
    }


# ---------------------------------------------------------------------------
# FR03 — Consolidated Entry Creation
# ---------------------------------------------------------------------------


def _create_consolidated_entry(
    cluster: list[MemoryEntry],
    content: str,
    detail: str,
    storage: StorageBackend,
    embedder: EmbeddingProvider | None = None,
    namespace: str = "default",
) -> MemoryEntry:
    """Create a new consolidated memory entry from a cluster.

    Derives the consolidated entry's fields from the cluster:
    - importance: max of cluster
    - tags: sorted union of all tags
    - evidence: union of all evidence (deduplicated)
    - recurrence: sum of cluster recurrences
    - q_value: max of cluster q_values

    Writes the entry via storage.store().

    Args:
        cluster: List of MemoryEntry objects being consolidated.
        content: Consolidated content text.
        detail: Consolidated detail text.
        storage: StorageBackend for persisting the new entry.
        namespace: Namespace for the new consolidated entry.

    Returns:
        The new consolidated MemoryEntry.
    """
    entry_id = "M-" + uuid4().hex

    # Inherit provenance from highest-importance source (PRD-CORE-099)
    best_source = max(cluster, key=lambda e: e.importance)

    now = datetime.now(timezone.utc)
    entry = MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        source="consolidated",
        source_identity=best_source.source_identity,
        client_profile=best_source.client_profile,
        model_id=best_source.model_id,
        consolidated_from=[e.id for e in cluster],
        importance=max(e.importance for e in cluster),
        tags=sorted({t for e in cluster for t in e.tags}),
        evidence=list(dict.fromkeys(ev for e in cluster for ev in e.evidence)),
        recurrence=sum(e.recurrence for e in cluster),
        q_value=max(e.q_value for e in cluster),
        status=MemoryStatus.ACTIVE,
        namespace=namespace,
        created_at=now,
        updated_at=now,
    )

    # Compute the embedding before opening the write transaction — pure CPU work
    # with no DB state, so a failure here must leave nothing written. The only
    # sinks for a consolidated entry's vector are the backend vector store and
    # graph similarity (which reads candidate vectors back from that store), so
    # both collapse to ``supports_vectors``; skip the embed when it is False.
    embedding: list[float] | None = None
    if embedder is not None and embedder.available() and storage.supports_vectors():
        try:
            embedding = embedder.embed(f"{entry.content} {entry.detail}")
        except Exception as exc:
            raise StorageError(
                f"failed to compute embedding for {entry.id!r}; entry was not written"
            ) from exc
    # S1-parity fix: commit the row + its vector in ONE transaction so a crash
    # between the two writes can no longer leave a row with no vector, and a
    # vector failure rolls the row back automatically. This matches
    # MemoryClient.store() / memory_store_impl instead of the older
    # compensating-delete path, giving every store seam one atomicity model.
    try:
        with storage.transaction():
            storage.store(entry)
            if embedding is not None:
                storage.upsert_vector(entry.id, embedding)
    except Exception as exc:
        raise StorageError(
            f"failed to persist entry+vector for {entry.id!r}; transaction rolled back"
        ) from exc
    try:
        # Consolidation lineage edges are secondary structure and should not keep
        # the consolidated entry itself on the write path.
        schedule_graph_update(
            entry,
            storage,
            embedding=embedding,
            config=getattr(storage, "_config", None),
        )
    except RuntimeError:
        logger.warning("consolidation_graph_schedule_failed", entry_id=entry.id, exc_info=True)

    logger.info(
        "consolidation_entry_created",
        entry_id=entry_id,
        cluster_size=len(cluster),
        consolidated_from=entry.consolidated_from,
    )
    return entry


# ---------------------------------------------------------------------------
# FR04 — Original Entry Archival
# ---------------------------------------------------------------------------


def _archive_originals(
    cluster: list[MemoryEntry],
    consolidated_id: str,
    storage: StorageBackend,
) -> None:
    """Archive original cluster entries after consolidation.

    For each entry in *cluster*:
    1. Sets ``consolidated_into`` to the consolidated entry's ID.
    2. Sets ``status`` to ``"archived"``.
    3. Updates via storage.update().

    On failure, logs ERROR and raises the exception (caller handles rollback).

    S4 fix: all per-entry archival updates run inside ONE ``storage.transaction()``
    so a crash mid-loop can never leave a cluster partially archived — either every
    original gets ``status=archived`` + ``consolidated_into`` or none do. The
    interface default ``transaction()`` is a no-op pass-through, so YAML/other
    backends keep their prior per-call-commit behaviour.

    Args:
        cluster: Original MemoryEntry objects being archived.
        consolidated_id: ID of the newly created consolidated entry.
        storage: StorageBackend for updating entries.
    """
    processed: list[str] = []

    with storage.transaction():
        for entry in cluster:
            try:
                updated = storage.update(
                    entry.id,
                    consolidated_into=consolidated_id,
                    status=MemoryStatus.ARCHIVED,
                    updated_at=datetime.now(timezone.utc),
                )
                if updated is None:
                    raise StorageError(f"failed to archive original entry {entry.id!r}")
                processed.append(entry.id)
            except (
                StorageError,
                ValueError,
                RuntimeError,
            ) as exc:  # per-item error handling: re-raise but log each failure individually
                logger.exception(
                    "consolidation_archive_failed",
                    entry_id=entry.id,
                    consolidated_id=consolidated_id,
                    error=str(exc),
                )
                raise

    logger.info(
        "consolidation_archive_complete",
        consolidated_id=consolidated_id,
        archived_count=len(processed),
    )


# ---------------------------------------------------------------------------
# FR06 — Dry-Run Mode + Helper
# ---------------------------------------------------------------------------


def _mean_pairwise_similarity(
    cluster: list[MemoryEntry],
    embedder: EmbeddingProvider,
) -> float:
    """Compute mean pairwise cosine similarity for dry-run preview.

    Returns 0.0 when cluster is too small or embeddings unavailable.
    """
    if len(cluster) < 2:
        return 0.0

    texts = [e.content + " " + e.detail for e in cluster]
    vectors = embedder.embed_batch(texts)
    valid: list[list[float]] = [v for v in vectors if v is not None]
    if len(valid) < 2:
        return 0.0

    pairs = [cosine_similarity(valid[i], valid[j]) for i in range(len(valid)) for j in range(i + 1, len(valid))]
    return sum(pairs) / len(pairs) if pairs else 0.0


def _restore_originals(
    cluster: list[MemoryEntry],
    storage: StorageBackend,
) -> None:
    """Restore original entries after a failed consolidation attempt."""
    for entry in cluster:
        storage.store(entry)


def _rollback_consolidation(
    cluster: list[MemoryEntry],
    new_entry: MemoryEntry,
    storage: StorageBackend,
) -> None:
    """Undo a partially applied consolidation so callers keep original data.

    Consolidation creates a brand-new semantic memory and then mutates the
    originals. If archival fails after the new entry is written, leaving both
    sides in place would duplicate knowledge and silently mark only part of the
    cluster as archived. Roll back to the pre-cycle state instead.

    Restoring the originals to ACTIVE is the safety-critical half and must run
    even if deleting the new consolidated entry fails: on a YAML backend
    ``_archive_originals``' ``transaction()`` is a no-op, so a mid-loop failure
    can leave some originals already ``status=archived`` + ``consolidated_into``
    set while the consolidated entry survives. Restore the originals FIRST
    (idempotent ``store`` of their pre-cycle snapshots), then surface any
    new-entry delete failure. This guarantees a partial consolidation never
    leaves originals archived alongside a surviving consolidated entry.
    """
    deleted = storage.delete(new_entry.id)
    # Always reinstate the originals, regardless of the delete outcome, so a
    # failed delete cannot strand them in the archived state.
    _restore_originals(cluster, storage)
    if not deleted:
        raise StorageError(f"failed to delete partially consolidated entry {new_entry.id!r}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def consolidate_cycle(
    storage: StorageBackend,
    embedder: EmbeddingProvider | None = None,
    *,
    max_entries: int = 50,
    dry_run: bool = False,
    namespace: str | None = None,
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    """Run one consolidation cycle across all active memory entries.

    Steps:
    1. Detect clusters via embedding similarity (FR01).
    2. In dry-run mode: return cluster summary without writes (FR06).
    3. For each cluster: summarize via LLM (FR02, stub) or fallback (FR05).
    4. Create consolidated entry (FR03).
    5. Archive originals (FR04).

    Args:
        storage: StorageBackend to read/write entries.
        embedder: EmbeddingProvider for generating vectors.
        max_entries: Maximum entries to consider for clustering.
        dry_run: If True, skip writes and return cluster preview.
        namespace: If provided, restrict consolidation to this namespace.
        config: MemoryConfig with consolidation thresholds.

    Returns:
        Dict with consolidation results including cluster count and
        consolidated_count. In dry_run mode: {dry_run: true, clusters: [...],
        consolidated_count: 0}.
    """
    cfg = config or MemoryConfig()
    entry_limit = min(max_entries, cfg.consolidation_max_per_cycle)

    # Cross-tenant safety: with namespace=None, find_clusters loads entries
    # across ALL namespaces and _create_consolidated_entry would persist the
    # merged result into a single namespace ("default"), leaking and relocating
    # other tenants' knowledge. Refuse the ambiguous path on multi-namespace
    # stores; callers that genuinely want a specific tenant already pass it.
    if namespace is None:
        try:
            existing_namespaces = storage.list_namespaces()
        except Exception:
            logger.warning("consolidation_list_namespaces_failed", exc_info=True)
            existing_namespaces = []
        if len(existing_namespaces) > 1:
            raise ValueError(
                "namespace required for consolidate_cycle on multi-tenant stores: "
                f"found {len(existing_namespaces)} namespaces; pass an explicit namespace "
                "to avoid clustering entries across tenants"
            )

    if not cfg.consolidation_enabled and not dry_run:
        return {
            "status": "disabled",
            "clusters_found": 0,
            "consolidated_count": 0,
            "skipped_reason": "consolidation_disabled",
        }

    clusters = find_clusters(
        storage,
        embedder,
        similarity_threshold=cfg.consolidation_similarity_threshold,
        min_cluster_size=cfg.consolidation_min_cluster,
        max_entries=entry_limit,
        namespace=namespace,
    )

    if dry_run:
        cluster_previews: list[dict[str, object]] = []
        for cluster in clusters:
            entry_ids = [e.id for e in cluster]
            mean_sim = 0.0
            if embedder is not None and embedder.available():
                mean_sim = _mean_pairwise_similarity(cluster, embedder)
            cluster_previews.append(
                {
                    "entry_ids": entry_ids,
                    "count": len(cluster),
                    "mean_similarity": round(mean_sim, 3),
                }
            )
        return {
            "dry_run": True,
            "clusters": cluster_previews,
            "consolidated_count": 0,
            "skipped_reason": "dry_run",
        }

    if not clusters:
        return {
            "status": "no_clusters",
            "clusters_found": 0,
            "consolidated_count": 0,
        }

    ns = namespace or "default"
    consolidated_count = 0
    errors: list[str] = []

    for cluster in clusters:
        cluster_ids = [e.id for e in cluster]
        new_entry: MemoryEntry | None = None
        try:
            # FR02/FR05: Cluster summarization (longest-entry selection)
            # Future: LLM summarization hook point — see consolidation design docs
            fallback = _summarize_cluster_fallback(cluster)
            content = fallback["summary"]
            detail = fallback["detail"]

            # FR03: Create consolidated entry
            new_entry = _create_consolidated_entry(
                cluster,
                content,
                detail,
                storage,
                embedder=embedder,
                namespace=ns,
            )
            consolidated_id = new_entry.id

            # FR04: Archive originals
            _archive_originals(cluster, consolidated_id, storage)
            consolidated_count += 1

        except Exception as exc:  # broad catch: per-cluster error boundary
            if new_entry is not None:
                try:
                    _rollback_consolidation(cluster, new_entry, storage)
                except Exception as rollback_exc:
                    logger.exception(
                        "consolidation_rollback_failed",
                        cluster_ids=cluster_ids,
                        consolidated_id=new_entry.id,
                    )
                    raise StorageError(
                        f"consolidation rollback failed for cluster {cluster_ids}: {rollback_exc}"
                    ) from rollback_exc
            logger.exception(
                "consolidation_cluster_failed",
                cluster_ids=cluster_ids,
                error=str(exc),
            )
            errors.append(f"cluster {cluster_ids}: {exc}")

    result: dict[str, object] = {
        "status": "completed",
        "clusters_found": len(clusters),
        "consolidated_count": consolidated_count,
    }
    if errors:
        result["errors"] = errors

    logger.info(
        "consolidation_cycle_complete",
        clusters_found=len(clusters),
        consolidated_count=consolidated_count,
        errors=len(errors),
    )
    return result
