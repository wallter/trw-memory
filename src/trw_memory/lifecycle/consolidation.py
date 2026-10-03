"""Memory consolidation engine for trw-memory.

Clusters semantically similar memory entries using embeddings and
complete-linkage agglomerative clustering, then consolidates each cluster
into a single entry via LLM summarization (with a longest-entry fallback).
Original entries are archived after consolidation.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar
from uuid import uuid4

import structlog

from trw_memory.embeddings._similarity_calibration import calibrated_threshold
from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.embeddings.provenance import generation_provenance_kwargs
from trw_memory.exceptions import DimensionMismatchError, StorageError
from trw_memory.graph import schedule_graph_update
from trw_memory.labels import LabelPolicy, Sink
from trw_memory.lifecycle._consolidated_fields import merged_entry_fields
from trw_memory.lifecycle._consolidation_metrics import mean_pairwise_similarity as _mean_pairwise_similarity

# Archive / restore / rollback live in the ``_consolidation_rollback`` sibling
# (extracted for the 350 eLOC gate). Re-exported so `consolidate_cycle` below and
# tests/test_consolidation_helpers.py keep one import point.
from trw_memory.lifecycle._consolidation_rollback import (
    _archive_originals as _archive_originals,
)
from trw_memory.lifecycle._consolidation_rollback import (
    _restore_originals as _restore_originals,
)
from trw_memory.lifecycle._consolidation_rollback import (
    _rollback_consolidation as _rollback_consolidation,
)
from trw_memory.lifecycle._redaction import redact_paths
from trw_memory.lifecycle.protection import is_removal_exempt
from trw_memory.models.config import MemoryConfig
from trw_memory.models.entry_factory import local_node_id_for, new_entry
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.storage._connection import outside_lane_deadline
from trw_memory.storage._shared import revision_of
from trw_memory.storage.interface import StorageBackend

# Compatibility re-export: consumers import ``_redact_paths`` from this facade.
# A module-level assignment (vs. an aliased import) makes the name an explicitly
# defined attribute under ``mypy --strict`` for downstream ``from … import``.
_redact_paths = redact_paths

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared clustering algorithm — used by both trw-memory and trw-mcp
# ---------------------------------------------------------------------------

T = TypeVar("T")

#: The most rows one consolidation cycle reads, embeds and clusters. ``consolidation_max_per_cycle``
#: (the daemon's config, or a caller's policy on ``memory_maintain``) can only lower it. On the daemon
#: the read, embed and cluster run off the write lane; each cluster's writes are one lane job (B71-89).
CONSOLIDATION_ROWS_MAX = 50

#: The largest cluster one cycle merges. A bigger one is skipped with a warning: one merge of many
#: entries is the recall-poisoning path PRD-FIX-114 closed (an entry matching every query).
CONSOLIDATION_CLUSTER_MAX = 10


#: What one cluster's write step (:func:`_write_cluster`) reports, besides a failure's message.
_WRITTEN, _SKIPPED = "written", "skipped"
#: One cluster's write step, run on *storage* by the caller's lane (``consolidate_cycle(lane=...)``).
ClusterWrite = Callable[[StorageBackend], str]


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

    def _pair_similarity(a: list[float], b: list[float]) -> float:
        # Mixed-dimension stores (e.g. after an embedding-model change) must not
        # abort the whole consolidation cycle: a dimension-mismatched pair simply
        # cannot belong to the same cluster, so it scores 0.0 instead of raising.
        try:
            return similarity_fn(a, b)
        except (DimensionMismatchError, ZeroDivisionError):
            return 0.0

    n = len(items)
    cluster_id: list[int] = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            sim = _pair_similarity(items[i][1], items[j][1])
            if sim >= similarity_threshold:
                cid_i = cluster_id[i]
                cid_j = cluster_id[j]
                if cid_i == cid_j:
                    continue
                # Check that ALL pairs between the two clusters satisfy threshold
                i_members = [k for k in range(n) if cluster_id[k] == cid_i]
                j_members = [k for k in range(n) if cluster_id[k] == cid_j]
                can_merge = all(
                    _pair_similarity(items[a][1], items[b][1]) >= similarity_threshold
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
    max_entries: int = CONSOLIDATION_ROWS_MAX,
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

    # Filter out already-consolidated entries and entries already archived. A cluster's
    # members are all archived, so neither a protected or permanent entry (PRD-CORE-244
    # FR10) nor a security canary row ever joins one; trw-mcp enforced both before
    # PRD-CORE-302 FR03 made this the only consolidation.
    entries = [
        e
        for e in entries
        if e.source != "consolidated"
        and e.consolidated_into is None
        and not is_removal_exempt({"protection_tier": e.protection_tier})
        and e.metadata.get("system_canary") != "true"
    ]
    # PRD-SEC-023 FR07: a row above team never joins a cluster, so neither the consolidated row, its tags nor an LLM summary
    # (nor the embedder) ever sees it. Team is the platform sink's clearance.
    entries = LabelPolicy.current().admit(entries, Sink.PLATFORM).admitted

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
        calibrated_threshold(similarity_threshold, embedder),
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
    embedding: list[float] | None = None,
) -> MemoryEntry:
    """Create a new consolidated memory entry from a cluster.

    Derives the consolidated entry's fields from the cluster:
    - importance: max of cluster
    - tags: the ``CONSOLIDATION_TAGS_MAX`` most frequent tags across the cluster, sorted
    - evidence: union of all evidence (deduplicated)
    - recurrence: the cluster size (one recurrence per merged entry, not the sum of their counts)

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

    now = datetime.now(timezone.utc)
    entry = new_entry(
        entry_id=entry_id,
        content=content,
        namespace=namespace,
        local_node_id=local_node_id_for(namespace),
        now=now,
        fields={"detail": detail, **merged_entry_fields(cluster)},
    )

    # Computed before the write transaction (pure CPU: a failure writes nothing), unless the caller
    # already computed it off the daemon's write lane (B71-89).
    if embedding is None:
        embedding = _entry_vector(embedder, storage, repr(entry.id), f"{entry.content} {entry.detail}")
    # S1-parity fix: commit the row + its vector in ONE transaction so a crash
    # between the two writes can no longer leave a row with no vector, and a
    # vector failure rolls the row back automatically. This matches
    # MemoryClient.store() / memory_store_impl instead of the older
    # compensating-delete path, giving every store seam one atomicity model.
    try:
        with storage.transaction():
            storage.store(entry)
            if embedding is not None:
                storage.upsert_vector(
                    entry.id,
                    embedding,
                    namespace=entry.namespace,
                    **generation_provenance_kwargs(embedder, f"{entry.content} {entry.detail}", embedding),
                )
    except Exception as exc:
        raise StorageError(f"failed to persist entry+vector for {entry.id!r}; transaction rolled back") from exc
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


def _entry_vector(
    embedder: EmbeddingProvider | None, storage: StorageBackend, label: str, text: str
) -> list[float] | None:
    """*text*'s vector, or ``None`` when nothing would store it (no embedder, or no vector store:
    graph similarity reads vectors back from that store too)."""
    if embedder is None or not embedder.available() or not storage.supports_vectors():
        return None
    try:
        return embedder.embed(text)
    except Exception as exc:
        raise StorageError(f"failed to compute embedding for {label}; entry was not written") from exc


def _write_cluster(
    cluster: list[MemoryEntry],
    chosen: dict[str, str],
    vector: list[float] | None,
    embedder: EmbeddingProvider | None,
    namespace: str,
    storage: StorageBackend,
) -> str:
    """Consolidate *cluster* on *storage*, as one job: re-read its rows and skip it when any changed
    since it was clustered (B71-89: the cluster was read off the lane, compared by ``revision_of`` --
    PRD-CORE-308's content digest, not the old ``(status, consolidated_into, updated_at)`` tuple a
    content change holding ``updated_at`` constant slipped past unnoticed, B71-135 d); else create
    the consolidated entry and archive the rows, rolled back on failure. Returns ``_WRITTEN``,
    ``_SKIPPED`` or the failure."""
    cluster_ids = [e.id for e in cluster]
    fresh = [storage.get(e.id, namespace=e.namespace) for e in cluster]
    current = [now for now, then in zip(fresh, cluster, strict=True) if now and revision_of(now) == revision_of(then)]
    if len(current) < len(cluster):
        logger.info("consolidation_cluster_changed", cluster_ids=cluster_ids)
        return _SKIPPED
    new_entry: MemoryEntry | None = None
    try:
        new_entry = _create_consolidated_entry(
            current, chosen["summary"], chosen["detail"], storage, embedder, namespace, embedding=vector
        )
        # FR04: archive the originals and close their validity windows at the consolidated entry's
        # valid_from (gap-free; OQ3 consolidation instant).
        _archive_originals(current, new_entry.id, storage, invalid_from=new_entry.valid_from)
    except Exception as exc:  # broad catch: per-cluster error boundary
        if new_entry is not None:
            try:
                with outside_lane_deadline():  # a half-applied cluster is undone whatever the job's clock says
                    _rollback_consolidation(current, new_entry, storage)
            except Exception as rollback_exc:
                logger.exception("consolidation_rollback_failed", cluster_ids=cluster_ids, consolidated_id=new_entry.id)
                raise StorageError(
                    f"consolidation rollback failed for cluster {cluster_ids}: {rollback_exc}"
                ) from rollback_exc
        logger.exception("consolidation_cluster_failed", cluster_ids=cluster_ids, error=str(exc))
        return f"cluster {cluster_ids}: {exc}"
    return _WRITTEN


# ---------------------------------------------------------------------------
# FR04 — Original Entry Archival
# ---------------------------------------------------------------------------


def consolidate_cycle(
    storage: StorageBackend,
    embedder: EmbeddingProvider | None = None,
    *,
    max_entries: int = CONSOLIDATION_ROWS_MAX,
    dry_run: bool = False,
    namespace: str | None = None,
    config: MemoryConfig | None = None,
    lane: Callable[[ClusterWrite], str] | None = None,
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
        lane: Runs one cluster's write step on a backend of its own (the daemon's write lane);
            ``None`` runs it on *storage*. Everything else reads *storage*.

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
        except Exception as exc:
            # Fail closed: if we cannot enumerate namespaces we cannot prove the
            # store is single-tenant, so refuse the ambiguous namespace=None path
            # rather than risk clustering + relocating other tenants' entries into
            # "default". A transient enumeration failure skips one maintenance
            # cycle, which is safe; a silent cross-tenant merge is not.
            logger.warning("consolidation_list_namespaces_failed", exc_info=True)
            raise ValueError(
                "namespace required for consolidate_cycle: could not enumerate "
                "namespaces to verify the store is single-tenant"
            ) from exc
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
            "clusters_found": len(clusters),
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
    write: Callable[[ClusterWrite], str] = lane or (lambda step: step(storage))
    outcomes: list[str] = []
    for cluster in clusters:
        if len(cluster) > CONSOLIDATION_CLUSTER_MAX:
            logger.warning(
                "consolidation_cluster_skipped",
                cluster_size=len(cluster),
                max_cluster_size=CONSOLIDATION_CLUSTER_MAX,
                entry_ids=[e.id for e in cluster],
            )
            outcomes.append(_SKIPPED)
            continue
        # FR02/FR05: longest-entry selection (the LLM summarization hook point); its vector is computed
        # here, off the daemon's write lane, and only the cluster's writes take the lane (B71-89).
        chosen = _summarize_cluster_fallback(cluster)
        try:
            vector = _entry_vector(embedder, storage, "a consolidated entry", f"{chosen['summary']} {chosen['detail']}")
        except StorageError as exc:
            outcomes.append(f"cluster {[e.id for e in cluster]}: {exc}")
            continue
        outcomes.append(write(functools.partial(_write_cluster, cluster, chosen, vector, embedder, ns)))
    consolidated_count = outcomes.count(_WRITTEN)
    errors = [outcome for outcome in outcomes if outcome not in (_WRITTEN, _SKIPPED)]

    result: dict[str, object] = {
        "status": "completed",
        "clusters_found": len(clusters),
        "consolidated_count": consolidated_count,
    }
    if skipped := outcomes.count(_SKIPPED):
        result["clusters_skipped"] = skipped
    if errors:
        result["errors"] = errors

    logger.info(
        "consolidation_cycle_complete",
        clusters_found=len(clusters),
        consolidated_count=consolidated_count,
        errors=len(errors),
    )
    return result
