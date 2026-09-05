"""Remote fetch helpers for memory sync.

A fetch has more outcomes than "here are the results". Sync switched off, an
unusable platform URL, a 401 on a rotated key, a body that did not parse, and a
peer that returned ten items the admission gate refused ALL previously left this
module by the same door: an empty list, identical to "the shared corpus held
nothing for this query". :class:`SharedFetchResult` keeps them apart so a caller
can say which one happened instead of reporting silence as absence.
"""

from __future__ import annotations

import json
from typing import Literal, NamedTuple

import httpx
import structlog

from trw_memory.embeddings.interface import EmbeddingProvider
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.security.pii import mask_query_credentials
from trw_memory.storage.interface import StorageBackend
from trw_memory.sync._remote_admission import admit_remote_results
from trw_memory.sync._remote_common import (
    FETCH_TIMEOUT,
    _raise_local_only_violation,
    build_platform_headers,
    decode_learning_api_v1_result,
    encode_learning_api_v1_search,
    is_valid_platform_url,
)

logger = structlog.get_logger(__name__)

#: Why a fetch returned what it returned.
#:
#: ``ok``
#:     The platform answered and every returned item was admitted.
#: ``partial``
#:     The platform answered and the gate refused some of what it returned.
#: ``refused``
#:     The platform returned items and the gate refused every one of them --
#:     an empty result that is NOT an empty corpus.
#: ``disabled``
#:     Sync is off or no platform is configured. Nothing was asked.
#: ``invalid_config``
#:     A platform URL that cannot be fetched from. Nothing was asked.
#: ``fetch_failed``
#:     The platform was asked and did not usefully answer (non-200, unparseable
#:     body, transport error).
FetchStatus = Literal["ok", "partial", "refused", "disabled", "invalid_config", "fetch_failed"]


class SharedFetchResult(NamedTuple):
    """Admitted shared results plus what happened to everything else."""

    results: list[dict[str, object]]
    status: FetchStatus
    #: Items the platform returned, before dedup and before admission.
    fetched: int
    #: Items the admission gate refused (see
    #: :class:`~trw_memory.sync._remote_admission.AdmissionOutcome`).
    refused: int

    def __bool__(self) -> bool:
        """Truthy exactly when results were returned, as the old list was."""
        return bool(self.results)


def _dedupe_shared_results(
    results: list[dict[str, object]],
    *,
    local_entries: list[MemoryEntry],
    embedder: EmbeddingProvider | None,
    dedup_threshold: float,
) -> list[dict[str, object]]:
    if not local_entries:
        return results

    local_ids = {entry.id for entry in local_entries}
    local_contents = {entry.content.lower().strip() for entry in local_entries}
    filtered: list[dict[str, object]] = []
    remote_texts: list[str] = []
    remote_candidates: list[dict[str, object]] = []

    for result in results:
        source_learning_id = str(result.get("source_learning_id", "")).strip()
        summary = str(result.get("summary", result.get("content", ""))).strip()
        detail = str(result.get("detail", "")).strip()
        if source_learning_id and source_learning_id in local_ids:
            continue
        if summary.lower() in local_contents:
            continue
        remote_candidates.append(result)
        remote_texts.append(f"{summary} {detail}".strip())

    if not remote_candidates or embedder is None:
        return remote_candidates

    try:
        if not embedder.available():
            return remote_candidates

        local_texts = [f"{entry.content} {entry.detail}".strip() for entry in local_entries]
        texts = [*local_texts, *remote_texts]
        vectors = embedder.embed_batch(texts)
        if len(vectors) != len(texts):
            raise ValueError(f"embedding batch length mismatch: expected {len(texts)}, got {len(vectors)}")
        local_vectors = [vector for vector in vectors[: len(local_entries)] if vector is not None]
        remote_vectors = vectors[len(local_entries) :]
        if not local_vectors:
            return remote_candidates

        for result, remote_vector in zip(remote_candidates, remote_vectors, strict=False):
            if remote_vector is None:
                filtered.append(result)
                continue
            if any(cosine_similarity(remote_vector, local_vector) > dedup_threshold for local_vector in local_vectors):
                continue
            filtered.append(result)
    except Exception:
        logger.warning("memory_shared_semantic_dedup_failed", exc_info=True)
        return remote_candidates
    return filtered


def fetch_shared_memories(
    query: str,
    cfg: MemoryConfig,
    *,
    backend: StorageBackend,
    embedding: list[float] | None = None,
    limit: int = 10,
    local_entries: list[MemoryEntry] | None = None,
    embedder: EmbeddingProvider | None = None,
    dedup_threshold: float = 0.92,
) -> SharedFetchResult:
    """Fetch shared memories from the platform, admitting only what the gate passes.

    ``backend`` is required, not optional: it is what the admission gate needs to
    evaluate a candidate, and a fetch that cannot be gated must not happen at all
    (PRD-CORE-245 FR06, NFR03 fail-closed). This is the ONE path to
    ``/v1/learnings/search`` in either package; the duplicate client in
    ``trw_mcp.telemetry.remote_recall`` was deleted with the same change.

    Returns:
        A :class:`SharedFetchResult`. ``results`` is what a caller merges;
        ``status`` says whether an empty ``results`` means "nothing matched",
        "nothing was asked", "the platform did not answer" or "everything was
        refused".
    """
    if cfg.local_only:
        logger.warning("memory_fetch_blocked_local_only")
        _raise_local_only_violation()
    if not cfg.sync_enabled or not cfg.platform_url:
        return SharedFetchResult([], "disabled", 0, 0)
    if not is_valid_platform_url(cfg.platform_url):
        logger.warning("memory_fetch_invalid_platform_url")
        return SharedFetchResult([], "invalid_config", 0, 0)

    request_payload: dict[str, object] = encode_learning_api_v1_search(
        query=mask_query_credentials(query), limit=limit, min_importance=cfg.sync_min_importance
    )
    if embedding:
        request_payload["embedding"] = embedding

    try:
        with httpx.Client(timeout=FETCH_TIMEOUT) as client:
            resp = client.post(
                f"{cfg.platform_url.rstrip('/')}/v1/learnings/search",
                json=request_payload,
                headers=build_platform_headers(cfg.platform_api_key),
            )
            # A failed fetch and an empty shared corpus both arrive as ``[]``, and
            # the caller merges either one identically. Log the failure (status
            # only — never the body) so "the platform is 401ing on a rotated key"
            # is distinguishable from "nothing matched"; absence of a measurement
            # is not a measurement of absence.
            if resp.status_code != 200:
                logger.warning("memory_fetch_rejected", status_code=resp.status_code)
                return SharedFetchResult([], "fetch_failed", 0, 0)
            raw = resp.json()
            if isinstance(raw, list):
                raw_results = [item for item in raw if isinstance(item, dict)]
            elif isinstance(raw, dict):
                wrapped = raw.get("results", raw.get("items", []))
                raw_results = [item for item in wrapped if isinstance(item, dict)] if isinstance(wrapped, list) else []
            else:
                logger.warning("memory_fetch_malformed_body", body_type=type(raw).__name__)
                return SharedFetchResult([], "fetch_failed", 0, 0)
            # Decode external wire vocabulary into canonical importance at the boundary.
            results = [decode_learning_api_v1_result(item) for item in raw_results]
    except (httpx.HTTPError, OSError, ConnectionError, json.JSONDecodeError, ValueError):
        logger.debug("memory_fetch_error", exc_info=True)
        return SharedFetchResult([], "fetch_failed", 0, 0)

    deduped = _dedupe_shared_results(
        results,
        local_entries=local_entries or [],
        embedder=embedder,
        dedup_threshold=dedup_threshold,
    )
    # PRD-CORE-245 FR06: the admission gate runs BEFORE the [shared] prefix and
    # before anything is returned, so a refused item never reaches the recall
    # response, and therefore never reaches an agent's context.
    outcome = admit_remote_results(deduped, config=cfg, backend=backend)
    shared: list[dict[str, object]] = []
    for result in outcome.admitted:
        summary = result.get("summary", result.get("content", ""))
        content = str(summary) if summary is not None else ""
        result["summary"] = f"[shared] {content}"
        result["content"] = f"[shared] {content}"
        result["source"] = "shared"
        shared.append(result)

    if outcome.refused and not shared:
        status: FetchStatus = "refused"
    elif outcome.refused:
        status = "partial"
    else:
        status = "ok"
    logger.debug(
        "memory_fetch_complete",
        fetched=len(results),
        after_dedup=len(deduped),
        after_admission=len(shared),
        refused=outcome.refused,
        status=status,
    )
    return SharedFetchResult(shared, status, len(results), outcome.refused)
