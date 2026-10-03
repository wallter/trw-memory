"""Remote publish helpers for memory sync."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import httpx
import structlog

from trw_memory.exceptions import MemoryError as TrwMemoryError
from trw_memory.labels import LabelPolicy, Sink
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.egress_gate import egress_refused, identity_keys
from trw_memory.security.pii import anonymize_installation_id, redact_paths, strip_pii
from trw_memory.sync._remote_common import (
    MAX_DETAIL_LENGTH,
    MAX_SUMMARY_LENGTH,
    MAX_TAGS_COUNT,
    PUBLISH_TIMEOUT,
    AnonymizedEntry,
    PublishResult,
    RetryDrainResult,
    build_platform_headers,
    encode_learning_api_v1,
    platform_contact_blocked,
)
from trw_memory.sync.retry_queue import RetryQueue

logger = structlog.get_logger(__name__)


def _anonymize_entry(entry: MemoryEntry, project_root: str = "") -> AnonymizedEntry:
    content = redact_paths(strip_pii(entry.content), project_root)
    detail = redact_paths(strip_pii(entry.detail), project_root)
    # Tags are egressed content too. The write path stores them verbatim (see
    # ``security/_runtime_pii``), so a credential or email pasted into a tag
    # leaves the machine raw unless this boundary masks it.
    tags = [strip_pii(tag) for tag in entry.tags[:MAX_TAGS_COUNT]]
    # Canonical ``importance`` -> external wire vocabulary via the sole
    # learning_api_v1 boundary encoder (PRD-CORE-181-FR06).
    return encode_learning_api_v1(
        summary=content[:MAX_SUMMARY_LENGTH],
        detail=detail[:MAX_DETAIL_LENGTH] if detail else None,
        tags=tags,
        importance=entry.importance,
        source_project=anonymize_installation_id(entry.metadata.get("installation_id", "")),
        source_learning_id=entry.id,
    )


def _extract_remote_id(response: httpx.Response) -> str | None:
    try:
        raw_body = response.json()
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if isinstance(raw_body, dict):
        raw_remote_id = raw_body.get("id")
        if raw_remote_id is not None:
            return str(raw_remote_id)
    return None


#: The ledger keys a queued payload carries (PRD-CORE-333): never sent, read at the POST site.
LEDGER_KEYS_FIELD = "_ledger_keys"


def _publish_payload_result(
    payload: dict[str, object],
    cfg: MemoryConfig,
    *,
    entry_id: str = "",
    ledger_keys: object = None,
) -> PublishResult:
    # Asked again at the one POST site, so every retry-drain send sees the switch as it is now.
    if platform_contact_blocked(cfg, "memory_publish"):
        return {"success": False, "remote_id": None, "retryable": False}
    # PRD-CORE-333: the quarantine ledger is asked here too, immediately before the POST, so a row quarantined
    # after it was queued never leaves the host. A payload with no identity cannot be proven clean: refused.
    refusal = egress_refused(ledger_keys, cfg) if isinstance(ledger_keys, list) else "blocked"
    if refusal is not None:
        logger.info("memory_publish_withheld_by_quarantine", entry_id=entry_id, reason=refusal)
        return {"success": False, "remote_id": None, "retryable": refusal == "unreadable"}

    try:
        publish_url = f"{cfg.platform_url.rstrip('/')}/v1/learnings"
        with httpx.Client(timeout=PUBLISH_TIMEOUT) as client:
            resp = client.post(
                publish_url,
                json=payload,
                headers=build_platform_headers(cfg.platform_api_key, publish_url),
            )
            if 200 <= resp.status_code < 300:
                remote_id = _extract_remote_id(resp)
                logger.debug("memory_published", entry_id=entry_id, remote_id=remote_id)
                return {"success": True, "remote_id": remote_id, "retryable": False}
            logger.warning("memory_publish_failed", entry_id=entry_id, status=resp.status_code)
            return {"success": False, "remote_id": None, "retryable": True}
    except (httpx.HTTPError, OSError, ConnectionError):
        logger.debug("memory_publish_error", entry_id=entry_id, exc_info=True)
        return {"success": False, "remote_id": None, "retryable": True}


def publish_memory_result(
    entry: MemoryEntry,
    cfg: MemoryConfig,
    *,
    project_root: str = "",
) -> PublishResult:
    if platform_contact_blocked(cfg, "memory_publish") or entry.importance < cfg.sync_min_importance:
        return {"success": False, "remote_id": None, "retryable": False}
    if not LabelPolicy.current().admit([entry], Sink.PLATFORM).admitted:
        # PRD-SEC-023 FR05: a row labelled above team never leaves the host; not retryable, so it is never queued either.
        logger.info("memory_publish_withheld_by_label", entry_id=entry.id)
        return {"success": False, "remote_id": None, "retryable": False}

    payload = _anonymize_entry(entry, project_root)
    return _publish_payload_result(
        cast("dict[str, object]", payload), cfg, entry_id=entry.id, ledger_keys=identity_keys(entry)
    )


#: Reads a stored row as it is now by ``(namespace, entry_id)``; ``None`` when it is gone.
RowResolver = Callable[[str, str], MemoryEntry | None]


def drain_retry_queue(queue: RetryQueue, cfg: MemoryConfig, *, current_row: RowResolver) -> RetryDrainResult:
    result, _ = _drain_retry_queue_with_ids(queue, cfg, current_row=current_row)
    return result


def _queued_identity(payload: dict[str, object]) -> tuple[str, str] | None:
    """The ``(namespace, entry_id)`` a queued payload was built from: its one ``id`` ledger key, which must name the
    payload's own ``source_learning_id``. ``None`` for a missing, ambiguous or inconsistent identity (refused)."""
    keys = payload.get(LEDGER_KEYS_FIELD)
    ids = [
        (key[1], key[2])
        for key in (keys if isinstance(keys, list) else [])
        if isinstance(key, list) and len(key) == 3 and key[0] == "id" and all(isinstance(part, str) for part in key)
    ]
    if len(ids) != 1 or ids[0][1] != payload.get("source_learning_id"):
        return None
    return ids[0]


#: A published retry record: its entry id, the ``(sync_seq, sync_hash)`` it was queued at (``None`` for a record
#: queued before revisions were recorded) and the platform's remote id, when it returned one.
DrainedPublish = tuple[str, tuple[int, str] | None, str | None]


def _drain_retry_queue_with_ids(
    queue: RetryQueue,
    cfg: MemoryConfig,
    *,
    current_row: RowResolver,
    namespace: str | None = None,
) -> tuple[RetryDrainResult, list[DrainedPublish]]:
    """Re-send the queued publishes, each only if the row it was built from is still admitted to the platform.

    A queued payload is an anonymized snapshot with no stamp or full tag list, so the label is asked of the row as it
    is now (SYNC-RETRY-LABEL-RECHECK): resolved by the payload's namespace-qualified identity, in its own namespace,
    immediately before its POST (after any backoff), against the label policy in force then. A row relabelled above
    team, gone, or named by no consistent identity is withheld. With *namespace*, a record queued by another namespace
    (whose rows this drainer cannot read) is left untouched for its own namespace's drain.
    """
    if platform_contact_blocked(cfg, "memory_retry_drain"):
        return {"drained": 0, "failed": 0, "skipped": queue.depth(), "remote_ids": {}}, []

    published_remote_ids: list[str | None] = []

    def publish_payload(payload: dict[str, object]) -> bool:
        # A payload queued before vectors left the wire (PRD-CORE-302 FR04) may still carry one.
        payload.pop("embedding", None)
        source_learning_id = payload.get("source_learning_id")
        entry_id = str(source_learning_id) if isinstance(source_learning_id, str) else ""
        # PRD-SEC-023 FR05 at send time, against the policy in force now: the same refusal publish_memory_result makes.
        identity = _queued_identity(payload)
        try:
            row = current_row(*identity) if identity is not None else None
        except (
            TrwMemoryError,
            OSError,
        ):  # trw-fail-silent-allow: fail closed -- this record is withheld and keeps its retry; logged below
            logger.warning("memory_publish_row_unreadable", entry_id=entry_id, exc_info=True)
            row = None
        if row is None or not LabelPolicy.current().admit([row], Sink.PLATFORM).admitted:
            logger.info("memory_publish_withheld_by_label", entry_id=entry_id, row_found=row is not None)
            return False
        # The keys stay in the queued record (a failed send is re-persisted from this dict); only the POST omits them.
        body = {key: value for key, value in payload.items() if key != LEDGER_KEYS_FIELD}
        result = _publish_payload_result(body, cfg, entry_id=entry_id, ledger_keys=payload.get(LEDGER_KEYS_FIELD))
        if result["success"]:
            published_remote_ids.append(result["remote_id"])
        return result["success"]

    def owns(payload: dict[str, object]) -> bool:
        identity = _queued_identity(payload)
        return identity is None or identity[0] == namespace  # no identity: this drain refuses it, as before

    drain_result, drained = (
        queue._drain_with_ids(publish_payload) if namespace is None else queue._drain_with_ids(publish_payload, owns)
    )
    # One entry per published RECORD: the same entry queued twice at two revisions acks each on its own.
    published = [
        (entry_id, revision, remote_id)
        for (entry_id, revision), remote_id in zip(drained, published_remote_ids, strict=True)
    ]
    return {
        "drained": drain_result["drained"],
        "failed": drain_result["failed"],
        "skipped": drain_result["skipped"],
        "remote_ids": {entry_id: remote_id for entry_id, _, remote_id in published if remote_id is not None},
    }, published


def retire_remote_memory(remote_id: str, cfg: MemoryConfig) -> bool:
    if not remote_id or platform_contact_blocked(cfg, "memory_retire"):
        return True

    try:
        retire_url = f"{cfg.platform_url.rstrip('/')}/v1/learnings/{remote_id}/status"
        with httpx.Client(timeout=PUBLISH_TIMEOUT) as client:
            resp = client.patch(
                retire_url,
                json={"status": "obsolete"},
                headers=build_platform_headers(cfg.platform_api_key, retire_url),
            )
            if 200 <= resp.status_code < 300:
                logger.debug("memory_retired_remote", remote_id=remote_id)
                return True
            logger.warning("memory_retire_failed", remote_id=remote_id, status=resp.status_code)
            return False
    except (
        httpx.HTTPError,
        OSError,
        ConnectionError,
    ):  # trw-fail-silent-allow: pre-existing; logs at debug and fail-open is documented sync behavior
        logger.debug("memory_retire_error", remote_id=remote_id, exc_info=True)
        return False
