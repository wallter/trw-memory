"""Shared helpers and typed contracts for remote sync."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import structlog
from typing_extensions import TypedDict

logger = structlog.get_logger(__name__)

PUBLISH_TIMEOUT = 5.0
FETCH_TIMEOUT = 3.0

MAX_SUMMARY_LENGTH = 1000
MAX_DETAIL_LENGTH = 10_000
MAX_TAGS_COUNT = 20


class AnonymizedEntry(TypedDict):
    summary: str
    detail: str | None
    tags: list[str]
    impact: float
    source_project: str
    source_learning_id: str


# ---------------------------------------------------------------------------
# learning_api_v1 protocol boundary (PRD-CORE-181-FR06)
# ---------------------------------------------------------------------------
#
# The first-party learning API speaks the legacy ``impact`` / ``min_impact``
# vocabulary on the wire, but every local storage / lifecycle reader is
# canonical ``importance`` after the memory_model_v2 cutover. This module is the
# SOLE, explicitly versioned translation boundary: publish/fetch call through
# these encoders/decoders and never reference ``impact`` directly, so a source
# census can prove the external vocabulary is contained here.


def encode_learning_api_v1(
    *,
    summary: str,
    detail: str | None,
    tags: list[str],
    importance: float,
    source_project: str,
    source_learning_id: str,
) -> AnonymizedEntry:
    """Encode canonical fields into the external ``learning_api_v1`` publish payload.

    Maps canonical ``importance`` onto the external ``impact`` wire field. This
    is the only place the outbound ``impact`` vocabulary is produced.
    """
    return AnonymizedEntry(
        summary=summary,
        detail=detail,
        tags=tags,
        impact=importance,
        source_project=source_project,
        source_learning_id=source_learning_id,
    )


def encode_learning_api_v1_search(
    *,
    query: str,
    limit: int,
    min_importance: float,
) -> dict[str, object]:
    """Encode a search request, mapping ``min_importance`` -> external ``min_impact``."""
    return {"query": query, "limit": limit, "min_impact": min_importance}


def decode_learning_api_v1_result(result: dict[str, object]) -> dict[str, object]:
    """Decode an external ``learning_api_v1`` result into canonical vocabulary.

    Maps the external ``impact`` field back onto canonical ``importance`` so
    downstream local readers never see the wire vocabulary. Results without an
    ``impact`` field pass through unchanged.
    """
    if "impact" not in result:
        return result
    decoded = dict(result)
    decoded["importance"] = decoded.pop("impact")
    return decoded


class PublishResult(TypedDict):
    success: bool
    remote_id: str | None
    retryable: bool


class RetryDrainResult(TypedDict):
    drained: int
    failed: int
    skipped: int
    remote_ids: dict[str, str]


class SnapshotHashPayload(TypedDict):
    digest: str
    size_bytes: int
    created_at: str
    installation_id: str


def is_valid_platform_url(platform_url: str) -> bool:
    if not platform_url.strip():
        return False
    parsed = urlparse(platform_url)
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and os.getenv("TRW_DEBUG", "").lower() == "true"


#: The official platform host. Always trusted over https, regardless of what
#: a project's tracked config claims ``platform_url`` is. Mirrors
#: trw_mcp.state._platform_trust.DEFAULT_TRUSTED_PLATFORM_HOST (trw-memory
#: cannot import trw-mcp, so this is a small, deliberately duplicated copy of
#: the same trust boundary, not a shared import).
DEFAULT_TRUSTED_PLATFORM_HOST = "api.trwframework.com"

#: Loopback hosts may receive the bearer over plain http — dev only.
_DEV_LOCALHOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

#: Comma-separated additional trusted hostnames (machine/operator controlled).
_ENV_TRUSTED_HOSTS = "TRW_PLATFORM_TRUSTED_HOSTS"


def _user_config_trusted_hosts() -> frozenset[str]:
    """Hosts from ``~/.trw/config.yaml`` ``platform_trusted_hosts`` — machine layer ONLY.

    Read directly from the user-level file rather than any project-merged
    config: a project's tracked ``.trw/config.yaml`` must never be able to
    add itself to its own trust list.
    """
    try:
        from ruamel.yaml import YAML

        path = Path.home() / ".trw" / "config.yaml"
        if not path.exists():
            return frozenset()
        yaml = YAML(typ="safe")
        data = yaml.load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return frozenset()
        raw = data.get("platform_trusted_hosts", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return frozenset()
        return frozenset(str(h).strip().lower() for h in raw if str(h).strip())
    except Exception:  # justified: boundary, a malformed machine config must not crash the trust gate
        logger.debug("platform_trusted_hosts_read_failed", exc_info=True)
        return frozenset()


def _env_trusted_hosts() -> frozenset[str]:
    raw = os.environ.get(_ENV_TRUSTED_HOSTS, "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def trusted_platform_hosts() -> frozenset[str]:
    """The full https trusted-host allowlist: default + user config + env.

    Never includes anything derived from a project's tracked
    ``.trw/config.yaml``.
    """
    return frozenset({DEFAULT_TRUSTED_PLATFORM_HOST}) | _user_config_trusted_hosts() | _env_trusted_hosts()


def bearer_allowed_for(url: str) -> bool:
    """Return True iff the platform bearer API key may be attached to *url*.

    Same policy as trw-mcp's ``state._platform_trust.bearer_allowed_for``:
    https to a trusted host, or http to a loopback dev host. A project's
    tracked ``MemoryConfig.platform_url`` can point anywhere it likes
    (self-hosted deployments stay possible); it just never causes that host
    to receive the credential.
    """
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    scheme = parts.scheme.lower()
    if scheme == "http" and host in _DEV_LOCALHOSTS:
        return True
    if scheme != "https":
        return False
    return host in trusted_platform_hosts()


def build_platform_headers(api_key: str | None, url: str) -> dict[str, str]:
    """Build request headers, attaching the bearer only when *url* is trusted.

    This is the ONE function every remote-sync call site uses to build
    headers — see ``tests/test_platform_trust.py``'s census check, which
    fails if any other module in this package contains a raw ``Bearer``
    header literal.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key and bearer_allowed_for(url):
        headers["Authorization"] = f"Bearer {api_key}"
    elif api_key:
        logger.warning("credential_withheld_untrusted_host", url=url)
    return headers
