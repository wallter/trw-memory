"""``JevHttpJudge`` — the opt-in HTTP backend for TypeSafe's Jev model.

Talks to the OpenRouter Decisions API per the Jev decision-backend design.
Never raises: every failure mode (bad key, timeout, malformed response, rate
limit exhausted) degrades to ``None`` so a caller's existing prose/deterministic
path is always the fallback. Never logs the API key or the state payload —
only site, latency, outcome and cost are structured-logged.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import structlog

from trw_memory.decisions._judge import DecisionState
from trw_memory.decisions._models import DecisionQuestion, DecisionResult
from trw_memory.decisions._wire import build_payload, parse_response

logger = structlog.get_logger(__name__)

#: The alpha OpenRouter Decisions endpoint (JEV-PRIMER.md). Configurable
#: because the primer notes the route may move while the API is in beta.
DEFAULT_BASE_URL = "https://openrouter.ai/api/alpha/decisions"

#: The leading ``~`` is REQUIRED — the plain slug returns HTTP 400 "does not
#: exist" (verified against the live OpenRouter Decisions API).
DEFAULT_MODEL = "~typesafe/jev-latest"

#: Statuses worth one retry: rate limit, overloaded, and the common transient
#: gateway failures. Anything else (401, 422, ...) degrades straight to None.
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504, 529})

#: Ceiling on how long a single retry sleeps, regardless of what the server's
#: ``retry-after`` header claims — a hard bound protects the caller's own
#: ``timeout_s`` budget from a server asking for an unreasonable wait.
_MAX_RETRY_AFTER_SECONDS = 5.0

#: Fallback sleep when a retryable response carries no usable ``retry-after``.
_DEFAULT_RETRY_AFTER_SECONDS = 1.0


def _parse_retry_after(value: str | None) -> float:
    """Parse a ``Retry-After`` header: either delta-seconds or an HTTP-date."""
    if not value:
        return _DEFAULT_RETRY_AFTER_SECONDS
    stripped = value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:  # trw-fail-silent-allow: not a plain number -- fall through to HTTP-date parsing below
        pass
    try:
        target = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return _DEFAULT_RETRY_AFTER_SECONDS
    if target.tzinfo is None:
        return _DEFAULT_RETRY_AFTER_SECONDS
    delta = (target - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)


class JevHttpJudge:
    """Sends state + questions to OpenRouter's Jev decisions endpoint.

    ``transport`` and ``client`` exist purely for test injection (an
    ``httpx.MockTransport`` or a preconfigured ``httpx.Client``); production
    callers pass neither and get a fresh client per call scoped to
    ``timeout_s``. Passing both is redundant — ``client`` wins.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
        redact: Callable[[DecisionState], DecisionState] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._transport = transport
        self._client = client
        self._redact = redact

    def __repr__(self) -> str:
        # trw:intentional never interpolate api_key here — repr() output can
        # land in a traceback, a log line, or a debugger watch expression.
        return f"JevHttpJudge(base_url={self._base_url!r}, model={self._model!r})"

    def decide(
        self,
        state: DecisionState,
        questions: Mapping[str, DecisionQuestion | Mapping[str, Any]],
        *,
        timeout_s: float = 10.0,
        session_id: str | None = None,
    ) -> DecisionResult | None:
        start = time.monotonic()
        try:
            wire_state = self._redact(state) if self._redact is not None else state
            payload = build_payload(self._model, wire_state, questions, session_id)
        except Exception as exc:  # trw-fail-silent-allow: DecisionJudge.decide never raises -- logged, then abstain
            # Type only: a traceback/validation message can echo state or question values.
            logger.warning("jev_decision_payload_build_failed", site=self._base_url, error_type=type(exc).__name__)
            return None

        try:
            response = self._send(payload, timeout_s)
        except Exception:  # trw-fail-silent-allow: DecisionJudge.decide never raises -- logged, then abstain
            latency_ms = (time.monotonic() - start) * 1000
            logger.info("jev_decision_error", site=self._base_url, latency_ms=round(latency_ms, 1), ok=False)
            return None

        latency_ms = (time.monotonic() - start) * 1000
        if response is None:
            logger.info("jev_decision_error", site=self._base_url, latency_ms=round(latency_ms, 1), ok=False)
            return None

        try:
            body = response.json()
            result = parse_response(body, backend="jev", latency_ms=latency_ms)
        except Exception as exc:  # trw-fail-silent-allow: DecisionJudge.decide never raises -- logged, then abstain
            # Type only: a traceback/validation message can echo response-derived values.
            logger.warning("jev_decision_parse_failed", site=self._base_url, error_type=type(exc).__name__)
            return None

        logger.info(
            "jev_decision_ok",
            site=self._base_url,
            latency_ms=round(latency_ms, 1),
            ok=True,
            cost=result.usage.get("cost"),
        )
        return result

    def _send(self, payload: dict[str, object], timeout_s: float) -> httpx.Response | None:
        """POST once, retry once on a retryable status, else return the last response."""
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.Client(transport=self._transport, timeout=timeout_s)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            response = client.post(self._base_url, json=payload, headers=headers, timeout=timeout_s)
            if response.status_code in _RETRYABLE_STATUS:
                sleep_s = min(_parse_retry_after(response.headers.get("retry-after")), _MAX_RETRY_AFTER_SECONDS)
                time.sleep(sleep_s)
                response = client.post(self._base_url, json=payload, headers=headers, timeout=timeout_s)
            if response.status_code >= 400:
                return None
            return response
        finally:
            if owns_client:
                client.close()


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "JevHttpJudge"]
