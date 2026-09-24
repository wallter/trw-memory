"""``JevHttpJudge`` — the opt-in HTTP backend for TypeSafe's Jev model.

Talks to the OpenRouter Decisions API per the Jev decision-backend design.
Never raises: every failure mode (bad key, timeout, malformed response, rate
limit exhausted) is returned as a typed :class:`DecisionFailure`, so a caller's
prose/deterministic path is always the fallback. Never logs the API key or the state payload —
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
from trw_memory.decisions._models import OVER_CEILING_HINT, DecisionFailure, DecisionOutcome, DecisionQuestion
from trw_memory.decisions._wire import build_payload, parse_response

logger = structlog.get_logger(__name__)

#: The alpha OpenRouter Decisions endpoint (JEV-PRIMER.md). Configurable
#: because the primer notes the route may move while the API is in beta.
DEFAULT_BASE_URL = "https://openrouter.ai/api/alpha/decisions"

#: The leading ``~`` is REQUIRED — the plain slug returns HTTP 400 "does not
#: exist" (verified against the live OpenRouter Decisions API).
DEFAULT_MODEL = "~typesafe/jev-latest"

#: Statuses worth one retry: rate limit, overloaded, and the common transient
#: gateway failures. Anything else (401, 422, ...) is classified without a retry.
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504, 529})

#: Ceiling on how long a single retry sleeps, regardless of what the server's
#: ``retry-after`` header claims; the retry is also skipped when the sleep would
#: leave nothing of the caller's ``timeout_s``.
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
    ) -> DecisionOutcome:
        """Answer every question, or return one :class:`DecisionFailure` naming why not. Never raises.

        Kinds follow the measured error table in JEV-STARTER: 400/422 and a payload that
        will not build are ``invalid_request`` (fix the request, never retry); 401/403 is
        ``auth``; 429 after the one retry is ``rate_limited``; a transport timeout is
        ``timeout``; any other 5xx/transport error is ``provider_error``; a 200 whose body
        does not parse is ``malformed_response``. Details name a status and an error type
        only -- never the state or the body, which can carry the caller's data.
        """
        start = time.monotonic()
        try:
            wire_state = self._redact(state) if self._redact is not None else state
            payload = build_payload(self._model, wire_state, questions, session_id)
        except Exception as exc:  # trw-fail-silent-allow: never raises -- classified and returned
            logger.warning("jev_decision_payload_build_failed", site=self._base_url, error_type=type(exc).__name__)
            return DecisionFailure(kind="invalid_request", detail=f"request would not build: {type(exc).__name__}")

        try:
            response = self._send(payload, timeout_s)
        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - start) * 1000
            logger.info("jev_decision_error", site=self._base_url, latency_ms=round(latency_ms, 1), ok=False)
            return DecisionFailure(kind="timeout", detail=f"no response within {timeout_s:g}s")
        except Exception as exc:  # trw-fail-silent-allow: never raises -- classified and returned
            latency_ms = (time.monotonic() - start) * 1000
            logger.info("jev_decision_error", site=self._base_url, latency_ms=round(latency_ms, 1), ok=False)
            return DecisionFailure(kind="provider_error", detail=f"transport error: {type(exc).__name__}")

        latency_ms = (time.monotonic() - start) * 1000
        if response.status_code >= 400:
            logger.info(
                "jev_decision_error",
                site=self._base_url,
                latency_ms=round(latency_ms, 1),
                ok=False,
                status=response.status_code,
            )
            return _classify_http_failure(response)

        try:
            body = response.json()
            result = parse_response(body, backend="jev", latency_ms=latency_ms)
        except Exception as exc:  # trw-fail-silent-allow: never raises -- classified and returned
            logger.warning("jev_decision_parse_failed", site=self._base_url, error_type=type(exc).__name__)
            return DecisionFailure(
                kind="malformed_response", detail=f"200 but body did not parse: {type(exc).__name__}"
            )

        if not result.answers and result.malformed_ids:
            logger.warning("jev_decision_parse_failed", site=self._base_url, error_type=result.malformed_error_type)
            return DecisionFailure(kind="malformed_response", detail="200 but every answer member failed validation")

        logger.info(
            "jev_decision_ok",
            site=self._base_url,
            latency_ms=round(latency_ms, 1),
            ok=True,
            cost=result.usage.get("cost"),
        )
        return result

    def _send(self, payload: dict[str, object], timeout_s: float) -> httpx.Response:
        """POST once, retry once on a retryable status, and return the LAST response.

        ``timeout_s`` bounds the whole call: the retry gets only what is left after its
        ``retry-after`` sleep, and is skipped when nothing would be. The caller classifies the
        status, so a 400 it must fix stays distinct from a 503.
        """
        deadline = time.monotonic() + timeout_s
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.Client(transport=self._transport, timeout=timeout_s)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            response = client.post(self._base_url, json=payload, headers=headers, timeout=timeout_s)
            if response.status_code in _RETRYABLE_STATUS:
                sleep_s = min(_parse_retry_after(response.headers.get("retry-after")), _MAX_RETRY_AFTER_SECONDS)
                remaining = deadline - time.monotonic() - sleep_s
                if remaining > 0:
                    time.sleep(sleep_s)
                    response = client.post(self._base_url, json=payload, headers=headers, timeout=remaining)
            return response
        finally:
            if owns_client:
                client.close()


#: Provider 4xx markers mapped to CANONICAL hints. Only the canonical text is ever surfaced;
#: the provider's own message is never copied, because a 400 body can echo the request.
_CANONICAL_HINTS = (
    ("Too many choices", "too many choice options (max 255)"),
    ("must have at most", "too many choice options (max 255)"),
    ("max_tokens_exceeded", OVER_CEILING_HINT),
)


def _classify_http_failure(response: httpx.Response) -> DecisionFailure:
    status = response.status_code
    if status in (401, 403):
        return DecisionFailure(kind="auth", detail=f"HTTP {status}")
    if status == 429:
        return DecisionFailure(kind="rate_limited", detail="HTTP 429 after one retry")
    if status in (400, 422):
        hint = ""
        try:
            body = response.json()
            text = str(body.get("detail") or body.get("error") or "") if isinstance(body, dict) else ""
            for marker, canonical in _CANONICAL_HINTS:
                if marker in text:
                    hint = ": " + canonical
                    break
        except Exception:  # trw-fail-silent-allow: a hint is optional; the kind is what matters
            hint = ""
        return DecisionFailure(kind="invalid_request", detail=f"HTTP {status}{hint}")
    return DecisionFailure(kind="provider_error", detail=f"HTTP {status}")


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "JevHttpJudge"]
