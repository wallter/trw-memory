"""Default egress redaction for decision state (POC).

The seam's first shipped version left redaction to a single caller: ``judge_from_env`` returned a
judge with ``redact=None``, and only trw-mcp's ``trw_decision`` tool scrubbed state before calling
it. Every other caller -- benchmarks, scripts, harnesses -- sent state verbatim (learning L-AYkg).
Redaction therefore belongs at the seam, on by default, with opt-out for callers who have already
scrubbed.

This is deliberately a *duplicate* of the credential patterns in trw-mcp's feedback redactor: a
public package cannot import them from trw-mcp, and R2-014 (single redactor) is the consolidation
that removes this file's patterns. Keep the two in sync until then.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from trw_memory.security.pii import strip_pii

__all__ = ["default_redactor", "redact_state"]

_PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_AUTH = re.compile(r"(Authorization\s*:\s*)(?:Bearer|Token|Basic|ApiKey)?\s*\S+", re.IGNORECASE)
_BEARER = re.compile(
    r"\b(Bearer|Token)\s+(?:(?=[A-Za-z0-9._~+/=-]*\d)[A-Za-z0-9._~+/=-]{8,}|[A-Za-z0-9._~+/=-]{16,})",
    re.IGNORECASE,
)
_CONN = re.compile(r"\b([a-z][a-z0-9+.-]*://)[^\s:/@]+:[^\s:/@]+@")
_APIKEY = re.compile(r"\b(?:sk|pk|rk|api|key|token|secret)[-_](?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{16,}\b", re.IGNORECASE)
_ENVVAR = re.compile(r"\b[A-Z][A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|API[_-]?KEY|ACCESS[_-]?KEY)\s*=\s*\S+")

#: A dict key whose *value* is dropped whole, whatever its shape. camelCase is split before matching.
_SECRET_KEY = re.compile(
    r"^(?:.*[_-])?(?:pass(?:word|wd|phrase)?|secret|token|api[_-]?key|apikey|access[_-]?key"
    r"|private[_-]?key|client[_-]?secret|credentials?|authorization|auth[_-]?token|bearer)(?:[_-].*)?$",
    re.IGNORECASE,
)


def _is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY.search(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)))


def default_redactor(text: str) -> str:
    """Credential shapes first (a PEM body would partial-match later passes), then PII."""
    out = _PEM.sub("<REDACTED:private_key>", text)
    out = _CONN.sub(r"\1<REDACTED:credentials>@", out)
    out = _AUTH.sub(r"\1<REDACTED:authorization>", out)
    out = _JWT.sub("<REDACTED:jwt>", out)  # before _BEARER: "token eyJ..." must read as a JWT
    out = _BEARER.sub(r"\1 <REDACTED:bearer>", out)
    out = _APIKEY.sub("<REDACTED:api_key>", out)
    out = _ENVVAR.sub("<REDACTED:env>", out)
    return strip_pii(out)


def unique_key(candidate: str, taken: Mapping[str, Any] | set[str]) -> str:
    """Disambiguate a redacted key that already exists, deterministically.

    Redaction is many-to-one (``alice@x``/``bob@x`` both become ``<email>``), so rewriting keys in
    place would silently DROP entries -- for a ``choice`` question that means losing an option after
    validation passed and scoring a question the caller never asked. Suffixing keeps the entry count
    equal to the input's (release-verify N2).
    """
    if candidate not in taken:
        return candidate
    suffix = 2
    while f"{candidate}#{suffix}" in taken:
        suffix += 1
    return f"{candidate}#{suffix}"


def redact_state(state: Any, redactor: Any = default_redactor) -> Any:
    """Redact every string leaf AND every string key; drop values under secret-named keys whole.

    * A secret-named key hides its value whatever the shape: a list of tokens, a nested dict or a
      numeric PIN under ``credential`` is still a credential (release-verify R2).
    * Keys are redacted too -- a key can carry a credential or an email as easily as a value -- and
      two keys that redact alike are kept apart by :func:`unique_key`, so a dict never loses entries.
    * A tuple is rendered as a list, which is what JSON egress does with it anyway.
    """
    if isinstance(state, str):
        return redactor(state)
    if isinstance(state, Mapping):
        out: dict[Any, Any] = {}
        for key, value in state.items():
            if isinstance(key, str) and _is_secret_key(key):
                new_value: Any = "<REDACTED:secret>"
            else:
                new_value = redact_state(value, redactor)
            new_key = unique_key(redactor(key), out) if isinstance(key, str) else key
            out[new_key] = new_value
        return out
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes)):
        return [redact_state(item, redactor) for item in state]
    return state


__all__ = ["default_redactor", "redact_state", "unique_key"]
