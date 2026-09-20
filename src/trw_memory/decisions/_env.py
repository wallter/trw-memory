"""``judge_from_env`` — resolve a :class:`DecisionJudge` from configuration.

Off by default (per the Jev decision-backend design's first constraint): a
missing enable flag or a missing key silently resolves to :class:`NullJudge`,
never an error. This is the ONE place an operator opts a project into the Jev
backend, so every default install — and every project that never sets these
variables — gets zero network calls out of this package.

**Trust split (release-verify R1).** Enablement (``TRW_JEV_ENABLED``), the
endpoint (``TRW_JEV_BASE_URL``) and the model (``TRW_JEV_MODEL``) are read from
the PROCESS environment ONLY. A project ``.env`` — which is repo-controlled and
therefore attacker-authored in a cloned repo — may supply ONLY the credential
``OPENROUTER_API_KEY``, and can never enable the backend or redirect where the
key is sent. The resolved base URL must additionally be ``https`` with a host in
:data:`_ALLOWED_BASE_URL_HOSTS` on the default port; anything else resolves to
:class:`NullJudge` (the call abstains, nothing leaves the machine) rather than
raising.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

import structlog

from trw_memory.decisions._dotenv import parse_dotenv_subset
from trw_memory.decisions._jev_http import DEFAULT_BASE_URL, DEFAULT_MODEL, JevHttpJudge
from trw_memory.decisions._judge import DecisionJudge, NullJudge

logger = structlog.get_logger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: The ONLY key a dotenv file is read for. Enablement, base URL and model are
#: deliberately NOT read from it — see the module docstring's trust split.
_DOTENV_ALLOWED_KEYS = frozenset({"OPENROUTER_API_KEY"})

#: Hosts the API key may be sent to. Kept as a constant rather than a new
#: config surface: the key belongs to exactly one provider.
_ALLOWED_BASE_URL_HOSTS = frozenset({"openrouter.ai"})


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def _inspect_base_url(base_url: str) -> tuple[bool, str, str]:
    """Return ``(allowed, host, scheme)`` for ``base_url`` — never raises.

    Allowed means: ``https`` scheme, host in :data:`_ALLOWED_BASE_URL_HOSTS`,
    and no explicit port other than 443. ``urlsplit`` lowercases the scheme and
    host, so ``HTTPS://OPENROUTER.AI/...`` is accepted. Deliberately rejected,
    because each is a host the allowlist did not vet: userinfo that shifts the
    real host (``https://openrouter.ai@evil.example/``, whose host is
    ``evil.example``), a trailing-dot FQDN (``openrouter.ai.``, compared whole
    rather than normalized), and a non-443 port. A malformed URL — ``urlsplit``
    raises ``ValueError`` on an unbalanced bracket or a non-numeric port — is
    rejected rather than propagated, since this resolver never raises into a
    caller.
    """
    try:
        parsed = urlsplit(base_url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return False, "", ""
    allowed = parsed.scheme == "https" and host in _ALLOWED_BASE_URL_HOSTS and port in (None, 443)
    return allowed, host, parsed.scheme


def judge_from_env(
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = None,
) -> DecisionJudge:
    """Resolve the configured judge from the process env, plus a key-only dotenv.

    Returns :class:`JevHttpJudge` only when ALL of: ``TRW_JEV_ENABLED`` is
    truthy in ``env``, an ``OPENROUTER_API_KEY`` is found, and the resolved
    ``TRW_JEV_BASE_URL`` is ``https`` on an allowlisted host. Otherwise
    :class:`NullJudge`. ``TRW_JEV_ENABLED``/``TRW_JEV_BASE_URL``/
    ``TRW_JEV_MODEL`` come from ``env`` only; ``dotenv_path``, when given, is
    parsed for ``OPENROUTER_API_KEY`` and nothing else (no ``python-dotenv``
    dependency; see :mod:`trw_memory.decisions._dotenv`). See
    :func:`_inspect_base_url` for exactly which URLs are allowed.
    """
    process_env = os.environ if env is None else env

    if not _truthy(process_env.get("TRW_JEV_ENABLED")):
        return NullJudge()

    api_key = process_env.get("OPENROUTER_API_KEY") or (
        parse_dotenv_subset(dotenv_path, allowed_keys=_DOTENV_ALLOWED_KEYS).get("OPENROUTER_API_KEY")
        if dotenv_path is not None
        else None
    )
    if not api_key:
        return NullJudge()

    base_url = process_env.get("TRW_JEV_BASE_URL") or DEFAULT_BASE_URL
    allowed, host, scheme = _inspect_base_url(base_url)
    if not allowed:
        logger.info(
            "jev_base_url_rejected",
            reason="base URL must be https on an allowlisted host, default port; abstaining without sending the key",
            host=host,
            scheme=scheme,
        )
        return NullJudge()

    model = process_env.get("TRW_JEV_MODEL") or DEFAULT_MODEL
    return JevHttpJudge(api_key, base_url=base_url, model=model)


__all__ = ["judge_from_env"]
