"""``judge_from_env`` — resolve a :class:`DecisionJudge` from configuration.

Off by default (per the Jev decision-backend design's first constraint): a
missing enable flag or a missing key silently resolves to :class:`NullJudge`,
never an error. This is the ONE place an operator or a project opts into the
Jev backend, so every default install — and every project that never sets
these variables — gets zero network calls out of this package.

**Enablement (2026-09-23 operator decision).** Whether the backend is on is
resolved by :func:`trw_memory.decisions._enablement.resolve_backend_enablement`
— the one precedence cascade both trw-memory and trw-mcp read: process env
``TRW_JEV_ENABLED`` beats project scope (``assess_enabled`` in the project's
``.trw/config.yaml``, or ``TRW_JEV_ENABLED`` in its ``.env``) beats user scope
(``assess_enabled`` in ``~/.trw/config.yaml``) beats off. A project MAY now
enable the backend (relaxing the prior "off only" rule) — see that module's
docstring for the full cascade and rationale.

**Trust split (release-verify R1), unchanged by the above.** The endpoint
(``TRW_JEV_BASE_URL``) and the model (``TRW_JEV_MODEL``) are read from the
PROCESS environment ONLY. A project ``.env`` — which is repo-controlled and
therefore attacker-authored in a cloned repo — may supply ONLY the credential
``OPENROUTER_API_KEY``, and can never redirect where the key is sent: enabling
the backend from project scope does not widen the credential or endpoint
trust boundary, only the on/off decision. The resolved base URL must
additionally be ``https`` with a host in :data:`_ALLOWED_BASE_URL_HOSTS` on
the default port; anything else resolves to :class:`NullJudge` (the call
abstains, nothing leaves the machine) rather than raising.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import structlog

from trw_memory.decisions._dotenv import parse_dotenv_subset
from trw_memory.decisions._judge import DecisionJudge, NullJudge

if TYPE_CHECKING:
    # Only for type hints — importing this at module scope would pull in httpx
    # (via _jev_http) on every import of this module, including the disabled path.
    from trw_memory.decisions.toolkit import Toolkit

logger = structlog.get_logger(__name__)

#: The ONLY key a dotenv file is read for. Enablement, base URL and model are
#: deliberately NOT read from it — see the module docstring's trust split.
_DOTENV_ALLOWED_KEYS = frozenset({"OPENROUTER_API_KEY"})

#: Hosts the API key may be sent to. Kept as a constant rather than a new
#: config surface: the key belongs to exactly one provider.
_ALLOWED_BASE_URL_HOSTS = frozenset({"openrouter.ai"})


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
    project_root: str | Path | None = None,
) -> DecisionJudge:
    """Resolve the configured judge from the process env, plus a key-only dotenv.

    Enablement is decided by :func:`~trw_memory.decisions._enablement.resolve_backend_enablement`
    — process env beats project scope beats user scope beats off (see that module's docstring).
    ``project_root``, when given, also enables project-scope layers (its ``.trw/config.yaml`` and
    ``.env``); without it only the process env and the user's ``~/.trw/config.yaml`` machine
    switch are consulted, so an in-process or daemon caller with no project context still sees the
    machine switch instead of silently ignoring it. Returns :class:`JevHttpJudge` only when ALSO
    an ``OPENROUTER_API_KEY`` is found and the resolved ``TRW_JEV_BASE_URL`` is ``https`` on an
    allowlisted host; otherwise :class:`NullJudge`. ``TRW_JEV_BASE_URL``/``TRW_JEV_MODEL`` come
    from ``env`` only; ``dotenv_path``, when given, is parsed for ``OPENROUTER_API_KEY`` and
    nothing else (no ``python-dotenv`` dependency; see :mod:`trw_memory.decisions._dotenv`). See
    :func:`_inspect_base_url` for exactly which URLs are allowed.
    """
    process_env = os.environ if env is None else env

    from trw_memory.decisions._enablement import resolve_backend_enablement

    root = Path(project_root) if project_root is not None else None
    enabled, _source = resolve_backend_enablement(root, process_env)
    if not enabled:
        return NullJudge()

    api_key = process_env.get("OPENROUTER_API_KEY") or (
        parse_dotenv_subset(dotenv_path, allowed_keys=_DOTENV_ALLOWED_KEYS).get("OPENROUTER_API_KEY")
        if dotenv_path is not None
        else None
    )
    if not api_key:
        return NullJudge()

    # Lazy: JevHttpJudge (and its httpx dependency) is imported only once we know the backend is
    # actually being enabled — every disabled/off-by-default caller never loads httpx (PRD-CORE-295-FR01).
    from trw_memory.decisions._jev_http import DEFAULT_BASE_URL, DEFAULT_MODEL, JevHttpJudge

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


def toolkit_from_env(
    env: Mapping[str, str] | None = None,
    *,
    redactor: Callable[[str], str],
    dotenv_path: str | Path | None = None,
    project_root: str | Path | None = None,
    timeout_s: float = 10.0,
    session_id: str | None = None,
) -> Toolkit:
    """Build the one :class:`~trw_memory.decisions.toolkit.Toolkit` every caller uses.

    ``redactor`` is required and has no opt-out: whatever reaches an enabled backend leaves the
    machine, so the caller names the redactor that stands at that boundary. ``project_root``, when
    given, is forwarded to :func:`judge_from_env` so project-scope enablement is consulted too.
    """
    if not callable(redactor):
        raise TypeError("toolkit_from_env needs a redactor: every state and question is redacted before egress")
    from trw_memory.decisions.toolkit import Toolkit

    judge = judge_from_env(env, dotenv_path=dotenv_path, project_root=project_root)
    return Toolkit(judge, redactor=redactor, timeout_s=timeout_s, session_id=session_id)


__all__ = ["judge_from_env", "toolkit_from_env"]
