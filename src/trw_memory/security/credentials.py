"""The one credential detector: every secret shape trw-memory and trw-mcp recognise.

Two consumers, one pattern set (so coverage cannot diverge):

* the memory write gate (:func:`trw_memory.security.pii.detect_pii`) reports every
  ``blocking`` pattern below as ``PIIType.API_KEY``, which BLOCKS the store; and
* :func:`mask_credentials` rewrites every pattern to a ``<REDACTED:...>`` marker; it is
  the credential stage of trw-mcp's ``redact_secrets`` and the redaction applied to any
  learn-journal record kept on disk after a refusal.

Add a new credential shape HERE. Pure functions, no I/O, idempotent: every pattern
refuses to re-consume a ``<REDACTED:...>`` placeholder.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from trw_memory.security._scan_normalize import (
    mask_with_invisible_splits as mask_with_invisible_splits,
)
from trw_memory.security._scan_normalize import (
    spans_with_invisible_splits,
)

# Generic ``<prefix>[-_]<20+ alnum>`` credential shape, tolerating ONE optional
# environment/scope segment between the prefix and the random body.
#
# That optional segment is why ``sk_live_…`` (Stripe) and ``sk-proj-…`` (OpenAI)
# escaped until 2026-07-30. ``sk`` was already in the prefix list — the author
# plainly intended to catch them — but requiring 20+ alnum IMMEDIATELY after the
# separator meant the pattern died on ``live``/``proj``, which every real provider
# key carries. The tokens were still *detected*, as HIGH_ENTROPY; but only
# ``PIIType.API_KEY`` is in ``_runtime_pii.BLOCKING_PII_TYPES`` and only the regex
# types are masked by ``strip_pii``. So a live Stripe key was persisted verbatim
# AND published verbatim to the platform, while an AWS key in the identical
# position was blocked and masked. Same threat class, opposite handling, decided
# by which regex happened to match.
#
# The optional segment is a CLOSED SET of real provider scope words, not a generic
# `[a-zA-Z0-9]{1,12}`. A generic segment is indistinguishable from an ordinary
# snake_case identifier, and `PIIType.API_KEY` BLOCKS the write — so the first
# draft of this fix rejected `pk_users_organizationmembership`,
# `key_error_troubleshootingnotes` and `token_cache_invalidationstrategy`, losing
# the learning outright. Every provider shape this exists to catch
# (`sk_live_`, `sk_test_`, `rk_live_`, `sk-proj-`) is covered by the closed set.
_SECRET_SCOPE_WORDS = "live|test|proj|prod|dev|sandbox|staging"  # noqa: S105 — regex alternation, not a credential
# Word-anchored HERE, in the shared constant, so `detect_pii` and `strip_pii` apply
# identical boundaries. `detect_pii` used to wrap it in `\b...\b` while `strip_pii`
# applied it bare, so `strip_pii` masked mid-token substrings the detector would
# not flag — a divergence in the one constant whose comment claims it exists to
# prevent divergence.
_SECRET_PREFIX_PATTERN = (
    r"\b(?:sk|pk|rk|api|key|token|secret)[-_]"
    rf"(?:(?:{_SECRET_SCOPE_WORDS})[-_])?[a-zA-Z0-9]{{20,}}\b"
)
# Provider-specific shapes that lack a "<prefix>[-_]" separator and fall below
# the Shannon-entropy backstop (GitHub PATs, AWS access key IDs), plus shapes that
# sit ABOVE it and were therefore mis-typed as HIGH_ENTROPY rather than API_KEY
# (Slack, Google). Anchored + bounded (no nested quantifiers) so they stay
# ReDoS-free.
_PROVIDER_SECRET_PATTERN = (
    r"\b(?:gh[posru]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b"  # noqa: S105 — regex, not a credential
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
    r"|\bxox[baprs]-[A-Za-z0-9-]{20,}\b"
    r"|\bAIza[A-Za-z0-9_-]{35}\b"
)


# ---------------------------------------------------------------------------
# redact_secrets — the credential-pattern set (formerly tools/_feedback_redaction.py)
# ---------------------------------------------------------------------------
# Single chokepoint NFR01 mandates for secret hygiene. Pure functions (no I/O,
# idempotent) so they are trivially unit-testable. Patterns compiled at import
# so redaction stays O(n) over the message body per call.
_LICENSE_KEY_RE = re.compile(r"trw_lic_\S+")
# Classic API-key formats: Stripe secret/publishable keys, AWS access-key ids,
# and the hyphenated ``sk-`` family (OpenAI ``sk-proj-…``, Anthropic
# ``sk-ant-api03-…``).
#
# This pattern used to carry a comment rejecting provider prefixes outright as
# "a per-vendor token zoo", on the rationale that the env-var pattern below
# already catches ``OPENAI_API_KEY=…`` / ``GITHUB_TOKEN=…``. That rationale is
# sound and it is why this list stays short — but it holds only for the
# ASSIGNMENT form. The shapes that actually reach a feedback box carry no
# ``=`` at all: a pasted ``curl -H 'Authorization: Bearer eyJ…'``, an HTTP
# trace, or plain prose ("the token eyJ… was rejected"). The env-var rule
# cannot anchor on any of them, so those pastes leaked in clear text.
#
# The line drawn now is not "every vendor" but UNAMBIGUOUS BY CONSTRUCTION:
# a fixed prefix the vendor publishes for secret scanning, or a structural
# format (below). A prefix that could plausibly occur in prose does not qualify.
#
# ``sk-`` specifically is the near-miss this file already half-covered: it
# matched Stripe's UNDERSCORE ``sk_live_`` while OpenAI's HYPHEN ``sk-proj-``
# walked straight through, which is the single most likely secret in a bug
# report filed against an AI framework.
_API_KEY_RE = re.compile(
    r"(?:sk_(?:live|test)_\S+"
    r"|pk_(?:live|test)_\S+"
    r"|AKIA[0-9A-Z]{16}"
    r"|\bsk-[A-Za-z0-9_-]{16,}"  # OpenAI / Anthropic (hyphen form)
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"  # GitHub PAT / OAuth / server / refresh
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}"  # GitHub fine-grained PAT
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}"  # Slack bot/user/app/refresh/legacy
    r"|\bAIza[A-Za-z0-9_-]{35}"  # Google API key
    # TRW platform key. Minted as ``trw_`` or ``trw_dk_`` (device flow) + ``secrets.token_urlsafe(32)``:
    # 43 chars of ``[A-Za-z0-9_-]``, so ``-``/``_`` occur INSIDE the body. The lookahead
    # demands an uppercase letter or digit (a random body has one with probability ~1), so a
    # long snake_case identifier such as a module or tool name is never consumed.
    r"|\btrw_(?:dk_)?(?=[A-Za-z0-9_-]*[A-Z0-9])[A-Za-z0-9_-]{43,}"
    r"|\btrw_[A-Za-z0-9]{24,}"  # legacy TRW key shape: ``trw_`` + 24+ alnum, no separators
    r")"
)
# A PEM private-key block. Structural, vendor-neutral, and the highest-severity
# thing a user can paste: the whole block collapses, header and footer included,
# so no base64 body survives. DOTALL because the body spans lines.
_PEM_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
# An ``Authorization:`` header value. The scheme (Bearer/Token/Basic/…) is
# preserved because it is diagnostically useful and is not itself a secret;
# everything after it is redacted whole. This is the form a user pastes when
# reporting a failing API call, and no ``key=value`` rule can reach it. The
# negative lookahead keeps the pass idempotent on its own placeholder.
# The "already redacted" lookahead has to span the OPTIONAL scheme, not just sit
# in front of the value: with the guard on the value alone, a second pass over
# ``Authorization: Bearer <REDACTED:authorization>`` backtracks — the optional
# scheme group gives up ``Bearer``, which then satisfies the value slot — and the
# line grows a second placeholder on every pass. Guarding the whole tail keeps
# redaction idempotent, which the NFR pins as hard as zero-false-negative.
_AUTH_HEADER_RE = re.compile(
    r"(?P<key>Authorization\s*:\s*)"
    r"(?!(?:(?:Bearer|Token|Basic|Digest|ApiKey)\s+)?<REDACTED:)"
    r"(?:(?P<scheme>Bearer|Token|Basic|Digest|ApiKey)(?P<ws>\s+))?"
    r"\S+",
    re.IGNORECASE,
)
# A JSON Web Token. Structural, not per-vendor: ``eyJ`` is base64 for ``{"``,
# so a three-segment dotted run starting with it is a JWT header by
# construction and effectively cannot be benign prose. Catches the bare token
# in a narrative error message, which the header rule above does not see.
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]*")
# A bare ``Bearer``/``Token`` value with no ``Authorization:`` header in front
# of it (e.g. a raw ``curl -H 'Bearer <token>'`` paste, or prose narrating a
# rejected token). The header rule above only fires with the header present;
# this catches the same credential when the header text was trimmed away.
_BARE_BEARER_RE = re.compile(
    r"\b(Bearer|Token)(\s+)(?:(?=[A-Za-z0-9._~+/=-]*\d)[A-Za-z0-9._~+/=-]{8,}|[A-Za-z0-9._~+/=-]{16,})",
    re.IGNORECASE,
)
# Connection-string credentials: scheme://user:password@host. Redact the
# user:password segment whole (preserve scheme + host for diagnostics). The
# password may contain URL-encoded chars / symbols, so it matches any non-`@`,
# non-`/` run. The username group is ``*`` (not ``+``) so an empty-username
# URL (``postgres://:pw@host``) still collapses. ``host`` is whatever follows
# the ``@``.
_CONN_STR_RE = re.compile(
    r"(?P<scheme>postgres|postgresql|mysql|mongodb|redis"
    r"|amqp|amqps|ldap|ldaps|ftp|sftp|mssql|sqlserver)://[^/\s:@]*:[^/\s@]+@",
    re.IGNORECASE,
)
# Query-string credentials: ``?password=…`` / ``&token=…`` etc. — credentials
# smuggled into a URL query rather than the userinfo segment. Preserve the
# separator + key for diagnostics, redact the value (stops at the next
# ``&``/``#``/whitespace). Runs alongside _CONN_STR_RE in the connection-string
# stage so a URL with BOTH userinfo and query creds is fully scrubbed.
_QUERY_CRED_RE = re.compile(
    r"(?P<lead>[?&])(?P<key>password|passwd|secret|token|api_key)=(?P<val>[^\s&#]+)",
    re.IGNORECASE,
)
# JSON-embedded secrets: "password": "…" / "api_key": "…" etc. Preserve the key
# for diagnostics, redact the value. Case-insensitive on the key; the value is
# any run of non-quote characters (handles empty and multi-word values). The
# key alternation accepts snake_case, kebab-case, AND camelCase variants
# (apiKey/apiToken/clientSecret/refreshToken/authToken) so a camelCase JSON
# secret is not a false-negative. ``client_id`` is deliberately NOT included —
# an id is an identifier, not a secret.
_JSON_SECRET_RE = re.compile(
    r"(?P<key>\"(?:"
    r"password|secret|token|private_key"
    r"|api[_-]?key|api[_-]?token|auth[_-]?token|client[_-]?secret|refresh[_-]?token"
    r"|access_key|access_token"
    # Credential-carrying HTTP headers serialized as JSON (a header dict in a log or payload).
    r"|(?:proxy[_-]?)?authorization|(?:set[_-]?)?cookie"
    r")\")"
    r"(?P<sep>\s*:\s*)"
    r"\"(?:[^\"\\]|\\.)*\"",  # a JSON string: escaped quotes and backslashes stay inside it
    re.IGNORECASE | re.DOTALL,
)
# Sensitive env-var KEY=value tokens. The key may carry a prefix
# (``DB_PASSWORD``, ``OPENAI_API_KEY``, ``GITHUB_TOKEN``): a leading ``\b``
# would never match between two word characters (``_`` is a word char), so a
# prefixed key would silently leak its value. We instead anchor on a non-key
# boundary (start-of-string or a non-``[A-Za-z0-9_]`` char) and allow an
# optional ``WORD_`` prefix segment before the sensitive keyword. The value
# captures an optionally-quoted token so ``PASSWORD="multi word secret"`` is
# redacted whole rather than leaking everything after the first space.
#
# A ``key=value`` shape also describes a URL query credential (``?password=…``)
# and an already-substituted placeholder (``password=<REDACTED:credentials>``).
# Those are handled by the connection-string stage which runs FIRST, so the
# value is a ``<REDACTED:…>`` marker by the time this pass runs. A negative
# lookahead on the value skips an already-redacted token: that (a) preserves
# the query-credential placeholder + its key for diagnostics instead of
# re-collapsing it into ``<REDACTED:env>``, and (b) keeps the whole pass
# idempotent (re-running never re-consumes a placeholder).
_ENV_RE = re.compile(
    r"(?:^|(?<=[^A-Za-z0-9_]))"  # boundary: start, or a non-identifier char
    r"(?:[A-Za-z0-9]*_)*"  # optional prefix segments (DB_, OPENAI_, AWS_SECRET_, ...)
    r"(?:PASSWORD|SECRET|TOKEN|API[_-]?KEY|ACCESS[_-]?KEY)"
    r"(?:[_-]?(?:KEY|TOKEN))?"  # optional KEY/TOKEN suffix (SECRET_KEY, ACCESS_TOKEN)
    r"(?:\s*=\s*|\s*:\s*(?=[\"']?[0-9a-fA-F]{32}))"  # ``NAME=value``, or ``secret: <32+ hex>`` (a digest behind a label)
    r"(?!<REDACTED:)"  # already-redacted value (query cred / 2nd pass): skip
    r"(?:\"[^\"]*\"|'[^']*'|\S+)",  # quoted value (any chars) or bare token
    re.IGNORECASE,
)


_API_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    _API_KEY_RE,
    re.compile(_SECRET_PREFIX_PATTERN, re.IGNORECASE),
    re.compile(_PROVIDER_SECRET_PATTERN),
)
#: Shapes the memory write gate REFUSES: high-confidence, vendor-published token formats and
#: structural secrets that cannot be a placeholder (a key, a JWT, a PEM private-key block).
_BLOCKING_PATTERNS: tuple[re.Pattern[str], ...] = (
    _PEM_KEY_RE,
    _LICENSE_KEY_RE,
    _JWT_RE,
    *_API_KEY_PATTERNS,
)
# Shapes that are MASKED in stored text, never blocked. ``KEY=value``, ``scheme://user:pw@host``,
# ``?token=…``, ``"password": "…"`` (JSON: value-based, the key names the field and the value is
# replaced) and ``Authorization:`` headers are routinely written as placeholders in real
# learnings ("API_KEY=x"), so a block would lose them; masking keeps the lesson and drops the
# value. The bare ``Bearer``/``Token`` catch is not applied to stored text at all: the word
# "token" followed by a long word is ordinary prose. It stays in :func:`mask_credentials`
# for egress, where over-masking is cheap.
_MASK_ONLY_SUBS: tuple[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...] = (
    (_CONN_STR_RE, r"\g<scheme>://<REDACTED:credentials>@"),
    (_QUERY_CRED_RE, r"\g<lead>\g<key>=<REDACTED:credentials>"),
    (_JSON_SECRET_RE, r'\g<key>\g<sep>"<REDACTED:json_secret>"'),
    (
        _AUTH_HEADER_RE,
        lambda m: (
            m.group("key")
            + ((m.group("scheme") + m.group("ws")) if m.group("scheme") else "")
            + "<REDACTED:authorization>"
        ),
    ),
    (_ENV_RE, "<REDACTED:env>"),
)


def credential_spans(text: str) -> list[tuple[int, int]]:
    """Non-overlapping ``(start, end)`` spans of every credential the write gate must block on.

    Patterns overlap (a GitHub token is both an ``_API_KEY_RE`` and a provider shape), so
    overlapping spans merge: one credential is one span.
    """
    spans = sorted(m.span() for pattern in _BLOCKING_PATTERNS for m in pattern.finditer(text))
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def credential_spans_with_invisible_splits(text: str) -> list[tuple[int, int]]:
    """Credential spans in original text, including secrets split by invisible characters."""
    return spans_with_invisible_splits(text, credential_spans)


def mask_low_confidence(text: str) -> str:
    """Mask the placeholder-prone shapes (``KEY=value``, URL credentials, JSON secrets, auth headers).

    The store path applies this to every stored text field. High-confidence tokens are left for
    the write gate to BLOCK.
    """
    if not text:
        return text
    for pattern, repl in _MASK_ONLY_SUBS:
        text = pattern.sub(repl, text)
    return text


def mask_credentials(text: str) -> str:
    """Replace every credential shape in *text* with a ``<REDACTED:kind>`` marker.

    A credential split by an invisible format character is masked too (PII-INVISIBLE-SPLIT): the text is
    also masked with format characters removed, and the union of both is masked.
    """
    return mask_with_invisible_splits(text, mask_credentials_as_written)


def mask_credentials_as_written(text: str) -> str:
    """The pattern pass over *text* exactly as given: no invisible-character handling. A building block for
    composing maskers; anything sending text off the box uses :func:`mask_credentials`."""
    if not text:
        return text
    text = _PEM_KEY_RE.sub("<REDACTED:private_key>", text)
    text = _LICENSE_KEY_RE.sub("<REDACTED:license_key>", text)
    for pattern, repl in _MASK_ONLY_SUBS[:-1]:
        text = pattern.sub(repl, text)
    text = _JWT_RE.sub("<REDACTED:jwt>", text)
    for pattern in _API_KEY_PATTERNS:
        text = pattern.sub("<REDACTED:api_key>", text)
    # The whitespace after the scheme is kept as written (a rewrite would shift every later character).
    text = _BARE_BEARER_RE.sub(r"\1\2<REDACTED:bearer>", text)
    return _MASK_ONLY_SUBS[-1][0].sub("<REDACTED:env>", text)


def mask_credentials_deep(value: object) -> object:
    """*value* with every string inside it (lists and dicts included) passed through :func:`mask_credentials`."""
    if isinstance(value, str):
        return mask_credentials(value)
    if isinstance(value, list):
        return [mask_credentials_deep(item) for item in value]
    if isinstance(value, dict):
        return {key: mask_credentials_deep(item) for key, item in value.items()}
    return value
