"""Namespace validation logic.

Extracted from ``trw_memory.namespace`` for package organisation.
"""

from __future__ import annotations

import re

from trw_memory.exceptions import ConfigError

# Valid namespace: scope:name or bare scope (global, default)
# name part: alphanumeric, hyphens, underscores (no dots, slashes, colons)
# user:<id> (PRD-CORE-185 FR03) is the machine-local user-space tier scope.
_NS_PATTERN = re.compile(
    r"^(project:[a-zA-Z0-9_-]+|global|default|"
    r"team:[a-zA-Z0-9_-]+|org:[a-zA-Z0-9_-]+|user:[a-zA-Z0-9_-]+)$"
)

_MAX_LENGTH = 128

#: The namespace a memory row carries when its writer named none. Under schema 5
#: identity is ``(namespace, id)`` (PRD-CORE-245 FR01), so every read and write
#: must name a namespace; this is the canonical name for "the unnamed one" and
#: exists so that fact is one constant rather than a literal at each call site.
DEFAULT_NAMESPACE = "default"


def validate_namespace(ns: str) -> str:
    """Validate a namespace string and return it unchanged.

    Whitespace is refused, never stripped (PRD-CORE-308 S4): an accepted name
    is byte-identical to its input, so a caller that discards the return value
    cannot authorize one spelling and store under another.

    Raises:
        ConfigError: If *ns* does not match one of the canonical patterns.
    """
    if not isinstance(ns, str):
        raise ConfigError("namespace must be a string")

    if not ns:
        raise ConfigError("namespace must not be empty")

    if len(ns) > _MAX_LENGTH:
        raise ConfigError(f"namespace too long: {len(ns)} chars (max {_MAX_LENGTH})")

    if not _NS_PATTERN.fullmatch(ns):  # fullmatch: ``$`` alone admits a trailing newline
        raise ConfigError(
            f"Invalid namespace {ns!r}. "
            "Must match project:<name>, global, default, team:<name>, org:<name>, "
            "or user:<name> where <name> is [a-zA-Z0-9_-]+."
        )
    return ns
