"""Namespace validation logic.

Extracted from ``trw_memory.namespace`` for package organisation.
"""

from __future__ import annotations

import re

from trw_memory.exceptions import ConfigError

# Valid namespace: scope:name or bare scope (global, default)
# name part: alphanumeric, hyphens, underscores (no dots, slashes, colons)
_NS_PATTERN = re.compile(r"^(project:[a-zA-Z0-9_-]+|global|default|team:[a-zA-Z0-9_-]+|org:[a-zA-Z0-9_-]+)$")

_MAX_LENGTH = 128


def validate_namespace(ns: str) -> str:
    """Validate a namespace string and return it unchanged.

    Args:
        ns: Namespace to validate.

    Returns:
        The validated namespace string.

    Raises:
        ConfigError: If *ns* does not match one of the canonical patterns.
    """
    if not ns or not ns.strip():
        raise ConfigError("namespace must not be empty")

    ns = ns.strip()

    if len(ns) > _MAX_LENGTH:
        raise ConfigError(f"namespace too long: {len(ns)} chars (max {_MAX_LENGTH})")

    if not _NS_PATTERN.match(ns):
        raise ConfigError(
            f"Invalid namespace {ns!r}. "
            "Must match project:<name>, global, default, team:<name>, or org:<name> "
            "where <name> is [a-zA-Z0-9_-]+."
        )
    return ns
