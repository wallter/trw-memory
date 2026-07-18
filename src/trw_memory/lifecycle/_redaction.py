"""Path-redaction helpers shared by consolidation surfaces."""

from __future__ import annotations

import re

__all__ = ["redact_paths"]

# Matches absolute paths under common roots, Windows drive paths, home-relative
# paths, and explicit relative prefixes. Order matters: ``../`` precedes ``./``
# so parent-relative paths are not partially matched.
_PATH_RE = re.compile(
    r"(?:/home/|/Users/|/mnt/|/tmp/|/var/|[A-Z]:\\|~/|\.\./|\./)[^\s,;\"')\]}>]*",
)


def redact_paths(text: str) -> str:
    """Replace filesystem paths with [REDACTED_PATH] before sending to LLM."""
    return _PATH_RE.sub("[REDACTED_PATH]", text)
