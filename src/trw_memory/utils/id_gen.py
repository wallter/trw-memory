"""Compact base-62 ID generation for memory entries."""

from __future__ import annotations

import logging
import secrets
import string

_BASE62 = string.ascii_letters + string.digits  # a-zA-Z0-9, 62 chars
_logger = logging.getLogger(__name__)


def generate_compact_id(
    prefix: str = "L",
    length: int = 4,
    existing_ids: set[str] | None = None,
    max_retries: int = 10,
) -> str:
    """Generate a compact base-62 ID like 'L-a3Fq'.

    Args:
        prefix: ID prefix (default "L" for learnings, "M" for memories).
        length: Number of random chars (default 4 = 62^4 ≈ 14.7M combinations).
        existing_ids: Set of IDs to avoid collisions with. Retry on collision.
        max_retries: Max collision retries before raising RuntimeError.

    Returns:
        ID string matching pattern '{prefix}-[a-zA-Z0-9]{length}'.

    Raises:
        RuntimeError: If max_retries exceeded (collision space exhausted).
    """
    for _attempt in range(max_retries):
        suffix = "".join(secrets.choice(_BASE62) for _ in range(length))
        candidate = f"{prefix}-{suffix}"
        if existing_ids is None or candidate not in existing_ids:
            return candidate
        _logger.debug(
            "id_collision_retry: attempt=%d candidate=%s prefix=%s",
            _attempt + 1,
            candidate,
            prefix,
        )
    raise RuntimeError(
        f"generate_compact_id: exceeded {max_retries} retries for prefix={prefix!r}; "
        "collision space may be exhausted"
    )
