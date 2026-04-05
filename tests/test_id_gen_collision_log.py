"""Tests for id_gen collision retry DEBUG logging.

Verifies that generate_compact_id logs a DEBUG message on collision retry.
"""

from __future__ import annotations

import logging

from trw_memory.utils.id_gen import generate_compact_id


def test_collision_retry_logs_debug(caplog: logging.LogRecord) -> None:  # type: ignore[type-arg]
    """Collision retry produces a DEBUG log entry."""
    # Create a set that will cause the first candidate to collide
    # We can't predict the exact ID, so we force collision by using
    # a very small set that we grow to cause guaranteed collisions
    # Instead, we use a custom approach: generate one ID, then make it
    # the "existing" set and generate again with very high probability
    # of needing a retry.
    first_id = generate_compact_id(prefix="T", length=1)
    # With length=1, there are only 62 possible IDs
    # Fill all but one to force retries
    existing = {f"T-{c}" for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}
    # Remove the first_id so there's exactly 1 slot open
    existing.discard(first_id)

    with caplog.at_level(logging.DEBUG, logger="trw_memory.utils.id_gen"):
        result = generate_compact_id(prefix="T", length=1, existing_ids=existing, max_retries=1000)

    assert result.startswith("T-")
    assert result == first_id  # Only possible value
    # With 61 out of 62 blocked, there should be collisions logged
    assert any("id_collision_retry" in r.message for r in caplog.records)
