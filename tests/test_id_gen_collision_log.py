"""Tests for id_gen collision retry DEBUG logging.

Verifies that generate_compact_id logs a DEBUG message on collision retry.
"""

from __future__ import annotations

import logging

import pytest

from trw_memory.utils import id_gen
from trw_memory.utils.id_gen import generate_compact_id


class _ScriptedChoice:
    """Stand-in for the stdlib ``secrets`` module exposing only ``.choice``.

    Substituted for the ``secrets`` name bound inside ``id_gen`` (not the real
    global ``secrets`` module, which other concurrently-running code may still
    rely on), so ``secrets.choice(...)`` inside ``generate_compact_id`` returns
    a scripted sequence instead of a real random draw.
    """

    def __init__(self, draws: list[str]) -> None:
        self._draws = iter(draws)

    def choice(self, _seq: str) -> str:
        return next(self._draws)


def test_collision_retry_logs_debug(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Collision retry produces a DEBUG log entry.

    Forcing a 61/62-blocked slot (the prior approach) still left ~1.6% of runs
    drawing the one open slot on the first attempt -- no collision, no log line,
    a flaky failure under repeated/parallel runs. The id generator's only
    source of randomness is ``secrets.choice``; scripting it via the module's
    own ``secrets`` seam makes the FIRST draw a guaranteed collision and the
    SECOND draw the only open slot, so the retry path is exercised
    deterministically every run.
    """
    first_id = generate_compact_id(prefix="T", length=1)
    # With length=1, there are only 62 possible IDs. Fill all but one to leave
    # exactly one open slot, then force the draw order via the RNG seam.
    existing = {f"T-{c}" for c in id_gen._BASE62}
    existing.discard(first_id)

    open_char = first_id.split("-", 1)[1]
    blocked_char = next(c for c in id_gen._BASE62 if c != open_char)
    monkeypatch.setattr(id_gen, "secrets", _ScriptedChoice([blocked_char, open_char]))

    with caplog.at_level(logging.DEBUG, logger="trw_memory.utils.id_gen"):
        result = generate_compact_id(prefix="T", length=1, existing_ids=existing, max_retries=1000)

    assert result == f"T-{open_char}"
    assert result == first_id  # Only possible value
    # The first draw is a guaranteed collision -- the retry log is not probabilistic.
    assert any("id_collision_retry" in r.message for r in caplog.records)
