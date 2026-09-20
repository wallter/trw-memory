"""REF-005: every dedup entry point validates thresholds the same way (PRD-CORE-042).

PRD-CORE-042 FR (merge >= skip): "log a WARNING and reset both to defaults (0.95 and
0.85)". ``check_duplicate`` did both; ``batch_dedup`` reset silently.

The fixture discriminates the reset: the pair's cosine is 0.9. Under the configured
(invalid) skip=0.80 the pair would be a SKIP; under the required defaults it is a MERGE.
"""

from __future__ import annotations

import math

import pytest
from structlog.testing import capture_logs

from trw_memory.lifecycle.dedup import batch_dedup, check_duplicate
from trw_memory.models.config import MemoryConfig

from ._test_dedup_support import StubEmbedder, make_entry

_COS_09 = [0.9, math.sqrt(1.0 - 0.81), 0.0]


def _embedder() -> StubEmbedder:
    embedder = StubEmbedder(available=True)
    embedder.set_vector("alpha entry ", [1.0, 0.0, 0.0])
    embedder.set_vector("alpha entry dupe ", _COS_09)
    return embedder


def _check(config: MemoryConfig) -> str:
    return check_duplicate("alpha entry dupe", [make_entry("e1", "alpha entry")], _embedder(), config=config).action


def _batch(config: MemoryConfig) -> str:
    """The batch pass records its verdict on the obsoleted newer entry: a SKIP marks it
    ``Auto-obsoleted: duplicate``, a MERGE marks it ``Auto-merged`` into the survivor."""
    entries = [make_entry("e1", "alpha entry"), make_entry("e2", "alpha entry dupe")]
    result = batch_dedup(entries, _embedder(), config=config)
    dupe = next(e for e in result["updated_entries"] if e.id == "e2")  # type: ignore[union-attr]
    if "[Auto-merged into e1" in dupe.detail:
        return "merge"
    return "skip" if "[Auto-obsoleted: duplicate of e1" in dupe.detail else dupe.detail


ENTRY_POINTS = [pytest.param(_check, id="check_duplicate"), pytest.param(_batch, id="batch_dedup")]


def _threshold_warnings(logs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [log for log in logs if log.get("event") == "dedup_threshold_invalid"]


@pytest.mark.unit
@pytest.mark.parametrize("dedup", ENTRY_POINTS)
def test_invalid_thresholds_warn_once_and_reset_to_the_defaults(dedup) -> None:  # type: ignore[no-untyped-def]
    with capture_logs() as logs:
        action = dedup(MemoryConfig(dedup_skip_threshold=0.80, dedup_merge_threshold=0.92))
    assert action == "merge", "cos 0.9 merges only under the defaults (0.95/0.85); raw skip=0.80 would skip it"
    warnings = _threshold_warnings(logs)
    assert len(warnings) == 1 and warnings[0]["log_level"] == "warning"
    assert (warnings[0]["merge"], warnings[0]["skip"]) == (0.92, 0.80)


@pytest.mark.unit
@pytest.mark.parametrize("dedup", ENTRY_POINTS)
def test_valid_thresholds_are_used_as_configured_without_a_warning(dedup) -> None:  # type: ignore[no-untyped-def]
    with capture_logs() as logs:
        action = dedup(MemoryConfig(dedup_skip_threshold=0.88, dedup_merge_threshold=0.80))
    assert action == "skip", "a valid config is honoured: cos 0.9 >= skip 0.88"
    assert _threshold_warnings(logs) == []
