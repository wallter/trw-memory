"""R9 #18: the distilled weight has ONE default, owned by ``source_policy``, on both recall routes.

The parity tests pin the two routes to each other under an explicit env override (1.0, 0.5). This
pins them to the shared DEFAULT (no override set), so a route that stopped resolving through
``SourcePolicy`` and fell back to its own literal would fail here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.retrieval.source_policy import DEFAULT_DISTILLED_WEIGHT, DEFAULT_SOURCE_WEIGHTS, SourcePolicy

_ENV = "TRW_MEMORY_DISTILLED_RECALL_WEIGHT"


def test_the_default_is_one_named_constant_the_policy_table_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)  # the default, not an operator's override
    assert DEFAULT_SOURCE_WEIGHTS["git_distilled"] == DEFAULT_DISTILLED_WEIGHT
    assert SourcePolicy.resolve().weights["git_distilled"] == DEFAULT_DISTILLED_WEIGHT


@pytest.mark.asyncio
async def test_both_routes_weight_distilled_rows_by_the_shared_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import _source_weighting_support as support

    monkeypatch.setenv("MEMORY_RRF_IMPORTANCE_ALPHA", "1.0")
    support.stub_cross_encoder(monkeypatch, None)
    store = support.make_client(tmp_path, monkeypatch)
    support.seed(store, support.PARITY_ROWS + support.PARITY_PADDING)

    monkeypatch.setenv(_ENV, "1.0")
    basis = {support.row_id(row): float(row["score"]) for row in support.daemon_recall(store)["memories"]}
    unweighted_order = [support.row_id(row) for row in await support.library_recall(store)]

    monkeypatch.delenv(_ENV)
    daemon = support.daemon_recall(store)["memories"]
    library = await support.library_recall(store)

    # Daemon route: every distilled row's reported score is the pipeline basis x the shared default.
    distilled = [row for row in daemon if support.is_distilled(row)]
    assert distilled
    for row in daemon:
        factor = DEFAULT_DISTILLED_WEIGHT if support.is_distilled(row) else 1.0
        assert float(row["score"]) == pytest.approx(basis[support.row_id(row)] * factor)
    # Library route: the same order, and the default moved at least one distilled row (not vacuous).
    library_order = [support.row_id(row) for row in library]
    assert library_order == [support.row_id(row) for row in daemon]
    assert library_order != unweighted_order
