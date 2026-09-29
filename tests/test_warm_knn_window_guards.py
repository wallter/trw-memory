"""PRD-CORE-318 FR02b guards: malformed sidecar rows and withheld ids around the warm KNN window.

Review r2 P1: the stop rule's ceiling folds EVERY live sidecar row into per-version maxima
(``ParsedSidecar.maxima``), including rows the hybrid pool already covers and discovery never
reads. A covered row whose ``tags`` is not a list aborted recall with ``TypeError``. A malformed
row is now skipped as a bound input with a logged reason, and never aborts the maxima.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timezone
from typing import Any

import pytest
from structlog.testing import capture_logs

from tests.test_recall_library_path_perf import (
    DURABLE,
    SPACE,
    _at_cosine,
    _bounded_and_unbounded,
    _entry,
    _policy,
    _put_warm,
    _unit,
    tiers,  # noqa: F401  (fixture)
)
from trw_memory.embeddings.provenance import VectorProvenance
from trw_memory.lifecycle.tiers import _runtime
from trw_memory.lifecycle.tiers._manager import TierManager
from trw_memory.lifecycle.tiers._manager_search import WindowRank
from trw_memory.lifecycle.tiers._warm import WarmTierStore
from trw_memory.lifecycle.tiers._warm_discovery import _StopRule
from trw_memory.lifecycle.tiers._warm_sidecar_cache import ParsedSidecar, ScoreMaxima
from trw_memory.models.config import MemoryConfig
from trw_memory.retrieval.recall_selection import LocalCandidate
from trw_memory.storage.sqlite_backend import SQLiteBackend

_VALID = {"id": "ok", "content": "needle", "importance": 0.4, "last_accessed_at": "2026-01-02", "tags": ["t"]}
_SKIPPED = "warm_tier_maxima_malformed_row_skipped"


def _sidecar(*entries: dict[str, object]) -> ParsedSidecar:
    parsed = ParsedSidecar({}, 1, False, 0)
    parsed.extend([{"id": e["id"], "entry": e} for e in entries])
    return parsed


def test_review_r2_repro_a_covered_row_with_int_tags_does_not_abort_the_stop_rule() -> None:
    """The reviewer's repro: a valid row plus a malformed covered row, then ``_StopRule.holds``."""
    parsed = _sidecar(_VALID, {"id": "covered", "content": "x", "tags": 1})
    rank = WindowRank(_policy(), query_tokens=["needle"], query_embedding=None, config=MemoryConfig())

    with capture_logs() as logs:
        held = _StopRule(parsed, rank).holds([("ok", 0.1)], 1)

    assert isinstance(held, bool)
    assert parsed.maxima() == ScoreMaxima().including({"entry": _VALID})  # the bad row is not a bound input
    assert [e["entry_id"] for e in logs if e["event"] == _SKIPPED] == ["covered"]


#: Every covered-row field the ceiling reads, at values JSON can carry that the row schema rejects.
_MALFORMED = [
    *(("tags", v) for v in (1, 1.5, True, "source_kind:git", {"source_kind:git": 1})),
    *(("importance", v) for v in ("invalid", [1], {"x": 1}, "nan", "inf", "-inf", True, "1e400")),
    *(("last_accessed_at", v) for v in ("garbage", 12, [1], {"a": 1}, "9999-99-99", "0001-01-01", "T", "")),
    *(("created_at", v) for v in ("garbage", -1, [None], "2026-13-01T00:00")),
    *(("metadata", v) for v in (1, "x", [1], {"source_kind": 1}, {"source": ["distilled:git:"]})),
    *(("embedding", v) for v in ("x", [1, 2], [[1.0]], {"a": 1}, 3)),
]


@pytest.mark.parametrize(("field", "value"), _MALFORMED, ids=[f"{f}={v!r}" for f, v in _MALFORMED])
@pytest.mark.parametrize("appended", [False, True], ids=["full-pass", "extend"])
def test_malformed_fields_never_abort_the_maxima_and_never_tighten_them(
    field: str, value: object, appended: bool
) -> None:
    """Both paths that fold a row (the first full pass, and ``extend`` on a cached version) survive
    every malformed value, and the maxima still bound every well-formed row (soundness)."""
    bad = {"id": "bad", "content": "x", field: value}
    if appended:
        parsed = _sidecar(_VALID)
        parsed.maxima()  # cached, so the bad row is folded by extend()
        parsed.extend([{"id": "bad", "entry": bad}])
    else:
        parsed = _sidecar(_VALID, bad)

    maxima = parsed.maxima()

    valid = ScoreMaxima().including({"entry": _VALID})
    assert maxima.importance >= valid.importance
    assert maxima.newest_access is not None and maxima.newest_access >= date(2026, 1, 2)
    assert valid.families <= maxima.families


def test_a_well_formed_row_is_never_skipped() -> None:
    with capture_logs() as logs:
        _sidecar(_VALID, {**_VALID, "id": "ok2", "tags": []}, {**_VALID, "id": "ok3", "tags": None}).maxima()
    assert [e for e in logs if e["event"] == _SKIPPED] == []


def test_recall_with_a_malformed_covered_row_equals_the_full_scan(
    tiers: tuple[TierManager, SQLiteBackend],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through ``TierManager.search``: the malformed row is in the pool (covered), so
    the window runs with the stop rule and must return exactly the unbounded result."""
    manager, backend = tiers
    stamp = datetime.now(timezone.utc)
    rng = random.Random(11)
    for i in range(40):
        entry = _entry(f"u{i:02}", kind=DURABLE, importance=0.5).model_copy(update={"created_at": stamp})
        _put_warm(manager, backend, entry, _unit(rng))
    bad = {**_entry("covered-bad").model_dump(mode="json"), "tags": 1, "metadata": {}}  # tags decide its family
    vector = _unit(rng)
    proof = VectorProvenance.for_vector(SPACE, "needle ", vector)
    manager._warm_store.warm_add("covered-bad", bad, vector, provenance=proof)
    real = TierManager.search
    monkeypatch.setattr(
        TierManager, "search", lambda self, *a, **k: real(self, *a, **{**k, "covered_ids": frozenset({"covered-bad"})})
    )
    consulted: list[bool] = []
    holds = _StopRule.holds
    monkeypatch.setattr(_StopRule, "holds", lambda self, *a: consulted.append(holds(self, *a)) or consulted[-1])

    bounded, unbounded = _bounded_and_unbounded(manager, backend, _unit(random.Random(12)), 2, monkeypatch)

    assert consulted  # non-vacuous: the bounded run folded the malformed row into the ceiling
    assert bounded == unbounded
    assert len(bounded) == 2 and "covered-bad" not in bounded


def test_an_id_the_read_layer_withholds_never_enters_recall_through_the_knn_window(
    tiers: tuple[TierManager, SQLiteBackend],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CORE-318 FR02 r2 P1-3 x FR02b: the withheld row is the nearest (uniform importance), so the
    bounded KNN window holds it; its canonical read is withheld (as a quarantined row's is) and its
    warm snapshot must not stand in for it."""
    manager, backend = tiers
    stamp = datetime.now(timezone.utc)
    rng = random.Random(21)
    for i in range(60):
        entry = _entry(f"w{i:02}", kind=DURABLE, importance=0.5).model_copy(update={"created_at": stamp})
        _put_warm(manager, backend, entry, _unit(rng))
    poison = _entry("poison", kind=DURABLE, importance=0.5).model_copy(update={"created_at": stamp})
    _put_warm(manager, backend, poison, _at_cosine(0.999, 1))
    windows: list[tuple[int | None, list[str]]] = []
    scan = WarmTierStore.discovery_entries

    def recording(self: WarmTierStore, *args: Any, **kwargs: Any) -> Any:
        found = scan(self, *args, **kwargs)
        windows.append((kwargs.get("limit"), [str(row.get("id")) for row in found]))
        return found

    monkeypatch.setattr(WarmTierStore, "discovery_entries", recording)

    def recall() -> list[str]:
        found = _runtime.tier_candidates(
            manager._config,
            "default",
            backend,
            query="needle",
            tags=None,
            limit=2,
            invocation=_policy(),
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            query_space=SPACE,
        )
        return [c.entry.id for c in found if isinstance(c, LocalCandidate)]

    assert recall()[0] == "poison"  # non-vacuous: recallable first, through a bounded window
    limit, window = windows[-1]
    assert limit is not None and "poison" in window and len(window) < 61
    real = backend.get_many
    monkeypatch.setattr(
        backend,
        "get_many",
        lambda ids, *, namespace: {k: v for k, v in real(ids, namespace=namespace).items() if k != "poison"},
    )

    after = recall()

    assert "poison" in windows[-1][1]  # the window still offers it ...
    assert "poison" not in after and after  # ... and discovery drops it, keeping the rest
