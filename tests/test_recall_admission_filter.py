"""PRD-DIST-2049 c802 — recall-time confidence / currentness admission filter.

Unit tests for ``apply_admission_filter`` helper + integration tests via
``MemoryClient.recall`` to confirm both env-var (config) and per-call kwarg
routes plumb correctly.
"""

from __future__ import annotations

from typing import cast

import pytest

from trw_memory._client_recall_helpers import apply_admission_filter
from trw_memory.client import MemoryClient, MemoryResultDict


def _r(memory_id: str, *, confidence: float | None = None, currentness: str | None = None) -> MemoryResultDict:
    meta: dict[str, object] = {}
    if confidence is not None:
        meta["confidence"] = confidence
    if currentness is not None:
        meta["currentness_status"] = currentness
    return cast(
        "MemoryResultDict",
        {
            "memory_id": memory_id,
            "content": "x",
            "detail": "",
            "tags": [],
            "importance": 0.5,
            "score": 0.5,
            "created_at": "",
            "updated_at": "",
            "namespace": "test",
            "source": "local",
            "last_accessed_at": "",
            "q_value": 0.0,
            "q_observations": 0,
            "recurrence": 1,
            "access_count": 0,
            "metadata": meta,
        },
    )


class TestApplyAdmissionFilterUnit:
    def test_default_off_returns_input_unchanged(self) -> None:
        results = [_r("a"), _r("b", confidence=0.5), _r("c", currentness="historical_only")]
        out = apply_admission_filter(results, confidence_floor=None, exclude_historical_only=False)
        # bit-for-bit identity (same list, same order, same dicts)
        assert out is results

    def test_confidence_floor_suppresses_below(self) -> None:
        results = [
            _r("hi", confidence=0.95),
            _r("low", confidence=0.65),
            _r("none"),  # no confidence -> 0.0 -> dropped
            _r("mid", confidence=0.7),
        ]
        out = apply_admission_filter(results, confidence_floor=0.7, exclude_historical_only=False)
        kept_ids = [r["memory_id"] for r in out]
        assert kept_ids == ["hi", "mid"]

    def test_historical_only_suppression(self) -> None:
        results = [
            _r("current", currentness="current"),
            _r("hist", currentness="historical_only"),
            _r("unset"),  # no currentness key -> kept
        ]
        out = apply_admission_filter(results, confidence_floor=None, exclude_historical_only=True)
        kept_ids = [r["memory_id"] for r in out]
        assert kept_ids == ["current", "unset"]

    def test_both_filters_or_semantics(self) -> None:
        results = [
            _r("ok", confidence=0.9, currentness="current"),
            _r("hist", confidence=0.9, currentness="historical_only"),  # suppressed by historical
            _r("low", confidence=0.5, currentness="current"),  # suppressed by confidence
            _r("both", confidence=0.5, currentness="historical_only"),  # suppressed by both
        ]
        out = apply_admission_filter(
            results,
            confidence_floor=0.7,
            exclude_historical_only=True,
        )
        kept_ids = [r["memory_id"] for r in out]
        assert kept_ids == ["ok"]

    def test_malformed_confidence_falls_back_to_zero(self) -> None:
        results = [_r("a"), _r("b", confidence=0.8)]
        # Manually corrupt the confidence to a non-numeric value
        results[0]["metadata"]["confidence"] = "not a float"  # type: ignore[typeddict-item]
        out = apply_admission_filter(results, confidence_floor=0.5, exclude_historical_only=False)
        # 'a' has malformed confidence → treated as 0.0 → dropped; 'b' kept
        assert [r["memory_id"] for r in out] == ["b"]


class TestRecallFilterIntegration:
    async def test_recall_filter_default_off_preserves_behavior(self, client: MemoryClient) -> None:
        # importance carries the trw-distill ingest confidence signal
        await client.store("first record", importance=0.9)
        await client.store("second record zombie", importance=0.65)
        results = await client.recall("record")
        assert len(results) >= 2, f"expected >=2 results with filter OFF, got {[r['memory_id'] for r in results]}"

    async def test_recall_per_call_confidence_floor_suppresses_zombie(self, client: MemoryClient) -> None:
        await client.store("alpha record", importance=0.9)
        await client.store("alpha record zombie", importance=0.65)
        filtered = await client.recall("alpha", confidence_floor=0.7)
        contents = [r["content"] for r in filtered]
        # zombie record (importance=0.65) must be absent
        assert all("zombie" not in c for c in contents), f"expected zombie filtered, got {contents}"
        # high-conf record present
        assert any("alpha record" == c for c in contents), f"expected high-conf present, got {contents}"

    async def test_recall_per_call_historical_only_suppresses(self, client: MemoryClient) -> None:
        await client.store("beta current record", importance=0.8)
        await client.store(
            "beta historical record",
            importance=0.8,
            metadata={"currentness_status": "historical_only"},
        )
        filtered = await client.recall("beta", exclude_historical_only=True)
        contents = [r["content"] for r in filtered]
        assert all("historical" not in c for c in contents), f"expected historical filtered, got {contents}"
        assert any("current" in c for c in contents), f"expected current present, got {contents}"


class TestAdmissionFilterSharedImplementation:
    """The recall-policy seam must resolve to a SINGLE shared Implementation —
    both recall Interfaces (SDK + MCP tool) and the back-compat re-export point
    at the same object so the policy cannot drift between them.
    """

    def test_client_helper_reexports_canonical(self) -> None:
        from trw_memory._client_recall_helpers import apply_admission_filter as client_side
        from trw_memory.retrieval.admission_policy import apply_admission_filter as canonical

        assert client_side is canonical

    def test_tool_path_uses_canonical(self) -> None:
        from trw_memory.retrieval.admission_policy import apply_admission_filter as canonical
        from trw_memory.tools import recall as recall_module

        assert recall_module.apply_admission_filter is canonical

    def test_canonical_accepts_plain_dicts(self) -> None:
        # The tool path passes plain dict[str, object] rows, not MemoryResultDict.
        from trw_memory.retrieval.admission_policy import apply_admission_filter

        rows: list[dict[str, object]] = [
            {"id": "keep", "metadata": {"confidence": 0.9}},
            {"id": "drop", "metadata": {"confidence": 0.4}},
            {"id": "fallback", "importance": 0.95},  # no metadata.confidence -> importance fallback
        ]
        out = apply_admission_filter(rows, confidence_floor=0.7, exclude_historical_only=False)
        assert [r["id"] for r in out] == ["keep", "fallback"]


@pytest.mark.parametrize(
    "confidence_floor,exclude_historical,expected_ids",
    [
        (None, False, ["a", "b", "c"]),  # default off
        (0.7, False, ["a", "c"]),  # 'b' at 0.65 dropped
        (None, True, ["a", "b"]),  # 'c' historical dropped
        (0.7, True, ["a"]),  # 'b' + 'c' both dropped
    ],
)
def test_filter_matrix(
    confidence_floor: float | None,
    exclude_historical: bool,
    expected_ids: list[str],
) -> None:
    results = [
        _r("a", confidence=0.9, currentness="current"),
        _r("b", confidence=0.65, currentness="current"),
        _r("c", confidence=0.9, currentness="historical_only"),
    ]
    out = apply_admission_filter(
        results,
        confidence_floor=confidence_floor,
        exclude_historical_only=exclude_historical,
    )
    assert [r["memory_id"] for r in out] == expected_ids
