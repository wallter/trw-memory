"""PRD-CORE-195 FR06 — HyPE benchmark runs two arms, no LLM/network."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("sqlite_vec")
pytest.importorskip("sentence_transformers")

from benchmarks.bench_hype import GoldenQuestionGenerator, HypeBenchmark
from tests.conftest import make_entry
from trw_memory.exceptions import (
    LocalOnlyViolationError,
    MemoryError,
    PIIBlockError,
    PoisoningError,
    SchemaValidationError,
    StorageError,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [LocalOnlyViolationError, StorageError, MemoryError, RuntimeError])
async def test_arm_propagates_infrastructure_errors_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    from benchmarks import bench_hype

    error = error_type("benchmark infrastructure unavailable")
    client = SimpleNamespace(
        _config=SimpleNamespace(),
        store=AsyncMock(side_effect=error),
        recall=AsyncMock(),
        close=AsyncMock(),
    )
    monkeypatch.setattr(bench_hype, "MemoryClient", lambda *args, **kwargs: client)
    entries = [{"id": "g1", "content": "fixture", "tags": [], "importance": 0.5, "queries": []}]

    with pytest.raises(error_type) as raised:
        await bench_hype._run_arm(entries, {}, tmp_path / "arm.db", hype_enabled=False)

    assert raised.value is error
    client.store.assert_awaited_once()
    client.recall.assert_not_awaited()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [SchemaValidationError, PoisoningError, PIIBlockError])
async def test_arm_excludes_only_explicit_content_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    from benchmarks import bench_hype

    client = SimpleNamespace(
        _config=SimpleNamespace(),
        store=AsyncMock(side_effect=[error_type("fixture content rejected"), {"memory_id": "accepted"}]),
        recall=AsyncMock(return_value=[{"memory_id": "accepted"}]),
        close=AsyncMock(),
    )
    monkeypatch.setattr(bench_hype, "MemoryClient", lambda *args, **kwargs: client)
    entries = [
        {
            "id": entry_id,
            "content": "fixture",
            "tags": [],
            "importance": 0.5,
            "queries": [{"query": entry_id, "relevant": True}],
        }
        for entry_id in ("rejected", "accepted")
    ]

    result = await bench_hype._run_arm(entries, {}, tmp_path / "arm.db", hype_enabled=True)

    assert result == {"recall_at_10": 1.0, "ndcg_at_10": 1.0, "queries": 1.0}
    assert client.store.await_count == 2
    client.recall.assert_awaited_once_with("accepted", limit=10)
    client.close.assert_awaited_once()


def test_golden_generator_returns_mapped_questions() -> None:
    gen = GoldenQuestionGenerator({"g1": ["q one", "q two"]})
    assert gen.generate(make_entry(entry_id="g1")) == ["q one", "q two"]
    assert gen.generate(make_entry(entry_id="missing")) == []


def test_two_arm_delta_deterministic_fake_embedder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR06 two-arm structure + HyPE-on >= off, with a deterministic embedder.

    Drives the real benchmark ``_run_arm`` path (store → recall → metrics) but
    patches the embedder so the test runs in <1s and is GPU-independent. A tiny
    synthetic fixture where the query's only dense match is the HyPE sibling
    proves the on-arm strictly beats the off-arm.
    """
    import json

    from benchmarks import bench_hype
    from tests.test_hype_recall import _LabelEmbedder

    # Isolate the relative ``.memory`` tier sidecars to a per-test dir. Without
    # this the benchmark's MemoryClient writes warm/cold sidecars under a
    # ``.memory`` dir resolved relative to the process cwd, which accumulates
    # entries across runs and across the bundled-fixture benchmark. That shared
    # pollution non-deterministically changes which docs fill the off-arm
    # top-10 (it MASKED this assertion in the monorepo by pushing the target out
    # of the off-arm window; a clean store surfaces it and the strict-uplift
    # assertion flips). Pinning a hermetic store makes both arms deterministic.
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))

    # Synthetic golden set: content uses label @doc; the positive query uses
    # @ask. With HyPE off the query cannot reach the entry via dense; with HyPE
    # on the sibling (which carries @ask) collapses to the parent.
    fixture = tmp_path / "mini.json"
    # The target content is dense-orthogonal to the query (@doc vs @ask). To keep
    # the off-arm from trivially returning the target as a low-score fallback, the
    # corpus must be LARGER than the recall limit (10) AND the distractors must
    # outrank the target on BM25 — so each distractor shares the query token
    # "request" (which the target does not) and there are 15 of them. On the off
    # arm the BM25-positive distractors fill the top-10 and the orthogonal target
    # is excluded (recall 0). With HyPE on, the target's @ask sibling (kept because
    # the query is >= hype_min_question_chars) provides the only dense hit that
    # collapses back to the target → strict uplift.
    distractors = [
        {
            "id": f"distractor-{i}",
            "content": f"paraphrased request filler entry number {i} zzz",
            "tags": ["t"],
            "importance": 0.3,
            "queries": [],
        }
        for i in range(15)
    ]
    target = {
        "id": "default-mini-1",
        "content": "statement @doc body",
        "tags": ["t"],
        "importance": 0.5,
        "queries": [{"query": "paraphrased request @ask here", "relevant": True}],
    }
    fixture.write_text(json.dumps({"entries": [target, *distractors]}), encoding="utf-8")
    shared = _LabelEmbedder()
    monkeypatch.setattr(
        "trw_memory.client.MemoryClient._get_embedder",
        lambda self: shared,
    )

    report = bench_hype.HypeBenchmark(fixture).run(tmp_path / "dbs")
    assert set(report) == {"off", "on", "delta"}
    # On-arm surfaces the target via its @ask sibling; off-arm cannot reach it
    # densely (orthogonal) → strict uplift.
    assert report["on"]["recall_at_10"] == 1.0
    assert report["on"]["recall_at_10"] > report["off"]["recall_at_10"]
    assert report["delta"]["recall_at_10"] > 0.0


@pytest.mark.slow
def test_two_arm_delta_reports_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercises the REAL store+recall path on the bundled fixture for both arms.
    # Keep tier sidecars inside the test sandbox; otherwise MemoryClient defaults
    # to a shared cwd-relative .memory directory and amplifies benchmark I/O
    # across repeated/local/xdist release runs.
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
    report = HypeBenchmark().run(tmp_path / "dbs")
    assert set(report) == {"off", "on", "delta"}
    for arm in ("off", "on"):
        assert 0.0 <= report[arm]["recall_at_10"] <= 1.0
        assert 0.0 <= report[arm]["ndcg_at_10"] <= 1.0
        assert report[arm]["queries"] > 0
    # The on-arm uses the golden queries themselves as hypothetical questions,
    # so HyPE recall must not be WORSE than the off arm.
    assert report["delta"]["recall_at_10"] >= 0.0
