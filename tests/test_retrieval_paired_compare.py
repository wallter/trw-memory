"""benchmarks/retrieval_paired_compare.py: the durable paired-significance tool
(PRD-CORE-284 acceptance, PRD-CORE-283 FR05)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks import retrieval_paired_compare as rpc

pytest.importorskip("scipy")
pytest.importorskip("numpy")


def _run(path: Path, label: str, hits: list[int], *, returned: int, query_s: float) -> Path:
    rows = [
        {
            "conv": i // 4,
            "q": i % 4,
            "category": 1 if i % 2 else 4,
            "hit@10": float(h),
            "recall@10": float(h) * 0.5,
            "mrr": float(h) / 2,
            "n_returned": returned + i % 3,
            "query_s": query_s + i * 0.001,
        }
        for i, h in enumerate(hits)
    ]
    path.write_text(json.dumps({"label": label, "k": [10], "per_question": rows}))
    return path


def test_mcnemar_exact_values() -> None:
    assert rpc.mcnemar(0, 0) == 1.0
    assert rpc.mcnemar(2, 11) == pytest.approx(0.0225, abs=5e-5)  # the LOCOMO bridge-hop result
    assert rpc.mcnemar(11, 2) == rpc.mcnemar(2, 11)
    # one-sided regression p: small only when B loses more than it gains
    assert rpc.mcnemar_regression(11, 2) == pytest.approx(sum(math.comb(13, k) for k in range(11, 14)) / 2**13)
    assert rpc.mcnemar_regression(2, 11) > 0.99


def test_tost_accepts_tiny_differences_and_rejects_large_ones() -> None:
    tiny = [0.001 * (1 if i % 2 else -1) for i in range(60)]
    assert rpc.tost(tiny, 0.05)["p"] < 0.05
    assert rpc.tost(tiny, 0.05, test="t")["p"] < 0.05
    large = [0.2 + 0.001 * i for i in range(60)]
    assert rpc.tost(large, 0.05)["p"] > 0.05
    with pytest.raises(ValueError):
        rpc.tost(tiny, 0.0)


def test_distribution_uses_nearest_rank_p90() -> None:
    assert rpc.distribution([float(v) for v in range(1, 11)]) == {"min": 1, "median": 5.5, "p90": 9, "max": 10}


def test_cli_pairs_questions_and_reports_every_section(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = _run(tmp_path / "a.json", "before", [1] * 30 + [0] * 10, returned=5, query_s=0.2)
    after = _run(tmp_path / "b.json", "after", [1] * 38 + [0] * 2, returned=20, query_s=0.25)
    out_json = tmp_path / "summary.json"
    flags = ["--latency", "--rows", "--tost", "recall@10", "--margin", "0.5", "--bootstrap", "500"]
    code = rpc.main([str(before), str(after), *flags, "--json", str(out_json)])
    assert code == 0
    text = capsys.readouterr().out
    assert "multi-hop" in text and "single-hop" in text
    assert "hit@10: A-only=0 B-only=8" in text
    assert "latency n=40" in text and "n_returned before: min=5" in text
    summary = json.loads(out_json.read_text())
    assert summary["n"] == 40 and summary["metrics"] == ["hit@10", "recall@10", "mrr"]
    assert summary["mcnemar"]["hit@10"]["p_two_sided"] == pytest.approx(rpc.mcnemar(0, 8))
    assert summary["n_returned"]["b"]["median"] > summary["n_returned"]["a"]["median"]
    lo, hi = summary["latency"]["median_delta_ci"]
    assert lo <= summary["latency"]["median_delta"] <= hi
    # deterministic bootstrap for a fixed seed
    again = rpc.compare(rpc.load(str(before))[1], rpc.load(str(after))[1], ["hit@10"], n_boot=500)
    assert again["latency"]["median_delta_ci"] == pytest.approx(tuple(summary["latency"]["median_delta_ci"]))


def test_cli_refuses_runs_with_no_shared_questions(tmp_path: Path) -> None:
    a = _run(tmp_path / "a.json", "a", [1, 0], returned=5, query_s=0.1)
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"label": "b", "k": [10], "per_question": [{"conv": 99, "q": 0, "hit@10": 1.0}]}))
    assert rpc.main([str(a), str(b)]) == 2


def test_complete_at_k_is_paired_by_mcnemar_and_in_the_default_metrics(tmp_path: Path) -> None:
    # complete@k is 0/1 per question like hit@k, so it is an exact McNemar, not a Wilcoxon.
    def run(path: Path, complete: list[int]) -> Path:
        rows = [
            {"conv": 0, "q": i, "category": 1, "hit@10": 1.0, "complete@10": float(c)} for i, c in enumerate(complete)
        ]
        path.write_text(json.dumps({"label": path.stem, "k": [10], "per_question": rows}))
        return path

    payload, a = rpc.load(str(run(tmp_path / "a.json", [1] * 20 + [0] * 10)))
    _, b = rpc.load(str(run(tmp_path / "b.json", [1] * 27 + [0] * 3)))
    assert rpc.default_metrics(payload, a) == ["hit@10", "complete@10"]
    result = rpc.compare(a, b, ["complete@10"], n_boot=100)
    assert result["mcnemar"]["complete@10"]["b_only"] == 7
    assert result["table"]["ALL"]["complete@10"]["p"] == pytest.approx(rpc.mcnemar(0, 7))
