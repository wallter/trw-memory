"""Tests for compare.py. Run: pytest trw-memory/benchmarks/locomo/test_compare.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import compare


def _write(d: Path, verdicts: dict[str, str], cut: str = "top_10", gold: str = "gold") -> None:
    """Records shaped like the stock runner's predict output plus a judged cutoff."""
    d.mkdir(parents=True, exist_ok=True)
    for qid, v in verdicts.items():
        rec = {"question_id": qid, "question": f"q {qid}", "ground_truth_answer": gold, "category": 4,
               "category_name": "single-hop",
               "cutoff_results": {cut: {"judgment": v, "memories_evaluated": 10, "generated_answer": "x"}}}  # fmt: skip
        (d / f"{qid}.json").write_text(json.dumps(rec))


def _ids(n_conv: int, per: int) -> list[str]:
    return [f"conv{c}_q{i}" for c in range(n_conv) for i in range(per)]


def _run(tmp_path: Path, a: str, b: str, *extra: str) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "r.json"
    code = compare.main([str(tmp_path / a), str(tmp_path / b), "--cutoff", "top_10", "--reps", "300",
                         "--json", str(out), *extra])  # fmt: skip
    return code, (json.loads(out.read_text()) if out.exists() else {})


def test_known_counts_mcnemar_and_decision(tmp_path: Path) -> None:
    ids = _ids(2, 10)
    _write(tmp_path / "a", {q: "CORRECT" if i < 12 else "WRONG" for i, q in enumerate(ids)})
    _write(tmp_path / "b", {q: "CORRECT" if i < 16 else "WRONG" for i, q in enumerate(ids)})
    code, rep = _run(tmp_path, "a", "b", "--expected", "20")
    assert code == 0
    row = rep["rows"]["ALL"]
    assert (row["a"], row["b"], row["a_only"], row["b_only"]) == (12, 16, 0, 4)
    assert row["mcnemar_p"] == pytest.approx(0.125)
    assert rep["diff"] == pytest.approx(0.2)
    assert rep["decision"] == "INCONCLUSIVE"  # p = 0.125 cannot support a difference


def test_clear_difference_needs_both_tests(tmp_path: Path) -> None:
    ids = _ids(10, 30)
    _write(tmp_path / "a", {q: "CORRECT" if i % 30 < 15 else "WRONG" for i, q in enumerate(ids)})
    _write(tmp_path / "b", {q: "CORRECT" if i % 30 < 25 else "WRONG" for i, q in enumerate(ids)})
    code, rep = _run(tmp_path, "a", "b")
    assert code == 0 and rep["decision"].startswith("DIFFERENCE") and rep["cluster95"][0] > 0


def test_refuses_mismatched_incomplete_or_malformed_runs(tmp_path: Path) -> None:
    ids = _ids(1, 4)
    _write(tmp_path / "a", dict.fromkeys(ids, "CORRECT"))
    _write(tmp_path / "short", dict.fromkeys(ids[:3], "CORRECT"))
    _write(tmp_path / "othergold", dict.fromkeys(ids, "CORRECT"), gold="other")
    _write(tmp_path / "maybe", {**dict.fromkeys(ids, "CORRECT"), ids[0]: "MAYBE"})
    _write(tmp_path / "top50", dict.fromkeys(ids, "CORRECT"), cut="top_50")
    for other in ("short", "othergold", "maybe", "top50"):
        assert _run(tmp_path, "a", other)[0] == 2, other
    assert _run(tmp_path, "a", "a", "--expected", "5")[0] == 2


def test_judge_errors_score_wrong_are_counted_and_all_error_sensitivity_is_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = _ids(1, 4)
    _write(tmp_path / "a", dict.fromkeys(ids, "CORRECT"))
    _write(tmp_path / "b", {**dict.fromkeys(ids, "CORRECT"), ids[0]: "ERROR"})
    code, rep = _run(tmp_path, "a", "b")
    assert code == 0 and rep["judge_errors"] == {"A": 1 - 1, "B": 1} and rep["rows"]["ALL"]["b"] == 3
    assert rep["sensitivity_n"] == 3 and rep["sensitivity_diff"] == 0.0
    _write(tmp_path / "allerr", dict.fromkeys(ids, "ERROR"))
    code, rep = _run(tmp_path, "a", "allerr")
    assert code == 0 and rep["sensitivity_n"] == 0 and rep["sensitivity_diff"] is None


def test_identical_runs_are_equivalent(tmp_path: Path) -> None:
    ids = _ids(10, 20)
    _write(tmp_path / "a", {q: "CORRECT" if i % 3 else "WRONG" for i, q in enumerate(ids)})
    code, rep = _run(tmp_path, "a", "a")
    assert code == 0 and rep["decision"].startswith("EQUIVALENT")
