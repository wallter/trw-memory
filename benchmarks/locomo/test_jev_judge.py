"""Tests for jev_judge.py. Run explicitly: pytest trw-memory/benchmarks/locomo/test_jev_judge.py

No test touches the network: the judge is a stand-in that answers from a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import jev_judge as jj

from trw_memory.decisions import DecisionFailure, DecisionResult, NoulAnswer


class _Judge:
    """Answers ``correct`` with the probability keyed by the generated answer; fails for "boom"."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def decide(self, state: dict[str, Any], questions: dict[str, Any], *, timeout_s: float) -> object:
        if state["generated_answer"] == "boom":
            return DecisionFailure(kind="provider_error", detail="HTTP 502")
        p = {"yes": 0.9, "no": 0.1}[state["generated_answer"]]
        return DecisionResult(model="jev-test", answers={"correct": NoulAnswer(noul=p)}, backend="t", latency_ms=1.0)


def _item(root: Path, qid: str, answer: str, judgment: str) -> None:
    doc = {
        "question_id": qid,
        "question": "q?",
        "ground_truth_answer": "a",
        "category": 1,
        "category_name": "multi-hop",
        "cutoff_results": {"top_10": {"generated_answer": answer, "judgment": judgment}},
    }
    (root / f"{qid}.json").write_text(json.dumps(doc))


def test_a_failed_call_is_reported_unscored_not_dropped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = SimpleNamespace(preprocess_answer=lambda _cat, gold: gold)
    monkeypatch.setattr(jj.bj, "load_harness", lambda _bench: (None, prompts))
    monkeypatch.setattr(jj.bj, "api_key", lambda: "k")
    monkeypatch.setattr(jj, "JevHttpJudge", _Judge)
    _item(tmp_path, "conv0_q0", "yes", "CORRECT")
    _item(tmp_path, "conv0_q1", "no", "WRONG")
    _item(tmp_path, "conv0_q2", "boom", "CORRECT")

    assert jj.main(["--judged", str(tmp_path), "--workers", "1"]) == 0

    out = json.loads((tmp_path / "jev.json").read_text())
    rows = out["rows"]
    assert rows["conv0_q0"]["p"] == pytest.approx(0.9)
    assert rows["conv0_q1"]["p"] == pytest.approx(0.1)
    assert rows["conv0_q2"]["p"] is None  # a DecisionFailure carries no answers
    assert out["provenance"]["models_reported"] == ["jev-test"]
