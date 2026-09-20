"""Tests for watch.py. Run: pytest trw-memory/benchmarks/locomo/test_watch.py (offline)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import watch


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(watch, "ROOT", tmp_path)
    monkeypatch.setattr(watch, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(watch, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(watch.batch_judge, "key_usage", lambda: 1.0)
    monkeypatch.setattr(watch.os, "getloadavg", lambda: (1.0, 1.0, 1.0))
    (tmp_path / "bench" / "results" / "locomo").mkdir(parents=True)
    return tmp_path


def _once(root: Path, *extra: str) -> int:
    return watch.main(["--phase", "P", "--budget", "5", "--bench-dir", str(root / "bench"), "--once", *extra])


def test_tail_leaves_a_torn_line_for_later(tmp_path: Path) -> None:
    p = tmp_path / "f.jsonl"
    p.write_text('{"a": 1}\n{"b": ')
    t = watch.Tail(p, from_end=False)
    assert t.lines() == ['{"a": 1}']
    p.write_text('{"a": 1}\n{"b": 2}\n')
    assert t.lines() == ['{"b": 2}']


def test_quiet_run_is_ok_and_leaves_no_stop(root: Path) -> None:
    assert _once(root) == 0 and not (root / "STOP").exists()


def test_ingest_writes_count_as_progress(root: Path) -> None:
    state = root / "state" / "mem0-v6-c0"
    state.mkdir(parents=True)
    (state / "add_log.jsonl").write_text(json.dumps({"ts": time.time(), "results": 2, "error": None}) + "\n")
    # no prediction yet, and a 0-second stall limit: the fresh write must still count as progress
    assert _once(root, "--pred", "predicted_x", "--stall-stop", "0.5") == 0


def test_failing_mem0_writes_stop_the_phase(root: Path) -> None:
    state = root / "state" / "a"
    state.mkdir(parents=True)
    later = time.time() + 60  # written after the watcher started (earlier rows are ignored)
    rows = [{"ts": later, "results": None, "error": "RateLimitError"}] * 3 + [{"ts": later, "results": 1}] * 10
    rows += [{"ts": time.time() - 3600, "results": None, "error": "an old run"}] * 50  # must not count
    (state / "add_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "P.pids").write_text("")
    assert _once(root) == 3
    assert "mem0 writes failing" in (root / "STOP").read_text()


def test_one_bad_early_prediction_does_not_stop(root: Path) -> None:
    pred = root / "bench" / "results" / "locomo" / "predicted_x"
    pred.mkdir()
    (pred / "conv0_q0.json").write_text(json.dumps({"retrieval": {"search_results": []}}))
    assert _once(root, "--pred", "predicted_x") == 0


def test_ledger_gates_only_this_phase(root: Path) -> None:
    other = {"phase": "Q", "event": "summary", "questions": 100, "judge_error": 50}
    (root / "ledger.jsonl").write_text(json.dumps(other) + "\n")
    assert _once(root) == 0  # a live watcher starts at the ledger's end
    assert _once(root, "--from-start") == 0  # and another phase's rows are ignored
    short = {"phase": "P", "event": "batch_done", "batch_id": "b", "status": "completed", "n": 100,
             "returned": 90, "errors": 0, "truncated": 0, "usage": {}}  # fmt: skip
    with (root / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps(short) + "\n")
    assert _once(root, "--from-start") == 3
    assert "90/100 back" in (root / "STOP").read_text()


def test_missing_questions_warn_but_errors_stop(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ok = {"phase": "P", "event": "summary", "questions": 100, "judged": 98, "missing": 2, "judge_error": 1}
    (root / "ledger.jsonl").write_text(json.dumps(ok) + "\n")
    assert _once(root, "--from-start") == 0 and "rerun the same command" in capsys.readouterr().out
    (root / "ledger.jsonl").write_text(json.dumps({**ok, "judge_error": 3}) + "\n")
    assert _once(root, "--from-start") == 3
