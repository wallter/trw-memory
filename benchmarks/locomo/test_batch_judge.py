"""Tests for batch_judge.py. Run explicitly: pytest trw-memory/benchmarks/locomo/test_batch_judge.py

The end-to-end tests need the pinned harness (bootstrap.sh) at $TRW_BENCH_DIR or
~/.cache/trw-bench/memory-benchmarks and are skipped without it. No test touches the network.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import batch_judge as bj

BENCH = Path(os.getenv("TRW_BENCH_DIR", Path.home() / ".cache/trw-bench/memory-benchmarks"))
needs_harness = pytest.mark.skipif(
    not (BENCH / "benchmarks/locomo/prompts.py").exists(), reason="harness not bootstrapped"
)


def test_parse_answer_follows_the_stock_rule() -> None:
    assert bj.parse_answer("step 1...\nANSWER: 7 May 2023") == ("7 May 2023", True)
    assert bj.parse_answer("a ANSWER: x ANSWER: y") == ("y", True)
    assert bj.parse_answer("  no marker  ") == ("no marker", False)
    assert bj.parse_answer(None) == ("", False)


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ('{"label": "CORRECT", "reasoning": "r"}', "CORRECT"),
        ('{"label": "correct"}', "CORRECT"),
        ('{"label": "WRONG"}', "WRONG"),
        ('{"final": {"label": "CORRECT"}}', "CORRECT"),
        ('{"final": "{\\"label\\": \\"WRONG\\"}"}', "WRONG"),
        ("not json", "ERROR"),
        ('{"verdict": "CORRECT"}', "ERROR"),
        ("", "ERROR"),
        (None, "ERROR"),
    ],
)
def test_parse_judge_flags_unparseable_replies_as_error(raw: str | None, label: str) -> None:
    assert bj.parse_judge(raw)[0] == label


def test_chat_body_temperature_zero_unless_reasoning() -> None:
    plain = bj.chat_body("m", "sys", "u", json_mode=True, reasoning=None)
    assert plain["temperature"] == 0 and plain["response_format"] == {"type": "json_object"}
    assert plain["messages"][0] == {"role": "system", "content": "sys"}
    reasoning = bj.chat_body("m", "", "u", json_mode=False, reasoning="low")
    assert "temperature" not in reasoning and reasoning["reasoning"] == {"effort": "low"}
    assert reasoning["messages"] == [{"role": "user", "content": "u"}]
    assert plain["max_tokens"] == reasoning["max_tokens"] == 4096


def test_result_rows_normalises_success_and_error() -> None:
    batch = {
        "results": [
            {
                "custom_id": "a",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": 3}},
                },
            },
            {"custom_id": "b", "error": {"message": "boom"}},
            {"custom_id": "c", "response": {"status_code": 400, "body": {"error": "bad"}}},
        ]
    }
    rows = {r["custom_id"]: r for r in bj.result_rows(batch)}
    assert rows["a"]["content"] == "hi" and rows["a"]["usage"] == {"total_tokens": 3}
    assert rows["b"]["content"] is None and rows["b"]["error"]
    assert rows["c"]["content"] is None and rows["c"]["error"]["status_code"] == 400


def test_state_keeps_a_success_over_a_later_error(tmp_path: Path) -> None:
    st = bj.State(tmp_path)
    st.add_results([{"custom_id": "x", "content": "ok", "usage": None, "error": None}])
    st.add_results([{"custom_id": "x", "content": None, "usage": None, "error": "late"}])
    st.add_results([{"custom_id": "y", "content": None, "usage": None, "error": "e"}])
    got = st.results()
    assert got["x"]["content"] == "ok" and got["y"]["content"] is None


def test_poll_records_terminal_results_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bj, "LEDGER", tmp_path / "ledger.jsonl")
    st = bj.State(tmp_path / "s")
    st.save_batches([{"id": "b1", "role": "answer", "n": 1, "status": "in_progress", "submitted": 0}])
    calls = iter(
        [
            {"status": "in_progress"},
            {
                "status": "completed",
                "usage": {"cost": 0.01},
                "results": [
                    {
                        "custom_id": "q:answer",
                        "response": {"body": {"choices": [{"message": {"content": "ANSWER: z"}}]}},
                    }
                ],
            },
        ]
    )
    bj.poll_batches(st, "answer", "t", 0, lambda _bid: next(calls))
    assert st.results()["q:answer"]["content"] == "ANSWER: z"
    assert st.batches()[0]["status"] == "completed"
    bj.poll_batches(st, "answer", "t", 0, lambda _bid: pytest.fail("terminal batch re-polled"))


def _fake_bench(tmp_path: Path, n: int) -> tuple[Path, list[str]]:
    bench = tmp_path / "bench"
    (bench / "datasets/locomo").mkdir(parents=True)
    (bench / "benchmarks").symlink_to(BENCH / "benchmarks")
    (bench / "datasets/locomo/locomo10.json").symlink_to(BENCH / "datasets/locomo/locomo10.json")
    run, prompts = bj.load_harness(bench)
    data = run.load_dataset(str(bench / "datasets/locomo/locomo10.json"))
    items = run.expected_locomo_question_items(data, [0], prompts.CATEGORIES_TO_EVALUATE, n)
    pred = bench / "results/locomo/predicted_fake"
    pred.mkdir(parents=True)
    for qid, _, _, qa in items:
        (pred / f"{qid}.json").write_text(
            json.dumps(
                {
                    "question": qa["question"],
                    "category_name": "x",
                    "retrieval": {
                        "search_results": [
                            {"memory": f"fact-{i:02d} of {qid}", "created_at": f"2023-05-{i + 1:02d}T00:00:00"}
                            for i in range(12)
                        ]
                    },
                }
            )
        )
    return bench, [i[0] for i in items]


@needs_harness
def test_dry_run_builds_stock_prompts_with_the_cutoff(tmp_path: Path) -> None:
    bench, _ = _fake_bench(tmp_path, 3)
    assert (
        bj.main(
            [
                "--bench-dir",
                str(bench),
                "--pred",
                "predicted_fake",
                "--tag",
                "t",
                "--model",
                "m",
                "--conversations",
                "0",
                "--max-questions",
                "3",
                "--dry-run",
            ]
        )
        == 0
    )
    lines = (bench / "results/locomo/predicted_fake__t/_batch/top_10/dryrun_answer.jsonl").read_text().splitlines()
    assert len(lines) == 3
    _, prompts = bj.load_harness(bench)
    first = json.loads(lines[0])
    pred = json.loads((bench / f"results/locomo/predicted_fake/{first['custom_id'].split(':')[0]}.json").read_text())
    expected = prompts.get_answer_generation_prompt(
        pred["question"], pred["retrieval"]["search_results"][:10], reference_date=None, user_profile=None
    )
    assert first["body"]["messages"] == [{"role": "user", "content": expected}]
    assert "fact-09" in expected and "fact-10" not in expected


@pytest.fixture
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No network, no shared ledger or STOP file."""
    monkeypatch.setattr(bj, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(bj, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(bj, "key_usage", lambda: 1.0)
    monkeypatch.setattr(bj, "key_info", lambda: {"usage": 1.0, "limit_remaining": 40.0})
    return tmp_path


def _args(bench: Path, n: int, *extra: str) -> list[str]:
    return ["--bench-dir", str(bench), "--pred", "predicted_fake", "--tag", "t", "--model", "m", "--price",
            "0.1,0.1", "--conversations", "0", "--max-questions", str(n), "--mode", "sync", "--workers", "1", *extra]  # fmt: skip


@needs_harness
def test_round_trip_retries_bad_judge_json_and_merges_cutoffs(offline: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, qids = _fake_bench(offline, 4)
    judged: list[str] = []

    def fake_http(method: str, path: str, body: dict[str, Any] | None = None, *_: Any, **__: Any) -> dict[str, Any]:
        user = body["messages"][-1]["content"]
        if body.get("response_format"):
            judged.append(user)
            content = "garbage" if len(judged) == 1 else '```json\n{"label": "CORRECT", "reasoning": "ok"}\n```'
        else:
            content = "thinking\nANSWER: 42"
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    monkeypatch.setattr(bj, "http", fake_http)
    assert bj.main(_args(bench, 4)) == 0
    out = bench / "results/locomo/predicted_fake__t"
    results = [json.loads((out / f"{q}.json").read_text())["cutoff_results"]["top_10"] for q in qids]
    assert {r["generated_answer"] for r in results} == {"42"}
    assert [r["judgment"] for r in results] == ["CORRECT"] * 4  # the garbage reply was retried
    assert len(judged) == 5 and all("42" in u for u in judged)
    summary = [json.loads(line) for line in (offline / "ledger.jsonl").read_text().splitlines()][-1]
    assert summary["judge_error"] == 0 and summary["correct"] == 4 and summary["missing"] == 0
    judged.clear()
    assert bj.main(_args(bench, 4)) == 0 and judged == []  # nothing left to buy
    assert bj.main(_args(bench, 4, "--cutoff", "50")) == 0  # a second cutoff keeps the first
    merged = json.loads((out / f"{qids[0]}.json").read_text())["cutoff_results"]
    assert set(merged) == {"top_10", "top_50"}


@needs_harness
def test_failed_answer_is_retried_not_judged(offline: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, _ = _fake_bench(offline, 3)
    fail_once = {"n": 0}
    judged: list[str] = []

    def fake_http(method: str, path: str, body: dict[str, Any] | None = None, *_: Any, **__: Any) -> dict[str, Any]:
        if body.get("response_format"):
            judged.append(body["messages"][-1]["content"])
            return {"choices": [{"message": {"content": '{"label": "CORRECT"}'}}]}
        if fail_once["n"] == 0:
            fail_once["n"] += 1
            raise SystemExit("POST /chat/completions -> HTTP 502")
        return {"choices": [{"message": {"content": "ANSWER: ok"}}]}

    monkeypatch.setattr(bj, "http", fake_http)
    with pytest.raises(SystemExit, match="answer quality gate"):
        bj.main(_args(bench, 3))  # 1 of 3 missing is over the 1% gate: nothing judged yet
    assert judged == []
    assert bj.main(_args(bench, 3)) == 0  # the rerun answers only the missing one, then judges all
    assert len(judged) == 3


@needs_harness
def test_configuration_change_budget_and_truncation_are_refused(offline: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, _ = _fake_bench(offline, 2)
    calls: list[str] = []

    def fake_http(method: str, path: str, body: dict[str, Any] | None = None, *_: Any, **__: Any) -> dict[str, Any]:
        calls.append(path)
        return {"choices": [{"message": {"content": "ANSWER: x"}, "finish_reason": "length"}]}

    monkeypatch.setattr(bj, "http", fake_http)
    with pytest.raises(SystemExit, match="exceeds --budget"):
        bj.main(_args(bench, 2, "--budget", "0.0000001"))
    assert calls == []
    with pytest.raises(SystemExit, match="answer quality gate"):
        bj.main(_args(bench, 2))  # every answer truncated
    with pytest.raises(SystemExit, match="different configuration"):
        bj.main([*_args(bench, 2), "--reasoning", "low"])


@needs_harness
def test_strict_rescore_reuses_answers_and_keeps_the_answerer(offline: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, qids = _fake_bench(offline, 2)
    bodies: list[dict[str, Any]] = []

    def fake_http(method: str, path: str, body: dict[str, Any] | None = None, *_: Any, **__: Any) -> dict[str, Any]:
        bodies.append(body)
        content = '{"label": "WRONG"}' if body.get("response_format") else "ANSWER: 1"
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    monkeypatch.setattr(bj, "http", fake_http)
    assert bj.main(_args(bench, 2, "--reasoning", "low")) == 0
    answers = [b for b in bodies if not b.get("response_format")]
    assert answers and all(b["reasoning"] == {"effort": "low"} and "temperature" not in b for b in answers)
    bodies.clear()
    strict = [a if a != "t" else "t2" for a in _args(bench, 2)]
    assert bj.main([*strict, "--judge-model", "j", "--rubric", "strict", "--judge-only-from", "predicted_fake__t"]) == 0
    assert bodies and all(b["model"] == "j" and b["temperature"] == 0 for b in bodies)  # judge calls only
    assert all("Gold answer:" in b["messages"][-1]["content"] for b in bodies)
    out = json.loads((bench / f"results/locomo/predicted_fake__t2__strict/{qids[0]}.json").read_text())
    assert out["cutoff_results"]["top_10"]["answerer_model"] == "m"
    with pytest.raises(SystemExit):
        bj.main([*strict, "--rubric", "strict"])  # strict without fixed answers is refused


def test_uncertain_submission_is_recorded_not_retried(offline: Path) -> None:
    st = bj.State(offline / "s")

    def boom(_payload: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("read timed out")

    with pytest.raises(SystemExit, match="--adopt"):
        bj.submit_batches(st, "answer", "m", {"a": {}, "b": {}}, 500, "t", boom)
    assert [r["status"] for r in st.batches()] == ["submitting"]
    with pytest.raises(SystemExit, match="unreconciled"):
        bj.poll_batches(st, "answer", "t", 0, lambda _bid: {})


@needs_harness
def test_judge_retry_semantics_match_the_stock_client(offline: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, qids = _fake_bench(offline, 2)
    pred = bench / "results/locomo/predicted_fake"
    by_question = {json.loads((pred / f"{q}.json").read_text())["question"]: q for q in qids}
    replies = {qids[0]: iter(['{"label": "MAYBE"}']), qids[1]: iter([None, '{"label": "CORRECT"}'])}
    judged: list[str] = []

    def fake_http(method: str, path: str, body: dict[str, Any] | None = None, *_: Any, **__: Any) -> dict[str, Any]:
        if not body.get("response_format"):
            return {"choices": [{"message": {"content": "ANSWER: a"}}]}
        prompt = body["messages"][-1]["content"]
        qid = next(q for text, q in by_question.items() if text in prompt)
        judged.append(qid)
        reply = next(replies[qid])
        if reply is None:
            raise SystemExit("read timed out")  # outcome unknown: may have been billed
        return {"choices": [{"message": {"content": reply}}]}

    monkeypatch.setattr(bj, "http", fake_http)
    # exit 2: the bad-label ERROR is 1 of 1 judged (> 1%); the failed call is left missing, not re-sent
    assert bj.main(_args(bench, 2)) == 2
    assert judged.count(qids[0]) == 1  # a parsed reply with a bad label is final (stock scores it WRONG)
    assert judged.count(qids[1]) == 1
    assert bj.main(_args(bench, 2)) == 2  # the next run retries only the unknown outcome (the gate still fails)
    assert judged.count(qids[1]) == 2 and judged.count(qids[0]) == 1


@needs_harness
def test_malformed_judge_json_is_retried_up_to_the_stock_limit(offline: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, _ = _fake_bench(offline, 1)
    judged: list[int] = []

    def fake_http(method: str, path: str, body: dict[str, Any] | None = None, *_: Any, **__: Any) -> dict[str, Any]:
        if not body.get("response_format"):
            return {"choices": [{"message": {"content": "ANSWER: a"}}]}
        judged.append(1)
        return {"choices": [{"message": {"content": "not json"}}]}

    monkeypatch.setattr(bj, "http", fake_http)
    assert bj.main(_args(bench, 1)) == 2  # still malformed after 5 attempts: ERROR, and the gate stops the run
    assert len(judged) == bj.JUDGE_ATTEMPTS == 5


@needs_harness
def test_a_reasked_answer_is_judged_afresh(offline: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench, qids = _fake_bench(offline, 1)
    answers = iter(["", "ANSWER: second"])  # first answer empty, re-asked with --retry-bad-answers
    judged: list[str] = []

    def fake_http(method: str, path: str, body: dict[str, Any] | None = None, *_: Any, **__: Any) -> dict[str, Any]:
        if not body.get("response_format"):
            return {"choices": [{"message": {"content": next(answers)}}]}
        judged.append(body["messages"][-1]["content"])
        return {"choices": [{"message": {"content": '{"label": "CORRECT"}'}}]}

    monkeypatch.setattr(bj, "http", fake_http)
    assert bj.main(_args(bench, 1, "--force-judge")) == 0  # judged the empty answer
    assert bj.main(_args(bench, 1, "--retry-bad-answers")) == 0
    assert len(judged) == 2 and "second" in judged[1]  # the new answer got its own verdict
    out = json.loads((bench / f"results/locomo/predicted_fake__t/{qids[0]}.json").read_text())
    assert out["cutoff_results"]["top_10"]["generated_answer"] == "second"


def test_torn_results_line_is_ended_before_appending(tmp_path: Path) -> None:
    st = bj.State(tmp_path)
    st.results_path.write_text('{"custom_id": "a", "content": "x"}\n{"custom_id": "b", "cont')
    st.add_results([{"custom_id": "c", "content": "y"}])
    assert set(st.results()) == {"a", "c"}
