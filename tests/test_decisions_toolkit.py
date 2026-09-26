"""Toolkit tests. No network: every judge here is a fake or an httpx.MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from trw_memory.decisions import NullJudge
from trw_memory.decisions._jev_http import JevHttpJudge
from trw_memory.decisions._models import DecisionFailure, DecisionResult
from trw_memory.decisions._redaction import default_redactor, redact_state
from trw_memory.decisions.toolkit import (
    MAX_CHOICE_OPTIONS,
    InvalidCriteria,
    InvalidRequest,
    Policy,
    Toolkit,
    choice,
    noul,
    reliability,
    score,
)

_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I3PlFUP0THsR8U"


class _FakeJudge:
    """Records what it was asked; answers from a canned map (defaults: noul 0.5)."""

    def __init__(self, answers: dict[str, Any] | None = None, *, fail: bool = False) -> None:
        self.answers = answers or {}
        self.fail = fail
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def decide(self, state, questions, *, timeout_s=10.0, session_id=None):  # type: ignore[no-untyped-def]
        self.calls.append((state, dict(questions)))
        if self.fail:
            return DecisionFailure(kind="provider_error", detail="fake failure")
        answers = {}
        for qid, q in questions.items():
            if qid in self.answers:
                answers[qid] = self.answers[qid]
            elif q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.5}
            elif q["type"] == "choice":
                first = next(iter(q["criteria"]))
                answers[qid] = {"type": "choice", "choice": first, "probabilities": {first: 1.0}, "confidence": 1.0}
            else:
                answers[qid] = {"type": "score", "score": 1.0, "legend": {}, "probabilities": {}}
        return DecisionResult(model="fake", answers=answers, usage={}, backend="fake", latency_ms=1.0)


# ---------------------------------------------------------------- ask: the primitive


def test_ask_returns_an_outcome_for_every_id_with_mixed_types() -> None:
    judge = _FakeJudge()
    result = Toolkit(judge).ask(
        {"ticket": "refund"},
        {
            "route": choice("Which queue?", {"billing": "money", "tech": "errors"}),
            "urgent": noul("Now?", true="blocked", false="can wait"),
            "sev": score("How bad?", ["nit", "minor", "major"]),
        },
    )
    assert set(result.outcomes) == {"route", "urgent", "sev"}
    assert result.status == "complete"
    assert result.choice("route").label == "billing"
    assert result.noul("urgent") == pytest.approx(0.5)
    assert result.score("sev").score == pytest.approx(1.0)
    assert len(judge.calls) == 1, "one call carries every question"


def test_ask_marks_missing_or_mistyped_answers_as_malformed_not_a_crash() -> None:
    class _Half(_FakeJudge):
        def decide(self, state, questions, *, timeout_s=10.0, session_id=None):  # type: ignore[no-untyped-def]
            return DecisionResult(
                model="fake",
                answers={"a": {"type": "noul", "noul": 0.9}, "b": {"type": "noul", "noul": 0.1}},
                usage={},
                backend="fake",
                latency_ms=1.0,
            )

    result = Toolkit(_Half()).ask(
        {},
        {
            "a": noul("?", true="y", false="n"),
            "b": choice("?", {"x": "x", "y": "y"}),  # answered with the wrong type
            "c": noul("?", true="y", false="n"),  # not answered at all
        },
    )
    assert result.status == "partial"
    assert result.noul("a") == pytest.approx(0.9)
    assert result.failures["b"].kind == "malformed_response" and "expected a choice" in result.failures["b"].detail
    assert result.failures["c"].kind == "malformed_response" and "missing" in result.failures["c"].detail
    assert result.choice("b").failure is not None


def test_ask_with_a_null_judge_reports_disabled_for_every_id() -> None:
    result = Toolkit(NullJudge()).ask({}, {"a": noul("?", true="y", false="n"), "b": score("?", ["lo", "hi"])})
    assert result.status == "failed"
    assert {f.kind for f in result.failures.values()} == {"disabled"}


def test_ask_passes_a_judge_failure_through_to_every_id() -> None:
    result = Toolkit(_FakeJudge(fail=True)).ask(
        {}, {"a": noul("?", true="y", false="n"), "b": score("?", ["lo", "hi"])}
    )
    assert result.status == "failed"
    assert {(f.kind, f.detail) for f in result.failures.values()} == {("provider_error", "fake failure")}


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"a": {"type": "verdict", "instructions": "?"}},
        {"a": {"type": "choice", "instructions": "?", "criteria": {}}},
        {"a": {"type": "noul", "instructions": "?", "criteria": {"yes": "y", "no": "n"}}},
    ],
)
def test_ask_raises_for_caller_errors_before_any_call(questions: dict[str, Any]) -> None:
    judge = _FakeJudge()
    with pytest.raises(InvalidRequest):
        Toolkit(judge).ask({}, questions)
    assert judge.calls == []


def test_over_cap_options_are_a_caller_error_at_build_and_at_ask() -> None:
    with pytest.raises(InvalidRequest, match="255"):
        choice("?", {f"o{i}": f"option {i}" for i in range(MAX_CHOICE_OPTIONS + 1)})
    judge = _FakeJudge()
    raw = {"type": "choice", "instructions": "?", "criteria": {f"o{i}": f"option {i}" for i in range(256)}}
    with pytest.raises(InvalidRequest, match="255"):
        Toolkit(judge).ask({}, {"a": raw})
    assert judge.calls == []


def test_wrong_noul_criteria_keys_name_the_fix() -> None:
    """Enforced at parse time (NoulQuestion, _models.py) now, not only at ask()-time; the message
    names the bad keys and points at the right ones (2026-09-24 usage audit)."""
    bad = {"type": "noul", "instructions": "?", "criteria": {"yes": "x", "no": "y"}}
    with pytest.raises(InvalidRequest, match=r"true.*false") as excinfo:
        Toolkit(_FakeJudge()).ask({}, {"a": bad})
    assert "yes" in str(excinfo.value) and "no" in str(excinfo.value)


def test_whole_request_is_redacted_state_instructions_criteria() -> None:
    judge = _FakeJudge()
    Toolkit(judge).ask(
        {"note": f"token {_JWT}", "password": ["opaque-canary"], "credential": 12345, "count": 3},
        {"a": noul(f"Compare with {_JWT}", true=f"like {_JWT}", false="no")},
    )
    state, questions = judge.calls[0]
    sent = json.dumps(state) + json.dumps(questions)
    assert _JWT not in sent and "opaque-canary" not in sent and "12345" not in sent
    assert state["count"] == 3


def test_redaction_can_be_disabled_for_callers_that_already_scrubbed() -> None:
    judge = _FakeJudge()
    Toolkit(judge, redactor=None).ask({"note": _JWT}, {"a": noul("?", true="y", false="n")})
    assert _JWT in json.dumps(judge.calls[0][0])


def test_to_wire_renders_answers_failures_and_choice_margin() -> None:
    judge = _FakeJudge(
        {"r": {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.7}}
    )
    wire = Toolkit(judge).ask({}, {"r": choice("?", {"a": "a", "b": "b"})}).to_wire()
    assert wire["r"]["margin"] == pytest.approx(0.4)
    assert "advice" not in wire["r"], "margin 0.4 is well clear of the near-tie threshold (0.2)"
    wire2 = Toolkit(NullJudge()).ask({}, {"r": choice("?", {"a": "a"})}).to_wire()
    assert wire2["r"]["failure"]["kind"] == "disabled"


# ---------------------------------------------------------------- near-tie advice (W19, PRD-CORE-295)


def test_choice_below_margin_threshold_carries_advice() -> None:
    judge = _FakeJudge(
        {"r": {"type": "choice", "choice": "a", "probabilities": {"a": 0.55, "b": 0.45}, "confidence": 0.55}}
    )
    wire = Toolkit(judge).ask({}, {"r": choice("?", {"a": "a", "b": "b"})}).to_wire()
    assert wire["r"]["margin"] == pytest.approx(0.1)
    assert wire["r"]["advice"] == "near-tie: take the safer or reversible option, or ask."


def test_choice_above_margin_threshold_carries_no_advice() -> None:
    """A margin clearly at/above NEAR_TIE_MARGIN (0.2) is decisive enough for no advice."""
    judge = _FakeJudge(
        {"r": {"type": "choice", "choice": "a", "probabilities": {"a": 0.625, "b": 0.375}, "confidence": 0.625}}
    )
    wire = Toolkit(judge).ask({}, {"r": choice("?", {"a": "a", "b": "b"})}).to_wire()
    assert wire["r"]["margin"] >= 0.2
    assert "advice" not in wire["r"]


@pytest.mark.parametrize("probability", [0.4, 0.5, 0.6])
def test_noul_inside_the_near_tie_band_carries_advice(probability: float) -> None:
    judge = _FakeJudge({"r": {"type": "noul", "noul": probability}})
    wire = Toolkit(judge).ask({}, {"r": noul("?", true="y", false="n")}).to_wire()
    assert wire["r"]["advice"] == "near-tie: take the safer or reversible option, or ask."


@pytest.mark.parametrize("probability", [0.0, 0.1, 0.39, 0.61, 0.9, 1.0])
def test_noul_outside_the_near_tie_band_carries_no_advice(probability: float) -> None:
    judge = _FakeJudge({"r": {"type": "noul", "noul": probability}})
    wire = Toolkit(judge).ask({}, {"r": noul("?", true="y", false="n")}).to_wire()
    assert "advice" not in wire["r"]


def test_score_answers_never_carry_advice() -> None:
    """Only choice margin and noul probability are near-tie signals (W19 scope); score is not."""
    judge = _FakeJudge({"r": {"type": "score", "score": 1.0, "legend": {}, "probabilities": {}}})
    wire = Toolkit(judge).ask({}, {"r": score("?", ["lo", "mid", "hi"])}).to_wire()
    assert "advice" not in wire["r"]


def test_a_decision_failure_never_carries_advice() -> None:
    wire = Toolkit(NullJudge()).ask({}, {"r": noul("?", true="y", false="n")}).to_wire()
    assert "advice" not in wire["r"] and "failure" in wire["r"]


# ---------------------------------------------------------------- seam failure kinds (HTTP)


def _http_judge(handler):  # type: ignore[no-untyped-def]
    return JevHttpJudge("k", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    ("status", "body", "kind", "detail_has"),
    [
        (401, {}, "auth", "401"),
        (400, {"detail": "Too many choices. Must have at most 255 choices."}, "invalid_request", "max 255"),
        (400, {"detail": "your secret state echoed back"}, "invalid_request", "HTTP 400"),
        (422, {}, "invalid_request", "422"),
        (500, {}, "provider_error", "500"),
    ],
)
def test_http_status_maps_to_a_failure_kind_without_echoing_the_body(status, body, kind, detail_has) -> None:  # type: ignore[no-untyped-def]
    judge = _http_judge(lambda req: httpx.Response(status, json=body))
    result = Toolkit(judge, redactor=None).ask({"s": 1}, {"a": noul("?", true="y", false="n")})
    failure = result.failures["a"]
    assert failure.kind == kind and detail_has in failure.detail
    assert "secret state" not in failure.detail


def test_http_429_after_retry_is_rate_limited() -> None:
    judge = _http_judge(lambda req: httpx.Response(429, headers={"retry-after": "0"}))
    failure = Toolkit(judge, redactor=None).ask({}, {"a": noul("?", true="y", false="n")}).failures["a"]
    assert failure.kind == "rate_limited"


def test_http_timeout_is_timeout() -> None:
    def _raise(req):  # type: ignore[no-untyped-def]
        raise httpx.ReadTimeout("slow", request=req)

    failure = Toolkit(_http_judge(_raise), redactor=None).ask({}, {"a": noul("?", true="y", false="n")}).failures["a"]
    assert failure.kind == "timeout"


def test_http_200_with_unparseable_body_is_malformed_response() -> None:
    judge = _http_judge(lambda req: httpx.Response(200, content=b"not json"))
    failure = Toolkit(judge, redactor=None).ask({}, {"a": noul("?", true="y", false="n")}).failures["a"]
    assert failure.kind == "malformed_response"


# ---------------------------------------------------------------- item batching


def _ceiling_handler(max_questions: int, calls: list[int]):  # type: ignore[no-untyped-def]
    """Refuse requests over ``max_questions`` (or carrying HUGE) the way the provider does."""

    def handler(request: httpx.Request) -> httpx.Response:
        questions = json.loads(request.content)["questions"]
        calls.append(len(questions))
        if len(questions) > max_questions or "HUGE" in request.content.decode():
            return httpx.Response(400, json={"detail": "max_tokens_exceeded"})
        answers = {qid: {"type": "noul", "noul": 0.5} for qid in questions}
        return httpx.Response(200, json={"model": "m", "answers": answers, "usage": {"cost": 0.001}})

    return handler


def test_batch_items_splits_a_chunk_that_exceeds_the_token_ceiling() -> None:
    calls: list[int] = []
    kit = Toolkit(_http_judge(_ceiling_handler(2, calls)), redactor=None)
    q = {"hit": noul("Relevant?", true="relevant", false="not relevant")}

    batch = kit.batch_items({k: k for k in "abcd"}, q, chunk_size=4)

    assert batch.status == "complete"
    assert calls == [4, 2, 2]  # refused whole, then two halves
    assert batch.chunk_of == {"a": 0, "b": 0, "c": 1, "d": 1}
    assert list(batch.per_item) == ["a", "b", "c", "d"]


def test_batch_items_fails_only_the_item_too_big_on_its_own() -> None:
    calls: list[int] = []
    kit = Toolkit(_http_judge(_ceiling_handler(10, calls)), redactor=None)
    q = {"hit": noul("Relevant?", true="relevant", false="not relevant")}

    batch = kit.batch_items({"a": "small", "b": "HUGE", "c": "small"}, q, chunk_size=3)

    assert batch.status == "partial" and batch.unanswered == ["b"]
    failure = batch.per_item["b"].failures["hit"]
    assert failure.kind == "invalid_request" and "token ceiling" in failure.detail
    assert batch.per_item["a"].noul("hit") == 0.5 and batch.per_item["c"].noul("hit") == 0.5


def test_batch_items_asks_mixed_questions_per_item_in_one_call() -> None:
    judge = _FakeJudge()
    result = Toolkit(judge).batch_items(
        {"f1": {"finding": "JWT leaked"}, "f2": {"finding": "typo"}},
        {
            "fix": noul("Fix before release?", true="defect", false="cosmetic"),
            "sev": score("Severity", ["nit", "major"]),
        },
    )
    assert len(judge.calls) == 1
    assert set(result.per_item) == {"f1", "f2"} and result.status == "complete"
    assert result.per_item["f1"].noul("fix") == pytest.approx(0.5)
    assert result.per_item["f2"].score("sev").answered


def test_embedded_schema_puts_each_item_in_its_own_question_and_keeps_state_minimal() -> None:
    judge = _FakeJudge()
    Toolkit(judge).batch_items(
        {"a": {"finding": "JWT leaked"}, "b": {"finding": "typo"}}, {"q": noul("?", true="y", false="n")}
    )
    state, questions = judge.calls[0]
    assert "JWT leaked" not in json.dumps(state) and "typo" not in json.dumps(state)
    texts = [q["instructions"] for q in questions.values()]
    assert sum("JWT leaked" in t for t in texts) == 1 and sum("typo" in t for t in texts) == 1
    assert not any("JWT leaked" in t and "typo" in t for t in texts)


def test_keyed_schema_shares_state_and_points_each_question_at_its_key() -> None:
    judge = _FakeJudge()
    Toolkit(judge).batch_items(
        {"finding-jwt": {"t": 1}, "finding-naming": {"t": 2}}, {"q": noul("?", true="y", false="n")}, schema="keyed"
    )
    state, questions = judge.calls[0]
    assert set(state["items"]) == {"finding-jwt", "finding-naming"}
    texts = sorted(q["instructions"] for q in questions.values())
    assert any('"finding-jwt"' in t for t in texts) and any('"finding-naming"' in t for t in texts)
    assert len(set(texts)) == 2


def test_embedded_item_text_is_redacted() -> None:
    judge = _FakeJudge()
    Toolkit(judge).batch_items({"a": {"note": f"token {_JWT}"}}, {"q": noul("?", true="y", false="n")})
    assert _JWT not in json.dumps(judge.calls[0][1])


def test_batch_items_rejects_unknown_schema_and_bad_chunk_before_any_call() -> None:
    judge = _FakeJudge()
    with pytest.raises(InvalidRequest):
        Toolkit(judge).batch_items({"a": {}}, {"q": noul("?", true="y", false="n")}, schema="typo")  # type: ignore[arg-type]
    with pytest.raises(InvalidRequest):
        Toolkit(judge).batch_items({"a": {}}, {"q": noul("?", true="y", false="n")}, chunk_size=0)
    # A malformed question is the caller's error too, not a KeyError from building item requests.
    # 'question' is the recurring typo (2026-09-24 usage audit) — the message names 'instructions'.
    with pytest.raises(InvalidRequest, match="use 'instructions', not 'question'"):
        Toolkit(judge).batch_items({"a": {}}, {"q": {"type": "noul", "question": "?"}})
    assert judge.calls == []


def test_batch_items_chunks_and_records_chunk_provenance() -> None:
    judge = _FakeJudge()
    result = Toolkit(judge).batch_items(
        {f"i{n}": {} for n in range(5)}, {"q": noul("?", true="y", false="n")}, chunk_size=2
    )
    assert len(judge.calls) == 3 and sorted(set(result.chunk_of.values())) == [0, 1, 2]


def test_rank_rejects_criteria_missing_true_and_false() -> None:
    with pytest.raises(InvalidCriteria):
        Toolkit(_FakeJudge()).rank({"a": {}}, instructions="?", criteria={"maybe": "x"})


def test_rank_orders_and_reports_unanswered_with_chunk() -> None:
    class _HalfFails(_FakeJudge):
        def decide(self, state, questions, *, timeout_s=10.0, session_id=None):  # type: ignore[no-untyped-def]
            self.calls.append((state, dict(questions)))
            if len(self.calls) > 1:
                return DecisionFailure(kind="provider_error", detail="fake failure")
            return DecisionResult(
                model="fake",
                answers={qid: {"type": "noul", "noul": 0.9} for qid in questions},
                usage={},
                backend="fake",
                latency_ms=1.0,
            )

    result = Toolkit(_HalfFails()).rank(
        {f"i{n}": {} for n in range(4)}, instructions="?", criteria={"true": "a", "false": "b"}, chunk_size=2
    )
    assert [r.key for r in result.ranked] == ["i0", "i1"] and result.unanswered == ["i2", "i3"]
    assert result.status == "partial" and {r.chunk for r in result.ranked} == {0}
    assert not hasattr(result, "__iter__")


# ---------------------------------------------------------------- conveniences, policy, redaction


def test_policy_decide_uses_the_results_own_margin_and_validates_inputs() -> None:
    judge = _FakeJudge({"_c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.51, "b": 0.49}}})
    result = Toolkit(judge).ask({}, {"_c": choice("Pick.", {"a": "x", "b": "y"})}).choice("_c")
    assert Policy(act_at=0.4).route(0.51) == "act"
    assert Policy(act_at=0.4).decide(result) == "escalate"
    with pytest.raises(ValueError):
        Policy(act_at=0.7).route(1.2)
    with pytest.raises(ValueError):
        Policy(act_at=1.5)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.95, "act"), (0.71, "escalate"), (0.69, "escalate"), (0.40, "escalate"), (0.10, "abstain"), (None, "abstain")],
)
def test_policy_routes_with_a_dead_band(probability: float | None, expected: str) -> None:
    assert Policy(act_at=0.7, escalate_below=0.2).route(probability) == expected


def test_secret_keyed_values_are_dropped_whatever_their_type_and_camel_case_is_caught() -> None:
    out = redact_state(
        {"password": ["x"], "credential": 12345, "accessToken": "abc", "maxTokens": 256, "note": "fine"},
        default_redactor,
    )
    assert out["password"] == out["credential"] == out["accessToken"] == "<REDACTED:secret>"
    assert out["maxTokens"] == 256 and out["note"] == "fine"


def test_reliability_reports_ece_and_brier() -> None:
    perfect = [(1.0, True)] * 10 + [(0.0, False)] * 10
    report = reliability(perfect, bins=5)
    assert report["n"] == 20 and report["ece"] == pytest.approx(0.0) and report["brier"] == pytest.approx(0.0)
    assert reliability([(0.95, False)] * 10)["ece"] == pytest.approx(0.95)


def test_decision_failure_helpers() -> None:
    assert not DecisionFailure(kind="auth").retryable
    assert DecisionFailure(kind="rate_limited").retryable


# ---------------------------------------------------------------- CLI exit codes (a contract)


def test_cli_exit_codes_distinguish_caller_error_failed_and_complete(tmp_path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """0 complete / 2 caller error / 3 failed; automation branches on these, so they are pinned."""
    from trw_memory.decisions import cli

    state = tmp_path / "s.json"
    state.write_text("{}", encoding="utf-8")
    good = tmp_path / "q.json"
    good.write_text(json.dumps({"a": noul("?", true="y", false="n")}), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"a": {"type": "noul", "instructions": "?", "criteria": {"yes": "y"}}}), encoding="utf-8")

    monkeypatch.setattr(cli, "toolkit_from_env", lambda *a, **k: Toolkit(NullJudge()))
    assert cli.main(["ask", "--state-file", str(state), "--questions-file", str(good)]) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert cli.main(["ask", "--state-file", str(state), "--questions-file", str(bad)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "caller_error"

    monkeypatch.setattr(cli, "toolkit_from_env", lambda *a, **k: Toolkit(_FakeJudge()))
    assert cli.main(["ask", "--state-file", str(state), "--questions-file", str(good)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


# ---------------------------------------------------------------- round-2 regressions


def test_batch_ids_cannot_collide_across_hostile_keys() -> None:
    """Items 'a' and 'a\x1fb' with questions 'b\x1fc' and 'c' used to collapse to 3 wire ids."""

    class _ByText(_FakeJudge):
        def decide(self, state, questions, *, timeout_s=10.0, session_id=None):  # type: ignore[no-untyped-def]
            self.calls.append((state, dict(questions)))
            answers = {
                qid: {"type": "noul", "noul": 0.9 if "HIGH" in q["instructions"] else 0.1}
                for qid, q in questions.items()
            }
            return DecisionResult(model="m", answers=answers, usage={}, backend="f", latency_ms=1.0)

    judge = _ByText()
    result = Toolkit(judge).batch_items(
        {"a": "LOW", "a\x1fb": "HIGH"},
        {"b\x1fc": noul("?", true="y", false="n"), "c": noul("?", true="y", false="n")},
    )
    assert len(judge.calls[0][1]) == 4
    assert result.per_item["a"].noul("b\x1fc") == pytest.approx(0.1)
    assert result.per_item["a\x1fb"].noul("c") == pytest.approx(0.9)


def test_structured_instructions_and_criteria_are_redacted_too() -> None:
    judge = _FakeJudge()
    Toolkit(judge).ask(
        {},
        {
            "a": {
                "type": "noul",
                "instructions": {"password": "opaque-canary"},
                "criteria": {"true": {"note": "someone@example.com"}, "false": "n"},
            }
        },
    )
    sent = json.dumps(judge.calls[0][1])
    assert "opaque-canary" not in sent and "someone@example.com" not in sent


def test_impossible_answers_are_malformed_not_complete() -> None:
    invented = _FakeJudge({"_c": {"type": "choice", "choice": "invented", "probabilities": {"invented": 1.0}}})
    r = Toolkit(invented).ask({}, {"_c": choice("Pick.", {"a": "x", "b": "y"})}).choice("_c")
    assert not r.answered and r.failure is not None and r.failure.kind == "malformed_response"
    big = _FakeJudge({"_s": {"type": "score", "score": 999.0, "legend": {}, "probabilities": {}}})
    assert not Toolkit(big).ask({}, {"_s": score("Rate.", ["lo", "hi"])}).score("_s").answered


def test_one_malformed_member_does_not_sink_valid_siblings() -> None:
    judge = _http_judge(
        lambda req: httpx.Response(
            200, json={"model": "m", "answers": {"a": {"type": "noul", "noul": 0.9}, "b": {"type": "noul", "noul": 2}}}
        )
    )
    result = Toolkit(judge, redactor=None).ask(
        {}, {"a": noul("?", true="y", false="n"), "b": noul("?", true="y", false="n")}
    )
    assert result.noul("a") == pytest.approx(0.9)
    assert result.failures["b"].kind == "malformed_response" and result.status == "partial"


def test_provider_error_text_is_never_copied_into_the_detail() -> None:
    judge = _http_judge(
        lambda req: httpx.Response(
            400, json={"detail": "max_tokens_exceeded; password=opaque-canary someone@example.com"}
        )
    )
    failure = Toolkit(judge, redactor=None).ask({}, {"a": noul("?", true="y", false="n")}).failures["a"]
    assert failure.kind == "invalid_request" and "token ceiling" in failure.detail
    assert "opaque-canary" not in failure.detail and "example.com" not in failure.detail


def test_batch_context_dict_is_preserved_in_both_schemas() -> None:
    judge = _FakeJudge()
    kit = Toolkit(judge)
    kit.batch_items({"i": "ticket"}, {"q": noul("?", true="y", false="n")}, context={"priority": "urgent"})
    assert judge.calls[0][0] == {"priority": "urgent"}
    kit.batch_items(
        {"i": "ticket"}, {"q": noul("?", true="y", false="n")}, context={"priority": "urgent"}, schema="keyed"
    )
    assert judge.calls[1][0]["_context"] == {"priority": "urgent"}


def test_reliability_rejects_nonsense_inputs() -> None:
    with pytest.raises(ValueError):
        reliability([(0.9, True)], bins=0)
    with pytest.raises(ValueError):
        reliability([(1.5, True)])
    with pytest.raises(TypeError):
        reliability([(0.5, 1)])  # type: ignore[list-item]


# ---------------------------------------------------------------- ported from the 5.0.0 hardening (R2/N1/N2/F1)


def test_question_ids_are_redacted_on_the_wire_and_mapped_back() -> None:
    """N1: the id is a JSON key in the POST body; the caller still reads its own id back."""
    judge = _FakeJudge()
    result = Toolkit(judge).ask({}, {f"leak-{_JWT}": noul("?", true="y", false="n")})
    _, sent = judge.calls[0]
    assert _JWT not in json.dumps(sent)
    assert result.noul(f"leak-{_JWT}") == pytest.approx(0.5)


def test_colliding_question_ids_stay_two_questions() -> None:
    """N1+N2: two ids that redact to the same placeholder must not collapse."""
    judge = _FakeJudge()
    result = Toolkit(judge).ask(
        {},
        {
            "alice@example.com": noul("first", true="y", false="n"),
            "bob@example.com": noul("second", true="y", false="n"),
        },
    )
    _, sent = judge.calls[0]
    assert len(sent) == 2 and "example.com" not in json.dumps(sent)
    assert result.status == "complete" and len(result.outcomes) == 2


def test_choice_criteria_keep_every_option_through_redaction_and_answers_map_to_wire_labels() -> None:
    """N2: last-write-wins used to drop an option after validation passed."""

    class _Echo(_FakeJudge):
        def decide(self, state, questions, *, timeout_s=10.0, session_id=None):  # type: ignore[no-untyped-def]
            self.calls.append((state, dict(questions)))
            ((qid, q),) = questions.items()
            labels = list(q["criteria"])
            return DecisionResult(
                model="m",
                answers={
                    qid: {"type": "choice", "choice": labels[1], "probabilities": {labels[0]: 0.2, labels[1]: 0.8}}
                },
                usage={},
                backend="f",
                latency_ms=1.0,
            )

    judge = _Echo()
    result = (
        Toolkit(judge)
        .ask({}, {"_c": choice("who?", {"alice@example.com": "approve", "bob@example.com": "reject"})})
        .choice("_c")
    )
    _, sent = judge.calls[0]
    crit = next(iter(sent.values()))["criteria"]
    assert len(crit) == 2 and "example.com" not in json.dumps(crit)
    assert result.answered and result.label in crit and result.margin == pytest.approx(0.6)


def test_state_keys_that_redact_alike_are_kept_apart_and_tuples_become_lists() -> None:
    out = redact_state({"alice@example.com": "approve", "bob@example.com": "reject"}, default_redactor)
    assert len(out) == 2 and sorted(out.values()) == ["approve", "reject"]
    token = "abcdefghijklmnopqrstuvwxyz0123"
    out2 = redact_state({"notes": ("plain note", f"Bearer {token}", 7)}, default_redactor)
    assert isinstance(out2["notes"], list) and token not in json.dumps(out2)
    assert out2["notes"][0] == "plain note" and out2["notes"][2] == 7


def test_secret_bearing_dict_keys_and_containers_under_secret_keys_are_redacted() -> None:
    out = redact_state({"sk-or-v1-abcdef0123456789abcdef": "some value", "plain": 1}, default_redactor)
    assert "sk-or-v1-abcdef0123456789abcdef" not in json.dumps(out) and out["plain"] == 1
    out2 = redact_state(
        {
            "password": ["plainlist1", {"deep": "plainlist2"}],
            "credentials": {"user": "ada"},
            "api_key": ("t",),
            "retries": 3,
        },
        default_redactor,
    )
    assert out2["password"] == out2["credentials"] == out2["api_key"] == "<REDACTED:secret>" and out2["retries"] == 3
    assert "plainlist" not in json.dumps(out2)


def test_validation_error_text_does_not_echo_question_prose() -> None:
    """F1: a malformed question must not reflect its raw prose in the error."""
    with pytest.raises(InvalidRequest) as excinfo:
        Toolkit(_FakeJudge()).ask(
            {}, {"q": {"type": "choice", "instructions": f"use {_JWT}", "criteria": {"only": "x"}, "note": 1}}
        )
    assert _JWT not in str(excinfo.value)
