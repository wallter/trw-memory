"""Unit tests for the trw-jev decision seam (trw_memory.decisions).

All HTTP is faked via ``httpx.MockTransport`` — no real network access, per
the ``unit`` marker's contract.
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest

from trw_memory.decisions import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ChoiceQuestion,
    JevHttpJudge,
    NoulQuestion,
    NullJudge,
    ScoreQuestion,
    judge_from_env,
)
from trw_memory.decisions._dotenv import parse_dotenv_subset
from trw_memory.decisions._wire import normalize_noul_criteria

pytestmark = pytest.mark.unit


def _judge(handler, **kwargs) -> JevHttpJudge:
    transport = httpx.MockTransport(handler)
    return JevHttpJudge("sk-test-key", transport=transport, **kwargs)


# -- Happy path: all three question types in one call --------------------


def test_happy_path_all_three_question_types() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "typesafe/jev-1.13-20260917",
                "provider": "TypeSafe",
                "answers": {
                    "is_duplicate": {"type": "noul", "noul": 0.87},
                    "category": {
                        "type": "choice",
                        "choice": "pattern",
                        "probabilities": {"pattern": 0.71, "incident": 0.1},
                        "confidence": 0.8,
                    },
                    "actionability": {
                        "type": "score",
                        "score": 3.2,
                        "legend": {"0": "not actionable"},
                        "probabilities": {"0": 0.01},
                        "confidence": 0.6,
                    },
                },
                "usage": {"input_tokens": 812, "output_tokens": 0, "cost": 0.000034},
            },
        )

    judge = _judge(handler)
    questions = {
        "is_duplicate": NoulQuestion(
            instructions="Is this a duplicate?",
            criteria={"true": "Same root cause", "false": "Different cause"},
        ),
        "category": ChoiceQuestion(
            instructions="Classify the learning.",
            criteria={"incident": "...", "pattern": "..."},
        ),
        "actionability": ScoreQuestion(
            instructions="How actionable is it?",
            criteria=["not actionable", "vague", "concrete"],
        ),
    }

    result = judge.decide("state text", questions, session_id="sess-1")

    assert result is not None
    assert result.backend == "jev"
    assert result.model == "typesafe/jev-1.13-20260917"
    assert result.usage["cost"] == pytest.approx(0.000034)
    assert result.answers["is_duplicate"].type == "noul"
    assert result.answers["is_duplicate"].noul == pytest.approx(0.87)
    assert result.answers["category"].type == "choice"
    assert result.answers["category"].choice == "pattern"
    assert result.answers["actionability"].type == "score"
    assert result.answers["actionability"].score == pytest.approx(3.2)
    assert result.latency_ms >= 0

    # Wire shape: model default, session_id present, criteria strings.
    body = captured["body"]
    assert body["session_id"] == "sess-1"
    assert body["state"] == "state text"
    assert body["questions"]["is_duplicate"]["criteria"] == {
        "true": "Same root cause",
        "false": "Different cause",
    }
    # Key never leaks into the sent headers as anything but the bearer token
    # (sanity check the transport actually authenticated).
    assert captured["headers"]["authorization"] == "Bearer sk-test-key"


# -- Noul criteria normalization -------------------------------------------


def test_noul_criteria_normalization_fills_missing_side() -> None:
    only_true = normalize_noul_criteria({"true": "Same root cause and remedy"})
    assert only_true is not None
    assert only_true["true"] == "Same root cause and remedy"
    assert only_true.get("false")

    only_false = normalize_noul_criteria({"false": "Different cause"})
    assert only_false is not None
    assert only_false.get("true")

    assert normalize_noul_criteria(None) is None
    assert normalize_noul_criteria({}) is None

    both = normalize_noul_criteria({"true": "T", "false": "F"})
    assert both == {"true": "T", "false": "F"}


def test_noul_question_wire_payload_always_has_both_sides() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        criteria = body["questions"]["q1"]["criteria"]
        assert "true" in criteria
        assert "false" in criteria
        return httpx.Response(200, json={"model": "m", "answers": {}, "usage": {}})

    judge = _judge(handler)
    result = judge.decide(
        "state",
        {"q1": NoulQuestion(instructions="Q?", criteria={"true": "only true side"})},
    )
    assert result is not None


def test_non_string_instructions_and_criteria_are_json_encoded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        wire_q = body["questions"]["q1"]
        assert isinstance(wire_q["instructions"], str)
        assert json.loads(wire_q["instructions"]) == {"rule": "structured"}
        assert isinstance(wire_q["criteria"]["a"], str)
        assert json.loads(wire_q["criteria"]["a"]) == [1, 2, 3]
        return httpx.Response(200, json={"model": "m", "answers": {}, "usage": {}})

    judge = _judge(handler)
    result = judge.decide(
        "state",
        {"q1": ChoiceQuestion(instructions={"rule": "structured"}, criteria={"a": [1, 2, 3], "b": "plain"})},
    )
    assert result is not None


# -- Retry-after on 429 then success ---------------------------------------


def test_retry_after_429_then_success() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "rate limited"})
        return httpx.Response(200, json={"model": "m", "answers": {}, "usage": {}})

    judge = _judge(handler)
    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    assert calls["n"] == 2
    assert result is not None
    assert result.backend == "jev"


def test_retryable_status_exhausted_after_one_retry_returns_none() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(529, json={"error": "overloaded"})

    judge = _judge(handler)
    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    assert calls["n"] == 2  # exactly ONE retry, never more
    assert result is None


# -- Non-retryable errors abstain immediately ------------------------------


@pytest.mark.parametrize("status", [401, 422])
def test_client_errors_return_none_without_retry(status: int) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, json={"error": "nope"})

    judge = _judge(handler)
    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    assert result is None
    assert calls["n"] == 1  # 401/422 are not retryable


def test_timeout_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    judge = _judge(handler)
    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    assert result is None


def test_malformed_response_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    judge = _judge(handler)
    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    assert result is None


# -- No network when disabled ----------------------------------------------


def test_null_judge_never_makes_a_network_call() -> None:
    judge = NullJudge()
    result = judge.decide("anything", {"q": NoulQuestion(instructions="Q?")})
    assert result is None


def test_judge_from_env_disabled_returns_null_judge_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = judge_from_env(env={})
    assert isinstance(judge, NullJudge)

    judge = judge_from_env(env={"TRW_JEV_ENABLED": "false", "OPENROUTER_API_KEY": "sk-x"})
    assert isinstance(judge, NullJudge)

    judge = judge_from_env(env={"TRW_JEV_ENABLED": "true"})  # no key
    assert isinstance(judge, NullJudge)


# -- Factory resolution -----------------------------------------------------


def test_judge_from_env_resolves_jev_judge_from_process_env() -> None:
    judge = judge_from_env(
        env={
            "TRW_JEV_ENABLED": "1",
            "OPENROUTER_API_KEY": "sk-live",
            "TRW_JEV_BASE_URL": "https://openrouter.ai/api/beta/decisions",
            "TRW_JEV_MODEL": "~typesafe/jev-custom",
        }
    )
    assert isinstance(judge, JevHttpJudge)
    assert judge._base_url == "https://openrouter.ai/api/beta/decisions"
    assert judge._model == "~typesafe/jev-custom"


def test_judge_from_env_takes_only_the_api_key_from_the_dotenv(tmp_path) -> None:
    """release-verify R1: a repo-controlled ``.env`` may supply the key and nothing else."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "UNRELATED_SECRET=do-not-read-me",
                "OPENROUTER_API_KEY=sk-from-dotenv",
                "TRW_JEV_BASE_URL=http://attacker.example/decisions",
                "TRW_JEV_MODEL=~attacker/model",
            ]
        ),
        encoding="utf-8",
    )

    judge = judge_from_env(env={"TRW_JEV_ENABLED": "true"}, dotenv_path=dotenv)

    assert isinstance(judge, JevHttpJudge)
    assert judge._api_key == "sk-from-dotenv"
    assert judge._base_url == DEFAULT_BASE_URL
    assert judge._model == DEFAULT_MODEL


def test_dotenv_cannot_enable_the_jev_backend(tmp_path) -> None:
    """release-verify R1: enablement is process-env only, so a cloned repo cannot opt in for you."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "TRW_JEV_ENABLED=true\nOPENROUTER_API_KEY=sk-operator-real-secret\n",
        encoding="utf-8",
    )

    judge = judge_from_env(env={}, dotenv_path=dotenv)

    assert isinstance(judge, NullJudge)
    assert judge.decide("state", {"q": NoulQuestion(instructions="Q?")}) is None


@pytest.mark.parametrize(
    "base_url",
    [
        "http://openrouter.ai/api/alpha/decisions",  # plaintext, allowlisted host
        "https://attacker.example/decisions",  # https, host not allowlisted
        "https://openrouter.ai.attacker.example/d",  # suffix-confusion host
        "https://openrouter.ai@evil.example/d",  # userinfo: the real host is evil.example
        "https://openrouter.ai./x",  # trailing-dot FQDN, compared whole not normalized
        "https://openrouter.ai:8443/x",  # allowlisted host, non-default port
        "https://openrouter.ai]",  # malformed: urlsplit raises ValueError
        "https://openrouter.ai:notaport/x",  # malformed port: .port raises ValueError
        "not a url",
    ],
)
def test_judge_from_env_abstains_on_a_rejected_base_url(base_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """release-verify R1: no https + allowlisted host means NullJudge — the key is never sent."""

    def _no_http(*args, **kwargs):  # pragma: no cover - the assertion is that this is never reached
        raise AssertionError("a rejected base URL must not open an HTTP client")

    monkeypatch.setattr(httpx, "Client", _no_http)

    judge = judge_from_env(
        env={
            "TRW_JEV_ENABLED": "1",
            "OPENROUTER_API_KEY": "sk-operator-real-secret",
            "TRW_JEV_BASE_URL": base_url,
        }
    )

    assert isinstance(judge, NullJudge)
    # NullJudge holds no key and opens no client; the abstention is the proof no request was built.
    assert judge.decide("state", {"q": NoulQuestion(instructions="Q?")}) is None


@pytest.mark.parametrize(
    "base_url",
    [
        "https://openrouter.ai/api/alpha/decisions",
        "HTTPS://OPENROUTER.AI/api/alpha/decisions",  # urlsplit lowercases scheme and host
        "https://openrouter.ai:443/api/alpha/decisions",  # explicit default port
    ],
)
def test_judge_from_env_accepts_an_allowlisted_base_url(base_url: str) -> None:
    judge = judge_from_env(env={"TRW_JEV_ENABLED": "1", "OPENROUTER_API_KEY": "sk-live", "TRW_JEV_BASE_URL": base_url})

    assert isinstance(judge, JevHttpJudge)
    assert judge._base_url == base_url


def test_dotenv_parser_only_reads_allowlisted_keys(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "DATABASE_URL=postgres://should-not-appear",
                "OPENROUTER_API_KEY=sk-abc",
                "TRW_JEV_ENABLED=true",
                "TRW_JEV_MODEL=~typesafe/jev-latest",
                'export TRW_JEV_BASE_URL="https://quoted.example"',
            ]
        ),
        encoding="utf-8",
    )
    values = parse_dotenv_subset(dotenv, allowed_keys=frozenset({"OPENROUTER_API_KEY"}))
    # Allowlist-only: no prefix wildcard, so the TRW_JEV_* lines are never parsed (F9/R1).
    assert values == {"OPENROUTER_API_KEY": "sk-abc"}
    assert "DATABASE_URL" not in values


def test_dotenv_missing_file_returns_empty() -> None:
    values = parse_dotenv_subset("/nonexistent/path/.env", allowed_keys=frozenset({"OPENROUTER_API_KEY"}))
    assert values == {}


def test_dotenv_parser_survives_a_non_utf8_file(tmp_path) -> None:
    """F6: the file is repo-controlled, so a non-decodable one abstains rather than raising."""
    dotenv = tmp_path / ".env"
    dotenv.write_bytes(b"OPENROUTER_API_KEY=sk-abc\n\xff\xfe\x00binary\n")

    values = parse_dotenv_subset(dotenv, allowed_keys=frozenset({"OPENROUTER_API_KEY"}))

    assert values == {}
    assert isinstance(judge_from_env(env={"TRW_JEV_ENABLED": "1"}, dotenv_path=dotenv), NullJudge)


def test_dotenv_parser_refuses_a_symlink(tmp_path) -> None:
    """N3/N4: git can commit a symlink, so a cloned repo could point .env at any readable file."""
    real = tmp_path / "real-secrets"
    real.write_text("OPENROUTER_API_KEY=sk-not-yours\n", encoding="utf-8")
    dotenv = tmp_path / ".env"
    dotenv.symlink_to(real)

    assert parse_dotenv_subset(dotenv, allowed_keys=frozenset({"OPENROUTER_API_KEY"})) == {}


@pytest.mark.timeout(15)
def test_dotenv_parser_refuses_a_fifo_without_blocking(tmp_path) -> None:
    """N3: read_text() on a FIFO at .env blocked the MCP handler forever."""
    dotenv = tmp_path / ".env"
    os.mkfifo(dotenv)

    started = time.monotonic()
    values = parse_dotenv_subset(dotenv, allowed_keys=frozenset({"OPENROUTER_API_KEY"}))

    assert values == {}
    assert time.monotonic() - started < 5.0, "opening a FIFO must not block the caller"


def test_dotenv_parser_refuses_a_directory(tmp_path) -> None:
    """N3: S_ISREG is the gate, so every non-regular kind is refused the same way."""
    assert parse_dotenv_subset(tmp_path, allowed_keys=frozenset({"OPENROUTER_API_KEY"})) == {}


def test_dotenv_parser_refuses_an_oversized_file(tmp_path) -> None:
    """F6: a multi-megabyte blob is not the small key/value file this parser is for."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENROUTER_API_KEY=sk-abc\n" + ("# padding\n" * 20000), encoding="utf-8")
    assert dotenv.stat().st_size > 64 * 1024

    assert parse_dotenv_subset(dotenv, allowed_keys=frozenset({"OPENROUTER_API_KEY"})) == {}


# -- Key never appears in logs or repr -------------------------------------


def test_api_key_never_appears_in_repr() -> None:
    judge = JevHttpJudge("sk-super-secret-value")
    assert "sk-super-secret-value" not in repr(judge)
    assert "sk-super-secret-value" not in str(judge)


def test_api_key_never_logged_on_success(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m", "answers": {}, "usage": {"cost": 0.001}})

    judge = _judge(handler)
    judge.decide("state with secrets sk-super-secret-value", {"q": NoulQuestion(instructions="Q?")})

    for record in caplog.records:
        assert "sk-test-key" not in record.getMessage()


def test_api_key_never_logged_on_error(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    judge = _judge(handler)
    judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    for record in caplog.records:
        assert "sk-test-key" not in record.getMessage()


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def __getattr__(self, level: str):  # type: ignore[no-untyped-def]
        return lambda event, **kw: self.calls.append((level, event, kw))


def test_failure_logs_carry_only_the_exception_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse/payload failures must not log tracebacks or values derived from state or response."""
    from trw_memory.decisions import _jev_http

    recorder = _RecordingLogger()
    monkeypatch.setattr(_jev_http, "logger", recorder)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m", "answers": {"q": {"type": "bogus", "leak": "MARKER-SECRET"}}})

    assert _judge(handler).decide("MARKER-STATE", {"q": NoulQuestion(instructions="Q?")}) is None
    failures = [c for c in recorder.calls if c[1] == "jev_decision_parse_failed"]
    assert failures and failures[0][2]["error_type"] == "ValidationError"
    for _level, _event, kwargs in recorder.calls:
        assert "exc_info" not in kwargs
        assert "MARKER" not in repr(kwargs)


# -- Optional redact hook ----------------------------------------------------


def test_redact_hook_applied_before_send() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["state"] = json.loads(request.content)["state"]
        return httpx.Response(200, json={"model": "m", "answers": {}, "usage": {}})

    judge = _judge(handler, redact=lambda state: "REDACTED")
    judge.decide("raw sensitive state", {"q": NoulQuestion(instructions="Q?")})

    assert captured["state"] == "REDACTED"


def test_build_payload_accepts_plain_dict_questions() -> None:
    """Library callers naturally pass wire-shaped dicts; they must not be silently rejected."""
    from trw_memory.decisions._wire import build_payload

    payload = build_payload(
        "~typesafe/jev-latest",
        {"task": "x"},
        {"q": {"type": "noul", "instructions": "Delegate?", "criteria": {"true": "yes"}}},
        None,
    )
    assert payload["questions"]["q"]["type"] == "noul"
    assert set(payload["questions"]["q"]["criteria"]) == {"true", "false"}
