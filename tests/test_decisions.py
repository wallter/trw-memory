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
    ChoiceQuestion,
    DecisionFailure,
    DecisionResult,
    NoulQuestion,
    NullJudge,
    ScoreQuestion,
    judge_from_env,
)
from trw_memory.decisions._dotenv import parse_dotenv_subset
from trw_memory.decisions._jev_http import DEFAULT_BASE_URL, DEFAULT_MODEL, JevHttpJudge
from trw_memory.decisions._wire import normalize_noul_criteria

pytestmark = pytest.mark.unit


@pytest.fixture
def _isolated_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """An empty HOME (no ``.trw/config.yaml``), for tests exercising user-scope enablement
    hermetically rather than against whatever the real machine happens to have configured.
    """
    home = tmp_path / "home"
    (home / ".trw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    return home


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


def test_retryable_status_exhausted_after_one_retry_is_provider_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(529, json={"error": "overloaded"})

    judge = _judge(handler)
    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    assert calls["n"] == 2  # exactly ONE retry, never more
    assert isinstance(result, DecisionFailure) and result.kind == "provider_error"


def test_retry_is_skipped_when_retry_after_would_exhaust_the_call_budget() -> None:
    """The timeout bounds the whole call: no retry that could only start after the budget is spent."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, headers={"retry-after": "3"})

    started = time.monotonic()
    result = _judge(handler).decide("state", {"q": NoulQuestion(instructions="Q?")}, timeout_s=1.0)

    assert calls["n"] == 1
    assert time.monotonic() - started < 1.0
    assert isinstance(result, DecisionFailure) and result.kind == "provider_error"


def test_retry_gets_only_the_remaining_budget() -> None:
    timeouts: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"]["read"])
        if len(timeouts) == 1:
            return httpx.Response(429, headers={"retry-after": "0.2"})
        return httpx.Response(200, json={"model": "m", "answers": {"q": {"type": "noul", "noul": 0.4}}})

    result = _judge(handler).decide("state", {"q": NoulQuestion(instructions="Q?")}, timeout_s=2.0)

    assert isinstance(result, DecisionResult)
    assert timeouts[0] == 2.0
    assert 0 < float(timeouts[1]) < 1.8


# -- Non-retryable errors abstain immediately ------------------------------


@pytest.mark.parametrize(("status", "kind"), [(401, "auth"), (422, "invalid_request")])
def test_client_errors_fail_typed_without_retry(status: int, kind: str) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, json={"error": "nope"})

    judge = _judge(handler)
    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    assert isinstance(result, DecisionFailure) and result.kind == kind
    assert calls["n"] == 1  # 401/422 are not retryable


def test_timeout_is_a_timeout_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    judge = _judge(handler)
    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    assert isinstance(result, DecisionFailure) and result.kind == "timeout"


def test_malformed_response_is_a_malformed_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    judge = _judge(handler)
    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")})

    assert isinstance(result, DecisionFailure) and result.kind == "malformed_response"


# -- No network when disabled ----------------------------------------------


def test_null_judge_never_makes_a_network_call() -> None:
    judge = NullJudge()
    result = judge.decide("anything", {"q": NoulQuestion(instructions="Q?")})
    assert isinstance(result, DecisionFailure) and result.kind == "disabled"


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


def test_dotenv_path_alone_cannot_enable_the_jev_backend(tmp_path) -> None:
    """``dotenv_path`` with no ``project_root`` stays key-only: passing a bare dotenv path (the
    shape every pre-2026-09-23 caller uses) must not newly start reading it for enablement too.
    Project-scope enablement is opt-in via ``project_root`` — see
    ``test_project_root_enables_via_its_trw_config_yaml`` and friends below.
    """
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "TRW_JEV_ENABLED=true\nOPENROUTER_API_KEY=sk-operator-real-secret\n",
        encoding="utf-8",
    )

    judge = judge_from_env(env={}, dotenv_path=dotenv)

    assert isinstance(judge, NullJudge)
    assert judge.decide("state", {"q": NoulQuestion(instructions="Q?")}).kind == "disabled"


# -- Enablement precedence via resolve_backend_enablement (2026-09-23 operator decision) ----------


def test_project_root_enables_via_its_trw_config_yaml(tmp_path, _isolated_home) -> None:
    """A project may now enable the backend from its own ``.trw/config.yaml`` — the prior rule
    (project files may only disable) is relaxed; the key still comes from the project ``.env``.
    """
    project = tmp_path / "project"
    (project / ".trw").mkdir(parents=True)
    (project / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    (project / ".env").write_text("OPENROUTER_API_KEY=sk-project-key\n", encoding="utf-8")

    judge = judge_from_env(env={}, dotenv_path=project / ".env", project_root=project)

    assert isinstance(judge, JevHttpJudge)
    assert judge._api_key == "sk-project-key"


def test_project_root_enables_via_its_dotenv(tmp_path, _isolated_home) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("TRW_JEV_ENABLED=true\nOPENROUTER_API_KEY=sk-project-key\n", encoding="utf-8")

    judge = judge_from_env(env={}, dotenv_path=project / ".env", project_root=project)

    assert isinstance(judge, JevHttpJudge)


def test_process_env_beats_project_root(tmp_path, _isolated_home) -> None:
    project = tmp_path / "project"
    (project / ".trw").mkdir(parents=True)
    (project / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")

    judge = judge_from_env(env={"TRW_JEV_ENABLED": "false"}, project_root=project)

    assert isinstance(judge, NullJudge)


def test_project_root_beats_user_scope(tmp_path, _isolated_home) -> None:
    (_isolated_home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    project = tmp_path / "project"
    (project / ".trw").mkdir(parents=True)
    (project / ".trw" / "config.yaml").write_text("assess_enabled: false\n", encoding="utf-8")

    judge = judge_from_env(env={}, project_root=project)

    assert isinstance(judge, NullJudge)


def test_user_scope_enables_with_no_project_root(_isolated_home) -> None:
    (_isolated_home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")

    judge = judge_from_env(env={"OPENROUTER_API_KEY": "sk-machine-key"})

    assert isinstance(judge, JevHttpJudge)


def test_nothing_configured_resolves_off(_isolated_home) -> None:
    assert isinstance(judge_from_env(env={}), NullJudge)


# -- resolve_backend_enablement: fail-closed on a present-but-broken layer, and lazy evaluation ---


#: Round 3 review: table-driven coverage of every project-YAML outcome, each checked against a
#: PERMISSIVE user-scope ``assess_enabled: true`` -- only "absent" may legitimately cascade to it.
_PROJECT_YAML_CASES: list[tuple[str, object, tuple[bool, str]]] = [
    ("absent", None, (True, "~/.trw/config.yaml")),  # no .trw dir at all: genuinely absent
    ("refused_non_utf8", b"assess_enabled: true\n\xff\xfe\x00binary\n", (False, "project .trw/config.yaml")),
    ("refused_oversized", ("assess_enabled: true\n" + "#" * (65 * 1024)).encode(), (False, "project .trw/config.yaml")),
    ("malformed", "assess_enabled: [unterminated\n", (False, "project .trw/config.yaml")),
    ("not_a_mapping", "- just\n- a\n- list\n", (False, "project .trw/config.yaml")),
    ("blank", 'assess_enabled: ""\n', (False, "project .trw/config.yaml")),
    ("null", "assess_enabled:\n", (False, "project .trw/config.yaml")),
    ("non_bool_int", "assess_enabled: 1\n", (False, "project .trw/config.yaml")),
    ("key_missing", "some_other_key: 1\n", (True, "~/.trw/config.yaml")),  # valid file, no opinion
    ("explicit_false", "assess_enabled: false\n", (False, "project .trw/config.yaml")),
]


@pytest.mark.parametrize(("case", "content", "expected"), _PROJECT_YAML_CASES, ids=[c[0] for c in _PROJECT_YAML_CASES])
def test_project_yaml_layer_table(
    tmp_path, _isolated_home, case: str, content: object, expected: tuple[bool, str]
) -> None:
    from trw_memory.decisions._enablement import resolve_backend_enablement

    (_isolated_home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    if content is not None:
        (project / ".trw").mkdir()
        target = project / ".trw" / "config.yaml"
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")

    assert resolve_backend_enablement(project, env={}) == expected


def test_project_yaml_unreadable_lstat_fails_closed(tmp_path, _isolated_home, monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 3 review: ``lstat`` itself can fail for a reason OTHER than absence (e.g.
    ``PermissionError`` on a parent directory) -- that must fail closed too, not read as absent.
    """
    from pathlib import Path

    import trw_memory.decisions._enablement as enablement_mod

    (_isolated_home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    project = tmp_path / "project"
    (project / ".trw").mkdir(parents=True)
    target_path = project / ".trw" / "config.yaml"
    target_path.write_text("assess_enabled: true\n", encoding="utf-8")

    real_lstat = enablement_mod.os.lstat

    def _guarded_lstat(path, *a, **kw):  # type: ignore[no-untyped-def]
        if Path(path) == target_path:
            raise PermissionError(13, "Permission denied")
        return real_lstat(path, *a, **kw)

    monkeypatch.setattr(enablement_mod.os, "lstat", _guarded_lstat)

    assert enablement_mod.resolve_backend_enablement(project, env={}) == (False, "project .trw/config.yaml")


def test_project_yaml_symlink_fails_closed(tmp_path, _isolated_home) -> None:
    """A symlinked ``.trw/config.yaml`` EXISTS (``lstat`` succeeds on the link itself) but the
    hardened reader refuses to follow it -- refused, not absent.
    """
    from trw_memory.decisions._enablement import resolve_backend_enablement

    (_isolated_home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    project = tmp_path / "project"
    (project / ".trw").mkdir(parents=True)
    target = tmp_path / "elsewhere.yaml"
    target.write_text("assess_enabled: true\n", encoding="utf-8")
    (project / ".trw" / "config.yaml").symlink_to(target)

    assert resolve_backend_enablement(project, env={}) == (False, "project .trw/config.yaml")


#: Same table shape, for the project ``.env`` layer -- only the cases that structurally apply to a
#: line-based KEY=value file (no "not a mapping"/"null" equivalent; a dotenv value is always text).
_PROJECT_DOTENV_CASES: list[tuple[str, bytes | None, tuple[bool, str]]] = [
    ("absent", None, (True, "~/.trw/config.yaml")),
    ("refused_non_utf8", b"TRW_JEV_ENABLED=true\n\xff\xfe\x00binary\n", (False, "project .env")),
    ("refused_oversized", ("TRW_JEV_ENABLED=true\n" + "#" * (65 * 1024)).encode(), (False, "project .env")),
    # Round 4 review: the key is NAMED (well-formed or not), so every one of these is explicit
    # False, never absent -- unlike a genuinely absent key, which the last row below still cascades.
    ("blank", b"TRW_JEV_ENABLED=\n", (False, "project .env")),
    ("blank_export", b"export TRW_JEV_ENABLED=\n", (False, "project .env")),
    ("no_equals_sign", b"TRW_JEV_ENABLED true\n", (False, "project .env")),
    ("bare_key_no_value", b"TRW_JEV_ENABLED\n", (False, "project .env")),
    # Round 5 review: an '=' further along the line (a garbled tail) must not let the "no '='
    # anywhere" check miss this -- the key's own leading token is what decides.
    ("garbled_tail_with_equals", b"TRW_JEV_ENABLED true=1\n", (False, "project .env")),
    ("quoted_but_broken_value", b'TRW_JEV_ENABLED="unterminated\n', (False, "project .env")),
    ("non_truthy_string", b"TRW_JEV_ENABLED=maybe\n", (False, "project .env")),
    ("key_missing", b"OTHER_KEY=1\n", (True, "~/.trw/config.yaml")),
    ("explicit_false", b"TRW_JEV_ENABLED=false\n", (False, "project .env")),
    # A DIFFERENT key that merely shares TRW_JEV_ENABLED as a textual prefix must never be
    # mistaken for a malformed mention of it -- the leading-token match is exact, not a prefix.
    ("unrelated_key_sharing_a_prefix", b"TRW_JEV_ENABLED_OTHER=true\n", (True, "~/.trw/config.yaml")),
]


@pytest.mark.parametrize(
    ("case", "content", "expected"), _PROJECT_DOTENV_CASES, ids=[c[0] for c in _PROJECT_DOTENV_CASES]
)
def test_project_dotenv_layer_table(
    tmp_path, _isolated_home, case: str, content: bytes | None, expected: tuple[bool, str]
) -> None:
    from trw_memory.decisions._enablement import resolve_backend_enablement

    (_isolated_home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    if content is not None:
        (project / ".env").write_bytes(content)

    assert resolve_backend_enablement(project, env={}) == expected


def test_project_dotenv_unreadable_lstat_fails_closed(
    tmp_path, _isolated_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    import trw_memory.decisions._enablement as enablement_mod

    (_isolated_home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    target_path = project / ".env"
    target_path.write_text("TRW_JEV_ENABLED=true\n", encoding="utf-8")

    real_lstat = enablement_mod.os.lstat

    def _guarded_lstat(path, *a, **kw):  # type: ignore[no-untyped-def]
        if Path(path) == target_path:
            raise PermissionError(13, "Permission denied")
        return real_lstat(path, *a, **kw)

    monkeypatch.setattr(enablement_mod.os, "lstat", _guarded_lstat)

    assert enablement_mod.resolve_backend_enablement(project, env={}) == (False, "project .env")


def test_malformed_user_yaml_fails_closed(tmp_path, _isolated_home) -> None:
    (_isolated_home / ".trw" / "config.yaml").write_text("assess_enabled: [unterminated\n", encoding="utf-8")

    from trw_memory.decisions._enablement import resolve_backend_enablement

    assert resolve_backend_enablement(None, env={}) == (False, "~/.trw/config.yaml")


def test_blank_process_env_still_cascades_unlike_a_blank_yaml_or_dotenv_value(tmp_path, _isolated_home) -> None:
    """Blank-as-unset applies ONLY to a raw process-env value (an unresolved ``${env:X}``
    template a client forwards) -- not to a present blank YAML value, nor (round 4) to a
    human-written dotenv line naming the key with a blank assignment, both of which are now
    explicit False since round 2/4.
    """
    from trw_memory.decisions._enablement import resolve_backend_enablement

    (_isolated_home / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()

    assert resolve_backend_enablement(project, env={"TRW_JEV_ENABLED": ""}) == (True, "~/.trw/config.yaml")

    (project / ".env").write_text("TRW_JEV_ENABLED=\n", encoding="utf-8")
    assert resolve_backend_enablement(project, env={}) == (False, "project .env")


def test_an_explicit_process_env_setting_reads_no_file(
    tmp_path, _isolated_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-09-23 review fix: layers are evaluated lazily -- an explicit env value must
    short-circuit before any project or user config file is even opened.
    """
    from trw_memory.decisions import _enablement

    def _boom_yaml(*_a: object, **_kw: object) -> bool | None:
        raise AssertionError("a decided process-env layer must not read any YAML file")

    def _boom_dotenv(*_a: object, **_kw: object) -> bool | None:
        raise AssertionError("a decided process-env layer must not read any dotenv file")

    monkeypatch.setattr(_enablement, "_read_yaml_bool", _boom_yaml)
    monkeypatch.setattr(_enablement, "_read_dotenv_bool", _boom_dotenv)
    project = tmp_path / "project"
    project.mkdir()

    assert _enablement.resolve_backend_enablement(project, env={"TRW_JEV_ENABLED": "false"}) == (
        False,
        "TRW_JEV_ENABLED",
    )
    assert _enablement.resolve_backend_enablement(project, env={"TRW_JEV_ENABLED": "true"}) == (
        True,
        "TRW_JEV_ENABLED",
    )


def test_a_decided_project_layer_never_reads_the_user_yaml(
    tmp_path, _isolated_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.decisions import _enablement

    project = tmp_path / "project"
    (project / ".trw").mkdir(parents=True)
    (project / ".trw" / "config.yaml").write_text("assess_enabled: true\n", encoding="utf-8")

    real_read_yaml_bool = _enablement._read_yaml_bool

    def _guarded(path, key):  # type: ignore[no-untyped-def]
        if path == _isolated_home / ".trw" / "config.yaml":
            raise AssertionError("a decided project layer must not read the user-scope file")
        return real_read_yaml_bool(path, key)

    monkeypatch.setattr(_enablement, "_read_yaml_bool", _guarded)

    assert _enablement.resolve_backend_enablement(project, env={}) == (True, "project .trw/config.yaml")


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
    assert judge.decide("state", {"q": NoulQuestion(instructions="Q?")}).kind == "disabled"


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

    assert _judge(handler).decide("MARKER-STATE", {"q": NoulQuestion(instructions="Q?")}).kind == "malformed_response"
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


def test_a_live_provider_call_is_blocked_and_recorded(_no_live_decision_backend: list[str]) -> None:
    """The conftest guard: a judge on the production transport never reaches openrouter.ai."""
    judge = JevHttpJudge("sk-test-key")  # no mock transport, default openrouter.ai URL

    result = judge.decide("state", {"q": NoulQuestion(instructions="Q?")}, timeout_s=2.0)

    assert isinstance(result, DecisionFailure) and result.kind == "provider_error"
    assert _no_live_decision_backend == [f"POST {DEFAULT_BASE_URL}"]
    _no_live_decision_backend.clear()  # this test proves the guard; any other test fails at teardown


def test_jev_env_is_cleared_for_every_test(_isolated_home) -> None:
    assert not any(os.environ.get(name) for name in ("TRW_JEV_ENABLED", "OPENROUTER_API_KEY"))
    assert isinstance(judge_from_env(), NullJudge)
