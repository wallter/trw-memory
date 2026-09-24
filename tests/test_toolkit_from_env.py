"""``toolkit_from_env`` — the one Toolkit constructor every caller SHOULD use (PRD-CORE-295-FR01).

The disabled-path import-graph assertion runs in a SUBPROCESS: this test file itself, and every
other test module collected in the same pytest session, may already have imported httpx (e.g.
via ``trw_memory.decisions._jev_http`` from an unrelated test), which would make an in-process
``sys.modules`` check meaningless.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from trw_memory.decisions import toolkit_from_env
from trw_memory.decisions._judge import DecisionState
from trw_memory.decisions._models import DecisionFailure, DecisionResult

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_disabled_path_never_imports_jev_http_or_httpx() -> None:
    """A fresh interpreter, Jev off: neither ``_jev_http`` nor ``httpx`` ever loads."""
    script = (
        "import sys\n"
        "import trw_memory.decisions as d\n"
        "kit = d.toolkit_from_env(env={}, redactor=d.default_redactor)\n"
        "assert isinstance(kit, d.Toolkit), type(kit)\n"
        "assert 'trw_memory.decisions._jev_http' not in sys.modules, 'jev_http loaded on the disabled path'\n"
        "assert 'httpx' not in sys.modules, 'httpx loaded on the disabled path'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": _SRC},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip() == "OK"


class _StubJudge:
    """Records every call; answers a fixed noul so the wrapping tests can assert call counts."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def decide(
        self, state: DecisionState, questions: Any, *, timeout_s: float = 10.0, session_id: str | None = None
    ) -> DecisionResult | DecisionFailure:
        self.calls.append((state, dict(questions)))
        return DecisionResult(
            model="stub",
            answers={qid: {"type": "noul", "noul": 0.5} for qid in questions},
            usage={},
            backend="stub",
            latency_ms=1.0,
        )


_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I3PlFUP0THsR8U"
_ENABLED = {"TRW_JEV_ENABLED": "true", "OPENROUTER_API_KEY": "sk-or-test-key"}
_QUESTIONS = {"a": {"type": "noul", "instructions": f"is {_JWT} valid?", "criteria": {"true": "y", "false": "n"}}}


def _capture_egress(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Route the enabled HTTP judge through a mock transport and record every outbound body."""
    import httpx

    from trw_memory.decisions import _jev_http

    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(500, json={"error": "stub"})

    real_client = httpx.Client
    monkeypatch.setattr(
        _jev_http.httpx, "Client", lambda *a, **k: real_client(transport=httpx.MockTransport(handler), timeout=5)
    )
    return bodies


def test_the_redactor_is_required() -> None:
    """Cross-vendor FR01 P1: there is no way to build a toolkit that sends unredacted text."""
    with pytest.raises(TypeError, match="needs a redactor"):
        toolkit_from_env(env={}, redactor=None)  # type: ignore[arg-type]


def test_an_enabled_http_judge_never_sends_a_jwt(monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.decisions._redaction import default_redactor

    bodies = _capture_egress(monkeypatch)
    kit = toolkit_from_env(env=_ENABLED, redactor=default_redactor)

    kit.ask({"note": f"token {_JWT} leaked in a log"}, _QUESTIONS)

    assert bodies, "the enabled judge made no request"
    assert all(_JWT.encode() not in body for body in bodies)


def test_the_cli_entry_point_never_sends_a_jwt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """P2: the decisions CLI, end to end, with the backend enabled and a stubbed transport."""
    import json

    from trw_memory.decisions import cli

    bodies = _capture_egress(monkeypatch)
    for key, value in _ENABLED.items():
        monkeypatch.setenv(key, value)
    state, questions = tmp_path / "state.json", tmp_path / "questions.json"
    state.write_text(json.dumps({"note": f"token {_JWT}"}), encoding="utf-8")
    questions.write_text(json.dumps(_QUESTIONS), encoding="utf-8")

    cli.main(["ask", "--state-file", str(state), "--questions-file", str(questions)])

    assert bodies and all(_JWT.encode() not in body for body in bodies)
