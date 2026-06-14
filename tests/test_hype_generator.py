"""PRD-CORE-195 FR02 — QuestionGenerator protocol + NoOp default + injection.

NFR03/NG1: the engine ships no LLM dependency — assert the HyPE module import
graph never pulls in an LLM runtime.
"""

from __future__ import annotations

import sys

import pytest

from trw_memory.hype import NoOpQuestionGenerator, QuestionGenerator
from tests.conftest import make_entry


def test_noop_returns_empty_list() -> None:
    entry = make_entry(content="Pydantic v2 requires strict mode")
    assert NoOpQuestionGenerator().generate(entry) == []


def test_noop_is_runtime_checkable_generator() -> None:
    assert isinstance(NoOpQuestionGenerator(), QuestionGenerator)


def test_arbitrary_object_satisfies_protocol() -> None:
    class _Custom:
        def generate(self, entry: object) -> list[str]:
            return ["q1"]

    assert isinstance(_Custom(), QuestionGenerator)


def test_object_without_generate_is_not_a_generator() -> None:
    class _NoGenerate:
        pass

    assert not isinstance(_NoGenerate(), QuestionGenerator)


def test_client_binds_noop_when_no_generator(memory_client: object) -> None:
    # Default injection: no question_generator → NoOpQuestionGenerator bound.
    gen = memory_client._question_generator  # type: ignore[attr-defined]
    assert isinstance(gen, NoOpQuestionGenerator)


def test_client_binds_injected_generator(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.client import MemoryClient

    class _Inject:
        def generate(self, entry: object) -> list[str]:
            return []

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))  # type: ignore[arg-type]
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    injected = _Inject()
    client = MemoryClient("default", mode="local", question_generator=injected)
    assert client._question_generator is injected


def test_hype_module_pulls_no_llm_runtime() -> None:
    # NG1: importing the HyPE seam in a CLEAN interpreter must not drag in an
    # LLM client library. Run in a subprocess so a pre-imported httpx (used by
    # the sync layer in this test session) doesn't false-positive.
    import subprocess

    code = (
        "import sys, trw_memory.hype\n"
        "forbidden = {'ollama', 'openai', 'anthropic', 'vllm'}\n"
        "leaked = forbidden & set(sys.modules)\n"
        "assert not leaked, leaked\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
