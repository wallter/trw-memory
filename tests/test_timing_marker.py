"""PRD-QUAL-141 FR01/FR04 for this package: one host-resource marker, and nothing deterministic behind it.

``requires_local_timing`` tests are skipped on CI runners (``tests/_timing.py``), so a bare ``assert``
inside one would leave the gating suite. The monorepo's trw-mcp lint checks this package too; this
file keeps the public trw-memory mirror checking itself.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._timing import MARKER, apply_timing_policy

_TESTS = Path(__file__).resolve().parent


def _marked_tests_with_bare_asserts(source: str) -> list[str]:
    tree = ast.parse(source)
    module_marked = any(
        isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == "pytestmark" for t in n.targets)
        and MARKER in ast.unparse(n.value)
        for n in tree.body
    )
    found: list[str] = []

    def visit(nodes: list[ast.stmt], prefix: str, marked: bool) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(
                    node.body,
                    f"{prefix}{node.name}::",
                    marked or any(MARKER in ast.unparse(d) for d in node.decorator_list),
                )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
                is_marked = marked or any(MARKER in ast.unparse(d) for d in node.decorator_list)
                if is_marked and any(isinstance(n, ast.Assert) for n in ast.walk(node)):
                    found.append(prefix + node.name)

    visit(tree.body, "", module_marked)
    return found


def test_marker_is_registered_and_perf_is_gone() -> None:
    pyproject = (_TESTS.parent / "pyproject.toml").read_text(encoding="utf-8")
    block = pyproject.split("markers = [", 1)[1].split("]", 1)[0]
    markers = [line.strip().strip('",').split(":", 1)[0] for line in block.splitlines() if line.strip().startswith('"')]
    assert MARKER in markers
    assert "perf" not in markers
    assert [f.name for f in _TESTS.rglob("*.py") if re.search(r"mark\.perf\b", f.read_text(encoding="utf-8"))] == []


def test_marked_tests_assert_only_through_assert_budget() -> None:
    offenders = [
        f"{f.relative_to(_TESTS)}::{t}"
        for f in sorted(_TESTS.rglob("test_*.py"))
        for t in _marked_tests_with_bare_asserts(f.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_the_check_catches_a_bare_assert_in_a_marked_test() -> None:
    source = "import pytest\n@pytest.mark.requires_local_timing\ndef test_x():\n    assert rows == 3\n"
    assert _marked_tests_with_bare_asserts(source) == ["test_x"]


@pytest.mark.parametrize(("env", "skipped"), [({}, False), ({"CI": "true"}, True), ({"GITHUB_ACTIONS": "true"}, True)])
def test_policy_skips_marked_items_only_on_a_ci_runner(
    env: dict[str, str], skipped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    added: list[object] = []
    item = SimpleNamespace(
        get_closest_marker=lambda name: object() if name == MARKER else None, add_marker=added.append
    )

    apply_timing_policy([item])  # type: ignore[list-item]

    assert bool(added) is skipped
