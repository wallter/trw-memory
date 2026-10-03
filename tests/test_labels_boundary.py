"""PRD-SEC-023 NFR04: ``trw_memory.labels`` is a deep module with a narrow interface, and nothing outside it decides a label.

Responsibility lives in one place: no other module in trw-memory reads the ``trw_label`` stamp, and the package imports nothing from trw-mcp.
"""

from __future__ import annotations

import ast
from pathlib import Path

import trw_memory.labels as labels

_SRC = Path(__file__).resolve().parents[1] / "src" / "trw_memory"
_PACKAGE = _SRC / "labels"
_PUBLIC = {"Admission", "LabelPolicy", "Level", "SessionMark", "Sink", "Surface"}


def _effective_loc(source: str) -> int:
    """Lines of code: not blank, not a comment, not inside a docstring (the repo's 350-line gate counts the same way)."""
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_lines.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    return sum(
        1
        for number, line in enumerate(source.splitlines(), start=1)
        if line.strip() and not line.strip().startswith("#") and number not in docstring_lines
    )


def test_the_public_interface_is_exactly_the_documented_one() -> None:
    assert set(labels.__all__) == _PUBLIC
    assert all(hasattr(labels, name) for name in _PUBLIC)


def test_every_file_is_under_the_350_line_gate() -> None:
    sizes = {path.name: _effective_loc(path.read_text(encoding="utf-8")) for path in _PACKAGE.glob("*.py")}
    assert sizes and all(size < 350 for size in sizes.values()), sizes


def test_the_package_docstring_states_responsibility_interface_invariants_and_knobs() -> None:
    doc = labels.__doc__ or ""
    for heading in ("Responsibility", "Interface", "Invariants", "Knobs"):
        assert heading in doc, heading


def test_the_package_imports_nothing_from_trw_mcp() -> None:
    for path in _PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(name.split(".")[0] == "trw_mcp" for name in names), (path.name, names)


def test_no_other_module_reads_the_stamp() -> None:
    """The stamp key is spelled in the labels package only; the rest of trw-memory goes through the policy."""
    offenders = [
        str(path.relative_to(_SRC))
        for path in _SRC.rglob("*.py")
        if _PACKAGE not in path.parents and "trw_label" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
