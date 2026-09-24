"""PRD-CORE-293: the retired reward fields appear nowhere in trw_memory's code.

Comments and docstrings may still explain the history; identifiers, attribute
names, keyword arguments and every other string literal (SQL, dict keys, field
descriptions) may not name them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import trw_memory

RETIRED = re.compile(r"\b(q_value|q_observations|helpful_count|unhelpful_count)\b")


def _docstring_ids(tree: ast.AST) -> set[int]:
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }


def _names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.keyword):
        return [node.arg or ""]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.alias):
        return [node.name, node.asname or ""]
    return []


def test_no_retired_reward_field_in_code() -> None:
    root = Path(trw_memory.__file__).parent
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_ids(tree)
        for node in ast.walk(tree):
            texts = _names(node)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                texts.append(node.value)
            hits.extend(
                f"{path.relative_to(root)}:{getattr(node, 'lineno', '?')}" for text in texts if RETIRED.search(text)
            )
    assert hits == []
