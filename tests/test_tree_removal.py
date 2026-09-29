"""PRD-FIX-156 FR01 (B71-10): a cleanup removal never raises and is never silent.

The contract of :func:`trw_memory._tree_removal.remove_tree`, on real files, and the census that keeps
every other ``shutil.rmtree`` in ``trw_memory`` from carrying its own error handler again.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

import trw_memory
from trw_memory._tree_removal import remove_tree

_HANDLER_KEYWORDS = frozenset({"ignore_errors", "onerror", "onexc"})


@pytest.fixture
def events() -> Iterator[list[dict[str, object]]]:
    with structlog.testing.capture_logs() as captured:
        yield captured


def _failures(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [e for e in events if e["event"] == "tree_removal_failed"]


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions a non-root user obeys")
def test_a_failed_removal_is_logged_and_the_rest_of_the_tree_still_goes(
    tmp_path: Path, events: list[dict[str, object]]
) -> None:
    root = tmp_path / "scratch"
    (root / "locked").mkdir(parents=True)
    (root / "locked" / "kept").write_text("x")
    (root / "free").mkdir()
    (root / "free" / "gone").write_text("x")
    (root / "locked").chmod(0o500)  # its entries cannot be unlinked
    try:
        remove_tree(root, purpose="test scratch")
    finally:
        (root / "locked").chmod(0o700)

    assert not (root / "free").exists(), "one failure does not stop the rest of the tree"
    assert (root / "locked" / "kept").exists()
    [failure] = _failures(events)
    assert failure["log_level"] == "warning"
    assert failure["path"] == str(root)
    assert failure["purpose"] == "test scratch"
    assert isinstance(failure["failures"], int) and failure["failures"] >= 1
    assert "kept" in str(failure["first_error"])


def test_a_symlink_is_unlinked_and_its_target_survives(tmp_path: Path, events: list[dict[str, object]]) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("x")
    link = tmp_path / "link"
    link.symlink_to(outside)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "inner").symlink_to(outside)

    remove_tree(link, purpose="leaf link")
    remove_tree(tree, purpose="tree with a link inside")

    assert not link.is_symlink() and not tree.exists()
    assert (outside / "keep").read_text() == "x"
    assert _failures(events) == []


def test_a_missing_path_is_a_quiet_no_op(tmp_path: Path, events: list[dict[str, object]]) -> None:
    remove_tree(tmp_path / "never-created", purpose="missing")

    assert _failures(events) == []


def _handler_rmtree_calls() -> list[str]:
    """Every ``rmtree(...)`` passing an error handler, as ``path: call source`` (content, not line)."""
    root = Path(trw_memory.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "_tree_removal.py" and path.parent == root:
            continue  # the helper itself
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name == "rmtree" and (len(node.args) > 1 or {k.arg for k in node.keywords} & _HANDLER_KEYWORDS):
                found.append(f"{path.relative_to(root)}: {ast.unparse(node)}")
    return found


def test_no_rmtree_outside_the_helper_carries_its_own_error_handler() -> None:
    """B71-10 census: ``ignore_errors``/``onerror``/``onexc`` hide or reinvent failure handling; use remove_tree."""
    assert _handler_rmtree_calls() == []
