"""trw_memory.namespaces.worktree: a linked worktree borrows its main checkout's pin only when git registers it.

Real ``git init`` / ``git worktree add``; nothing about the binding is mocked. The
pin reader is the caller's (trw-mcp and trw-distill each read the pin their own
way), so these tests pass a recording one.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from trw_memory.namespaces.identity import resolve_project_namespace
from trw_memory.namespaces.worktree import WorktreeRefusedError, main_checkout_binding


def _git(cwd: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, Path]:
    main = (tmp_path / "main").resolve()
    main.mkdir()
    _git(main, "init", "-q")
    (main / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(main, "add", "a.py")
    _git(main, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "a")
    worktree = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", str(worktree))
    return main, worktree


def test_a_registered_worktree_borrows_the_main_pin(repo: tuple[Path, Path]) -> None:
    main, worktree = repo
    pin = resolve_project_namespace(main)
    asked: list[Path] = []

    binding = main_checkout_binding(worktree, lambda trw: asked.append(trw) or pin)

    assert binding is not None and binding.namespace == pin and binding.trw_dir == main / ".trw"
    assert asked == [main / ".trw"]


def test_the_main_checkout_itself_never_borrows(repo: tuple[Path, Path]) -> None:
    main, _ = repo
    assert main_checkout_binding(main, lambda _t: pytest.fail("the pin is read only after the link is proven")) is None


def test_a_copied_git_pointer_is_refused_before_any_pin_is_read(repo: tuple[Path, Path], tmp_path: Path) -> None:
    _, worktree = repo
    forged = tmp_path / "forged"
    forged.mkdir()
    (forged / ".git").write_text((worktree / ".git").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(WorktreeRefusedError, match="not registered"):
        main_checkout_binding(forged, lambda _t: pytest.fail("a forged pointer must not reach the pin"))


def test_a_pin_other_than_the_canonical_namespace_is_refused(repo: tuple[Path, Path]) -> None:
    _, worktree = repo
    with pytest.raises(WorktreeRefusedError, match="canonical namespace"):
        main_checkout_binding(worktree, lambda _t: "project:someone-else-00000000")


def test_an_unpinned_main_checkout_lends_nothing(repo: tuple[Path, Path]) -> None:
    _, worktree = repo
    assert main_checkout_binding(worktree, lambda _t: None) is None
