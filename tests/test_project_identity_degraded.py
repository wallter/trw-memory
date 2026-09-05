"""W16 — a git that cannot run must not silently re-key a worktree's namespace.

``_git_common_dir`` returned ``None`` for two unrelated facts: "git says this is
not a repository" and "git could not be executed at all". The second then took
the non-git fallback, which keys a LINKED WORKTREE on its own path instead of on
the main checkout's common dir -- so one project's rows split across two
namespaces the moment git was absent, hung, or killed, with nothing in the
returned namespace to say so.

Every test here injects the fault on the real path by emptying ``PATH`` so the
``git`` subprocess raises ``FileNotFoundError``, exactly as it would on a box
without git. No function under test is mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.namespaces.identity import resolve_project_identity

pytestmark = pytest.mark.integration


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make the ``git`` executable unfindable, so ``subprocess.run`` raises."""
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))


def _linked_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Build a main checkout and a linked worktree pointing back into its ``.git``."""
    main = tmp_path / "checkout"
    control = main / ".git"
    (control / "worktrees" / "feature").mkdir(parents=True)
    worktree = tmp_path / "feature-worktree"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {control / 'worktrees' / 'feature'}\n", encoding="utf-8")
    return main, worktree


def test_worktree_keeps_the_main_checkouts_namespace_without_git(no_git: None, tmp_path: Path) -> None:
    """The failure is recovered from on-disk evidence, not conceded to."""
    main, worktree = _linked_worktree(tmp_path)

    from_worktree = resolve_project_identity(worktree)
    from_main = resolve_project_identity(main)

    assert from_worktree.namespace == from_main.namespace
    assert from_worktree.canonical_root == main.resolve()
    assert from_worktree.source == "filesystem"
    # Recovered means recovered: nothing here is degraded.
    assert from_worktree.degraded is None


def test_unreadable_repository_evidence_is_reported_as_degraded(no_git: None, tmp_path: Path) -> None:
    """A ``.git`` that exists but cannot be interpreted is named, not papered over."""
    checkout = tmp_path / "broken"
    checkout.mkdir()
    (checkout / ".git").write_text("this is not a gitdir pointer\n", encoding="utf-8")

    identity = resolve_project_identity(checkout)

    assert identity.degraded == "unreadable_gitdir_pointer"
    assert identity.source == "path"
    # It still answers -- a degraded identity is reported, not raised, so a
    # store or recall can proceed while the caller can see it is provisional.
    assert identity.namespace.startswith("project:broken-")


def test_a_plain_directory_is_not_degraded_without_git(no_git: None, tmp_path: Path) -> None:
    """No ``.git`` anywhere up is the same conclusion git would have reached."""
    plain = tmp_path / "notes"
    plain.mkdir()

    identity = resolve_project_identity(plain)

    assert identity.degraded is None
    assert identity.source == "path"
    assert identity.canonical_root == plain.resolve()


def test_a_real_repository_resolves_through_git(tmp_path: Path) -> None:
    """With git available the ordinary path is unchanged and undegraded."""
    identity = resolve_project_identity(Path(__file__).parent)

    assert identity.source == "git"
    assert identity.degraded is None
