"""A linked git worktree borrows its main checkout's memory pin and grant.

Responsibility. A linked worktree (``git worktree add``) has its own untracked
``.trw``, so it carries neither the ``project_namespace`` pin nor the
``.trw/runtime/memory-token`` grant that ``memory migrate`` / ``memory token``
wrote into the main checkout. The canonical identity (:mod:`.identity`) already
maps a worktree to its main checkout's namespace, so every consumer that opens
the store for a checkout -- trw-mcp's ``selected_store`` (recall) and
trw-distill's ``open_daemon_store`` (live ingest) -- asks this module for the
main checkout when the worktree holds no pin of its own. It lives here, not in
trw-mcp, because trw-distill must not import trw-mcp.

Interface. :func:`main_checkout_binding` takes the directory and the caller's
pin reader (each consumer reads the pin its own way) and returns the main
checkout's ``.trw`` directory (its grant is the one presented) and its pin, or
``None`` when the directory is not a linked worktree, or its main checkout is
unpinned too.

Invariants.

* Only a genuine linked worktree of the SAME repository borrows. git decides,
  never a parent-directory walk: the directory must be the worktree's top
  level, its git dir must differ from the common dir and sit under
  ``<common>/worktrees/``, and the main checkout's ``.git`` must be that common
  dir. A separate clone, a submodule and a plain subdirectory of the main
  checkout all have their own top level or git dir == common dir, so none borrow.
* The link must be REGISTERED both ways, as git itself keeps it: the
  directory's ``.git`` file points at ``<common>/worktrees/<name>``, and that
  admin dir's ``gitdir`` file names THIS directory's ``.git`` (realpaths). A
  copied or crafted ``.git`` pointer, or a worktree moved without
  ``git worktree repair``, is refused. The directory is realpath'd first, so a
  symlink to a registered worktree borrows exactly as the worktree does.
* The borrowed pin must EQUAL the worktree's canonical namespace
  (``resolve_project_identity``), established without degradation. A mismatch
  (a moved main checkout that kept an older pin, say) fails closed naming both.
* The pin and grant come ONLY from that main worktree's ``.trw``; a worktree with
  a pin of its own never reaches here (each caller keeps its own pin and
  grant), so the two are never merged.
* Read-only: nothing is written into either checkout.
* Out of scope: an attacker who can write inside ``<common>/worktrees`` (or
  anywhere in the main checkout's ``.git``) can register any directory as a
  worktree, as ``git worktree add`` does; that access already reaches the
  main checkout's ``.trw`` and grant directly.
* git that cannot run raises :class:`GitUnavailableError`; the caller keeps
  today's fail-closed refusal and names the fix. git answering "not a
  repository" (or any non-zero exit) is ``None``: no borrowing. Every later
  failure on the path -- an unreadable pointer or registration, a canonical
  identity git could not establish itself -- raises :class:`WorktreeRefusedError`.
* Inherited ``GIT_*`` variables (a hook's ``GIT_DIR``) are dropped, so git
  answers about the directory asked, not the repository the caller runs in.

Knobs: none. :data:`GIT_TIMEOUT_SECONDS` bounds the one ``git rev-parse``.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)

#: Seconds the one ``git rev-parse`` may take. A hung git (stale network filesystem)
#: must fail closed promptly rather than hold a pre-edit hint; the identity resolver
#: uses the same bound.
GIT_TIMEOUT_SECONDS = 5.0

_DOT_GIT = ".git"
_GITDIR_PREFIX = "gitdir:"
#: The admin dir's back-pointer: the absolute path of the worktree's ``.git`` file.
_GITDIR_FILE = "gitdir"
_WORKTREES = "worktrees"
_TRW = ".trw"


class GitUnavailableError(RuntimeError):
    """git could not run, so whether this is a linked worktree is unknown."""


class WorktreeRefusedError(RuntimeError):
    """The link, its registration or its namespace could not be proven; nothing is borrowed."""


class MainCheckoutBinding(NamedTuple):
    """The main checkout a linked worktree borrows from: its ``.trw`` and its pin."""

    trw_dir: Path
    namespace: str


def main_checkout_binding(root: Path, read_pin: Callable[[Path], str | None]) -> MainCheckoutBinding | None:
    """The main checkout's ``.trw`` and pin when *root* is a linked worktree of it; else ``None``.

    *read_pin* receives the main checkout's ``.trw`` directory and returns its pin (or
    ``None``); it is called only after the link and its registration are proven.
    Raises :class:`GitUnavailableError` when git cannot run, :class:`WorktreeRefusedError`
    when the registration or the canonical namespace cannot be proven to match, and
    whatever *read_pin* raises when the main checkout's config is unreadable.
    """
    try:
        root = root.resolve()
        main_root = _linked_main_root(root)
    except (GitUnavailableError, WorktreeRefusedError):
        raise
    except (OSError, RuntimeError) as exc:  # RuntimeError: a symlink loop in resolve()
        raise WorktreeRefusedError(
            f"{root}'s git worktree link could not be read ({type(exc).__name__}: {exc}); nothing is borrowed. "
            f"Fix the checkout's .git, or pin it: run `trw-mcp update-project` there"
        ) from exc
    if main_root is None:
        return None
    main_trw = main_root / _TRW
    pinned = read_pin(main_trw)
    if not pinned:
        return None
    from trw_memory.namespaces.identity import resolve_project_identity

    identity = resolve_project_identity(root)
    # source "git" only: a canonical root read from disk because git failed this time proves nothing.
    if identity.source != "git" or identity.canonical_root != main_root or identity.namespace != pinned:
        raise WorktreeRefusedError(
            f"{root} is a linked worktree of {main_root}, whose project_namespace is {pinned!r}, but this "
            f"repository's canonical namespace is {identity.namespace!r} (established by: {identity.source}); "
            f"nothing is borrowed. To route this worktree to {pinned!r}, set `project_namespace: {pinned}` in "
            f"{root / _TRW / 'config.yaml'} and run `trw-mcp memory token` there"
        )
    logger.debug("worktree_memory_binding_borrowed", worktree=str(root), main=str(main_root), namespace=pinned)
    return MainCheckoutBinding(main_trw, pinned)


def _linked_main_root(root: Path) -> Path | None:
    """*root*'s main checkout when *root* is the top level of a linked worktree of it."""
    answer = _rev_parse(root)
    if answer is None:
        return None
    toplevel, git_dir, common_dir = answer
    if toplevel != root or git_dir == common_dir:
        return None  # a subdirectory, a main checkout, a clone or a submodule: never borrows
    if git_dir.parent != common_dir / _WORKTREES or common_dir.name != _DOT_GIT:
        return None  # not git's linked-worktree layout, or a bare repository with no main checkout
    main_root = common_dir.parent
    main_dot_git = main_root / _DOT_GIT
    # The real main worktree: a .git DIRECTORY (not a pointer file, not a symlink) that IS the common dir.
    if main_dot_git.is_symlink() or not main_dot_git.is_dir() or main_dot_git.resolve() != common_dir:
        return None
    _require_registered(root, git_dir)
    return main_root


def _require_registered(root: Path, git_dir: Path) -> None:
    """Refuse unless *root*'s ``.git`` points at *git_dir* AND *git_dir*'s ``gitdir`` names *root*'s ``.git``.

    git follows a ``.git`` pointer without checking the admin dir points back, so a copied
    pointer (or a worktree moved without ``git worktree repair``) passes ``rev-parse``.
    Both are compared as realpaths, and neither file may be a symlink: a symlinked ``.git``
    in another directory would otherwise resolve onto the registered one. Both files are
    git's own; OSError propagates and the caller refuses.
    """
    dot_git = root / _DOT_GIT  # *root* is already a realpath, so this is the file's own realpath
    registration = git_dir / _GITDIR_FILE
    if dot_git.is_symlink() or registration.is_symlink():
        raise WorktreeRefusedError(
            f"{root}'s .git or its registration {registration} is a symlink; nothing is borrowed. "
            f"Recreate the worktree with `git worktree add`"
        )
    pointer = dot_git.read_text(encoding="utf-8").strip() if dot_git.is_file() else ""
    points_at = (
        (root / pointer.removeprefix(_GITDIR_PREFIX).strip()).resolve() if pointer.startswith(_GITDIR_PREFIX) else None
    )
    registered = registration.read_text(encoding="utf-8").strip()
    # git writes it absolute; with worktree.useRelativePaths it is relative to the admin dir.
    registered_at = (git_dir / registered).resolve() if registered else None
    if points_at != git_dir or registered_at != dot_git:
        raise WorktreeRefusedError(
            f"{root} is not registered as the worktree {git_dir.name!r} of {git_dir.parent.parent.parent}: "
            f"its .git points at {points_at}, and git records that worktree at {registered_at}. Nothing is "
            f"borrowed. If this worktree was moved, run `git worktree repair {root}` from the main checkout"
        )


def _rev_parse(root: Path) -> tuple[Path, Path, Path] | None:
    """git's top level, git dir and common dir for *root*, realpaths; ``None`` outside a repository."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "--git-dir", "--git-common-dir"],  # noqa: S607 - PATH git, as identity
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitUnavailableError(f"git could not run in {root} ({type(exc).__name__}: {exc})") from exc
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(lines) != 3:
        return None
    # git answers the git dirs relative to *root* when they sit beneath it.
    toplevel, git_dir, common_dir = ((root / line.strip()).resolve() for line in lines)
    return toplevel, git_dir, common_dir
