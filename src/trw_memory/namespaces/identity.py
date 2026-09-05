"""Project namespace identity -- PRD-CORE-253 FR01.

A project's rows in the single user-space store are keyed by a namespace, so
that namespace has to be **stable** across working directories and **distinct**
between two checkouts that look alike. Two identities that were already in the
tree fail one of those tests:

* the basename (``resolve_project_root().name``) is not unique on a filesystem
  -- two checkouts named ``memory`` in different parents collide; and
* the git remote set is not an identity either. Measured 2026-09-03, one of
  seven checkouts on this box had a ``remote.origin.url`` at all, and
  ``trw-framework`` and ``trw-framework-frontend-ux`` carry six remotes each
  with *byte-identical* remote sets -- a remote-derived key merges two distinct
  stores.

So the identity is ``project:<slug>-<digest8>`` over a **canonical root**:

``digest8``
    the first 8 hex characters of
    ``sha256(PROJECT_NAMESPACE_DOMAIN + str(canonical_root))``. The literal
    domain prefix separates this digest from any bare path hash used elsewhere,
    so the two can never be confused for one another.
``slug``
    the canonical root's basename, lowercased, every character outside
    ``[a-z0-9-]`` replaced by a hyphen, runs of hyphens collapsed, truncated to
    :data:`SLUG_MAX_CHARS`. It carries no identity -- it exists so a human
    reading a namespace can tell which checkout it names.

The canonical root is the realpath of the repository's **common** git
directory's parent (``git rev-parse --git-common-dir``), not its top level.
That choice is what makes a linked worktree resolve to the SAME namespace as
the main checkout it belongs to: ``--show-toplevel`` differs per worktree,
``--git-common-dir`` does not. A second *clone* of the same upstream is a
distinct checkout with its own common dir, and therefore a distinct namespace
by design -- the alternative would merge two independently-evolving working
copies into one namespace. Outside a git repository the canonical root is the
resolved project root (``TRW_PROJECT_ROOT`` or the current directory), so a
plain directory resolves without error.

Every path is resolved through ``Path.resolve()`` (realpath), so a symlinked
checkout and its target yield one namespace rather than two.

**When git cannot answer.** ``git rev-parse`` failing to EXECUTE -- absent from
PATH, timing out on a stale network filesystem -- is not the same fact as "this
directory is not a repository", and treating it as one silently re-keys a linked
worktree from its main checkout's root to its own path, splitting one project's
rows across two namespaces. So execution failure falls back to the repository's
own on-disk evidence (``.git`` as a directory, or the ``gitdir:`` pointer file a
linked worktree carries), which yields the SAME canonical root without running
git at all. Only when that evidence is absent too is the identity degraded, and
then :func:`resolve_project_identity` names the reason rather than handing back
an ordinary-looking namespace.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Literal, NamedTuple

import structlog

from trw_memory.namespaces.validation import validate_namespace

__all__ = [
    "DIGEST_CHARS",
    "PROJECT_NAMESPACE_DOMAIN",
    "SLUG_MAX_CHARS",
    "ProjectIdentity",
    "canonical_project_root",
    "project_slug",
    "resolve_project_identity",
    "resolve_project_namespace",
]

logger = structlog.get_logger(__name__)

#: Domain-separation prefix hashed ahead of the canonical root. Changing this
#: string re-keys every project namespace, which is why it carries an explicit
#: ``:v1`` generation marker.
PROJECT_NAMESPACE_DOMAIN = "trw-project-namespace:v1\n"

#: Hex characters of the sha256 digest kept in the namespace. 8 hex characters
#: is 32 bits: with the ~10^2 checkouts a developer machine actually holds, the
#: birthday collision probability is below 10^-5, and a collision is contained
#: because the slug is still there to disambiguate for a human.
DIGEST_CHARS = 8

#: Maximum slug length. The namespace grammar
#: (:func:`trw_memory.namespaces.validation.validate_namespace`) caps a
#: namespace at 128 characters; ``project:`` + 32 + ``-`` + 8 = 49, which
#: leaves the ceiling untouched for even the longest directory name.
SLUG_MAX_CHARS = 32

#: Slug used when a canonical root's basename sanitises to nothing (e.g. a
#: directory named entirely out of characters outside ``[a-z0-9-]``). Without
#: it the namespace would be ``project:-<digest>``, which the grammar rejects.
FALLBACK_SLUG = "root"

#: Seconds to wait for ``git rev-parse``. A hung git (a stale network
#: filesystem, a credential prompt) must degrade to the non-git fallback rather
#: than hang the first store or recall of a session.
_GIT_TIMEOUT_SECONDS = 5.0

#: The directory name git uses for a repository's control directory. When
#: ``--git-common-dir`` names it, the checkout root is its parent; a bare
#: repository's common dir IS the repository, so it is used as-is.
_GIT_DIR_NAME = ".git"

#: Prefix of the one-line ``.git`` FILE a linked worktree (and a submodule)
#: carries in place of a control directory. Its target is
#: ``<common-dir>/worktrees/<name>``, so the common dir is two levels up -- the
#: same answer ``git rev-parse --git-common-dir`` gives, read without git.
_GITDIR_PREFIX = "gitdir:"

#: Path component git interposes between a common dir and each linked worktree's
#: control directory. Used to walk a ``gitdir:`` pointer back to the common dir.
_WORKTREES_DIR_NAME = "worktrees"

#: How far up the tree the filesystem fallback looks for ``.git``. Bounded so a
#: resolution started deep inside a non-repository cannot walk to ``/``.
_DOTGIT_SEARCH_PARENTS = 64

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9-]+")
_HYPHEN_RUNS = re.compile(r"-{2,}")

#: How the canonical root was established. ``git`` is the ordinary answer,
#: ``filesystem`` is the on-disk fallback used when git could not execute, and
#: ``path`` means neither could answer -- either an ordinary non-repository
#: directory or a repository whose identity evidence was unreadable.
IdentitySource = Literal["git", "filesystem", "path"]


class ProjectIdentity(NamedTuple):
    """A project namespace plus how much of it could actually be established.

    ``degraded`` is ``None`` on every ordinary resolution, INCLUDING a plain
    non-git directory: that is a real answer, not a failure. It names a reason
    only when git was supposed to be able to answer and could not, and the
    on-disk evidence could not stand in -- the case where the namespace may have
    silently re-keyed away from the checkout this project's rows are under.
    """

    namespace: str
    canonical_root: Path
    source: IdentitySource
    degraded: str | None


class _CommonDirProbe(NamedTuple):
    """Tri-state result of asking git for a repository's common directory."""

    path: Path | None
    #: ``resolved``, ``not_a_repository`` (a real answer), or ``unavailable``
    #: (git could not be run or did not finish -- NOT an answer).
    status: Literal["resolved", "not_a_repository", "unavailable"]
    reason: str | None = None


def _resolved_start(start: Path | str | None) -> Path:
    """Return the directory identity resolution starts from, as a realpath.

    Mirrors ``trw_mcp.state._paths.resolve_project_root`` (``TRW_PROJECT_ROOT``
    then the current directory). It is re-stated here rather than imported
    because trw-mcp depends on trw-memory and not the reverse; the daemon must
    resolve an identity with trw-mcp absent.
    """
    if start is not None:
        return Path(start).resolve()
    env_root = os.environ.get("TRW_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path.cwd().resolve()


def canonical_project_root(start: Path | str | None = None) -> Path:
    """Resolve the canonical root that identifies *start*'s project.

    Args:
        start: Directory to resolve from. Defaults to ``TRW_PROJECT_ROOT`` or
            the current working directory.

    Returns:
        The realpath of the repository's common git directory's parent when
        *start* is inside a git repository (so every linked worktree of one
        repository shares one root), otherwise the realpath of *start* itself.

    See :func:`resolve_project_identity` when the caller needs to know whether
    that root was actually established or only fallen back to.
    """
    return _canonical_root(_resolved_start(start))[0]


def _canonical_root(root: Path) -> tuple[Path, IdentitySource, str | None]:
    """Return *root*'s canonical root, how it was established, and any degradation."""
    probe = _git_common_dir(root)
    if probe.path is not None:
        return _root_of_common_dir(probe.path), "git", None
    if probe.status == "not_a_repository":
        return root, "path", None
    # git could not answer. Its answer is derivable from the repository's own
    # on-disk evidence, so read that before conceding: conceding here is what
    # re-keys a linked worktree onto its own path.
    from_disk = _common_dir_from_disk(root)
    if from_disk.path is not None:
        logger.warning("project_identity_git_unavailable", path=str(root), reason=probe.reason, recovered="filesystem")
        return _root_of_common_dir(from_disk.path), "filesystem", None
    if from_disk.status == "not_a_repository":
        # No ``.git`` anywhere up to the filesystem root: the tree carries no
        # repository evidence, which is the same conclusion git would have
        # reached. The path identity is the answer, not a degradation.
        return root, "path", None
    reason = from_disk.reason or probe.reason or "git_unavailable"
    logger.warning("project_identity_degraded", path=str(root), reason=reason)
    return root, "path", reason


def _root_of_common_dir(common_dir: Path) -> Path:
    """A common dir named ``.git`` identifies its parent; a bare repo is itself."""
    return common_dir.parent if common_dir.name == _GIT_DIR_NAME else common_dir


def _git_common_dir(root: Path) -> _CommonDirProbe:
    """Ask git for *root*'s common git directory, distinguishing "no" from "cannot".

    A non-zero exit is git ANSWERING that *root* is not in a repository. A raised
    ``OSError``/``SubprocessError`` (git absent, timed out, killed) is git failing
    to answer at all. Collapsing the second into the first is what let a broken or
    hung git silently re-key a worktree, so they are returned as distinct states.
    """
    if not root.is_dir():
        return _CommonDirProbe(None, "not_a_repository")
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],  # noqa: S607 - resolved via PATH by design
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _CommonDirProbe(None, "unavailable", type(exc).__name__)
    if completed.returncode != 0:
        return _CommonDirProbe(None, "not_a_repository")
    raw = completed.stdout.strip()
    if not raw:
        # git exited 0 without naming a directory: it did not answer either.
        return _CommonDirProbe(None, "unavailable", "empty_output")
    # git answers relatively (".git", "../.git") when the answer is inside the
    # directory it was asked from, so it is joined onto *root* before resolving.
    return _CommonDirProbe((root / raw).resolve(), "resolved")


def _common_dir_from_disk(root: Path) -> _CommonDirProbe:
    """Derive the common git directory from on-disk evidence, without running git.

    Mirrors what ``--git-common-dir`` reports: a ``.git`` DIRECTORY is the common
    dir itself; a ``.git`` FILE holds ``gitdir: <common>/worktrees/<name>``, whose
    common dir is two components up.

    Tri-state like :func:`_git_common_dir`, and for the same reason. Reaching the
    filesystem root without finding ``.git`` is EVIDENCE that this is not a
    repository -- the conclusion git would have reached -- so it is an answer.
    Finding ``.git`` and failing to read it is not.
    """
    parents = list(root.parents)
    for candidate in [root, *parents[:_DOTGIT_SEARCH_PARENTS]]:
        dot_git = candidate / _GIT_DIR_NAME
        if dot_git.is_dir():
            return _CommonDirProbe(dot_git.resolve(), "resolved")
        if dot_git.is_file():
            pointed = _common_dir_from_pointer(dot_git)
            if pointed is None:
                return _CommonDirProbe(None, "unavailable", "unreadable_gitdir_pointer")
            return _CommonDirProbe(pointed, "resolved")
    if len(parents) > _DOTGIT_SEARCH_PARENTS:
        return _CommonDirProbe(None, "unavailable", "gitdir_search_bounded")
    return _CommonDirProbe(None, "not_a_repository")


def _common_dir_from_pointer(dot_git: Path) -> Path | None:
    """Read a ``gitdir:`` pointer file and walk it back to the common dir."""
    try:
        raw = dot_git.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not raw.startswith(_GITDIR_PREFIX):
        return None
    target = raw[len(_GITDIR_PREFIX) :].strip()
    if not target:
        return None
    control_dir = (dot_git.parent / target).resolve()
    # <common>/worktrees/<name> -> <common>. A submodule's pointer has no
    # ``worktrees`` component, and its control dir IS its common dir.
    if control_dir.parent.name == _WORKTREES_DIR_NAME:
        return control_dir.parent.parent
    return control_dir


def project_slug(root: Path | str) -> str:
    """Return the human-readable, non-identifying slug for *root*."""
    name = Path(root).name.lower()
    slug = _HYPHEN_RUNS.sub("-", _NON_SLUG_CHARS.sub("-", name)).strip("-")
    slug = slug[:SLUG_MAX_CHARS].strip("-")
    return slug or FALLBACK_SLUG


def resolve_project_identity(start: Path | str | None = None) -> ProjectIdentity:
    """Resolve *start*'s project namespace together with how it was established.

    Args:
        start: Directory to resolve from. Defaults to ``TRW_PROJECT_ROOT`` or
            the current working directory.

    Returns:
        A :class:`ProjectIdentity`. ``degraded`` is ``None`` whenever the
        canonical root is the one this module promises -- from git, or read from
        the repository's own on-disk evidence when git could not run. It names a
        reason only when neither could answer, which is the case where the
        namespace may differ from the one the project's existing rows are under.

    Raises:
        ConfigError: If the derived value violates the namespace grammar --
            which would be a defect in this module, not caller input, and is
            surfaced rather than silently repaired.
    """
    root, source, degraded = _canonical_root(_resolved_start(start))
    digest = hashlib.sha256((PROJECT_NAMESPACE_DOMAIN + str(root)).encode("utf-8")).hexdigest()[:DIGEST_CHARS]
    namespace = validate_namespace(f"project:{project_slug(root)}-{digest}")
    return ProjectIdentity(namespace, root, source, degraded)


def resolve_project_namespace(start: Path | str | None = None) -> str:
    """Resolve the project namespace for *start*.

    Args:
        start: Directory to resolve from. Defaults to ``TRW_PROJECT_ROOT`` or
            the current working directory.

    Returns:
        ``project:<slug>-<digest8>``, already validated against the namespace
        grammar so no caller has to re-check it. A caller that must not act on a
        possibly re-keyed identity should call :func:`resolve_project_identity`
        and read ``degraded``.

    Raises:
        ConfigError: If the derived value violates the namespace grammar --
            which would be a defect in this module, not caller input, and is
            surfaced rather than silently repaired.
    """
    return resolve_project_identity(start).namespace
