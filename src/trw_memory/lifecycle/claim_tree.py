"""Which commit a checkout is at, and whether a claim was written at a later one (PRD-CORE-362).

A learning's assertions are checked against the files of one checkout. When that
checkout is *behind* the commit an assertion was written at, a missing file says
nothing about the claim, so the verification pass holds the assertion as unknown
instead of recording a failure. Every other answer evaluates the assertion as it
always has: the rule fails only toward evaluation.

Every git process this change starts is started by :func:`_git`, the only place in
the package that does. The untrusted inputs there are the repository's own
configuration, a stored ``commit_hash`` and the caller's environment, so:

* the process runs with ``cwd`` set to the one root given, never ``-C``,
  ``--git-dir`` or ``--work-tree``;
* every ``GIT_*`` variable of the caller is removed from its environment, and then
  ``GIT_GRAFT_FILE`` is set to the null device: it is the one ``GIT_`` variable this
  module sets itself, so a grafts file cannot invent ancestry;
* it has a timeout, ``core.fsmonitor`` is off, ``--no-replace-objects`` stops replace
  refs from rewriting history, and ``--no-lazy-fetch`` keeps a question about a
  missing object from starting a fetch;
* the argument list is fixed: the only values that vary are two object names that
  matched ``^[0-9a-f]{40}$`` first;
* no shell, output captured as bytes and decoded with ``errors="replace"``, no input,
  so nothing in :func:`_git` raises on what git prints.

A stored id counts only when it is itself a commit object (``<id>^{commit}`` resolves
to the same id): a tag, tree or blob id is evaluated, never held. HEAD is resolved as
a commit too (``HEAD^{commit}``), so a HEAD that names a tag object is read as the
commit it sits on. Parent links come from the commit objects themselves: besides
replacement objects and the grafts file, the commit-graph file is switched off
(``core.commitGraph=false``), because the repository holds it and it could carry
parent links the commits do not have.

Limits. The root is validated when HEAD is looked up and later calls use the same path
as ``cwd``; an actor who can replace the granted directory with another directory or a
link between those calls could steer this sweep's ancestry answers. That is outside
what this module defends against: that actor controls the files the assertions read
anyway. The summed time budget is checked before each process, so the one call in
flight can exceed it: the bound is the budget plus one timeout.

A git that cannot answer (missing, too old for ``--no-lazy-fetch``, timing out,
not a repository, a repository whose top level is not the given root) yields no
answer, and no answer never holds an assertion.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import structlog

from trw_memory.namespaces.worktree import GIT_TIMEOUT_SECONDS

logger = structlog.get_logger(__name__)

__all__ = [
    "CLAIM_TREE_GIT_BUDGET_SECONDS",
    "CLAIM_TREE_MAX_DISTINCT_SHAS",
    "ClaimTree",
    "checkout_head",
    "is_valid_sha",
]

#: Most distinct stamped commits one :class:`ClaimTree` asks git about; past it entries are evaluated.
CLAIM_TREE_MAX_DISTINCT_SHAS = 256
#: Summed wall time of one tree's git processes after which it starts no more.
CLAIM_TREE_GIT_BUDGET_SECONDS = GIT_TIMEOUT_SECONDS

_SHA = re.compile(r"[0-9a-f]{40}")


def is_valid_sha(value: object) -> bool:
    """True only for exactly 40 lowercase hex characters (no newline, no case, no other length)."""
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _git(root: Path, args: tuple[str, ...]) -> tuple[int, str] | None:
    """THE git starter. ``(exit code, stdout)``, or ``None`` when git did not run to an exit
    (missing, a timeout, or any other OS or subprocess error)."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_GRAFT_FILE"] = os.devnull
    argv = [
        "git",
        "--no-lazy-fetch",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.commitGraph=false",  # parent links are read from the commits, not from a graph file the repository holds
        *args,
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, validated object names only
            argv,
            cwd=root,
            env=env,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ) as exc:  # trw-fail-silent-allow: no answer means the assertion is evaluated
        logger.debug("claim_tree_git_unavailable", error=type(exc).__name__)
        return None
    out = completed.stdout
    return completed.returncode, out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)


def _head_of(root: Path) -> tuple[str | None, bool]:
    """``(HEAD's object name or None, git ran to an exit)`` for the checkout whose top level is *root*."""
    # HEAD^{commit}: a HEAD that names a tag object would otherwise differ from the commit it sits on and read as behind it
    answered = _git(root, ("rev-parse", "--show-toplevel", "HEAD^{commit}"))
    if answered is None:
        return None, False
    code, out = answered
    lines = out.splitlines()
    if code != 0 or len(lines) != 2 or lines[0] != str(root) or not is_valid_sha(lines[1]):
        return None, True
    return lines[1], True


def checkout_head(root: Path) -> str | None:
    """HEAD of the checkout whose top level is exactly *root*, or ``None`` for any other state."""
    return _head_of(root)[0]


class ClaimTree:
    """One sweep's view of its checkout: HEAD looked up at most once, one ancestry answer per distinct sha.

    Holds nothing between calls of its owner; build one per sweep (or per daemon call) and drop it.
    """

    def __init__(self, root: Path, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._root = root
        self._clock = clock
        self._head: str | None = None
        self._looked_up = False
        self._broken = False
        self._spent = 0.0
        self._answers: dict[str, bool] = {}

    def _may_start(self) -> bool:
        return not self._broken and self._spent <= CLAIM_TREE_GIT_BUDGET_SECONDS

    def _run(self, args: tuple[str, ...]) -> tuple[int, str] | None:
        started = self._clock()
        answered = _git(self._root, args)
        self._spent += self._clock() - started
        if answered is None:
            self._broken = True
        return answered

    @property
    def head(self) -> str | None:
        """The checkout's HEAD, or ``None``; at most one git process over the tree's life."""
        if not self._looked_up:
            self._looked_up = True
            if self._may_start():
                started = self._clock()
                head, ran = _head_of(self._root)
                self._spent += self._clock() - started
                self._broken = self._broken or not ran
                self._head = head
        return self._head

    def behind(self, sha: object) -> bool:
        """True only when HEAD differs from *sha* and is a strict ancestor of it; false in every other case."""
        if not isinstance(sha, str) or not is_valid_sha(sha):
            return False
        if sha in self._answers:
            return self._answers[sha]
        if len(self._answers) >= CLAIM_TREE_MAX_DISTINCT_SHAS:
            return False
        head = self.head
        answer = False
        if head is not None and head != sha and self._may_start():
            # The id must itself be a commit: a tag object's id peels to a commit in the ancestry check below.
            checked = self._run(("rev-parse", "--quiet", "--verify", f"{sha}^{{commit}}"))
            if checked is None:
                return False  # no answer is not cached as an answer; the tree is broken and asks no more
            if checked[0] == 0 and checked[1].strip() == sha and self._may_start():
                answered = self._run(("merge-base", "--is-ancestor", head, sha))
                if answered is None:
                    return False
                answer = answered[0] == 0
        self._answers[sha] = answer
        return answer
