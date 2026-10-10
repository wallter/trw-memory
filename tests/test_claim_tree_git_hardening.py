"""PRD-CORE-362 NFR01, NFR04, FR08: every git process the claim tree starts is hardened, bound to one root,
and bounded in number and time.

``subprocess.run`` is stubbed where the call itself is the subject (arguments, environment, counts, clocks);
real repositories in ``tmp_path`` where git's own answer is.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from trw_memory.lifecycle import claim_tree as claim_tree_module
from trw_memory.lifecycle import verification_pass
from trw_memory.lifecycle.claim_tree import ClaimTree, checkout_head
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry
from trw_memory.namespaces.worktree import GIT_TIMEOUT_SECONDS
from trw_memory.storage.sqlite_backend import SQLiteBackend

HEAD = "a" * 40
CLAIM = "b" * 40

_IDENT = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


class FakeGit:
    """A ``subprocess.run`` that records every call and answers like a healthy git for ``root``."""

    def __init__(self, root: Path, *, ancestry_code: int = 0, on_call: Callable[[], None] | None = None) -> None:
        self.root = root
        self.ancestry_code = ancestry_code
        self.on_call = on_call
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((argv, kwargs))
        if self.on_call is not None:
            self.on_call()
        if "--verify" in argv:
            return subprocess.CompletedProcess(argv, 0, argv[-1].removesuffix("^{commit}").encode() + b"\n", b"")
        if "rev-parse" in argv:
            return subprocess.CompletedProcess(argv, 0, f"{self.root}\n{HEAD}\n".encode(), b"")
        return subprocess.CompletedProcess(argv, self.ancestry_code, b"", b"")


def _install(monkeypatch: pytest.MonkeyPatch, fake: Callable[..., Any]) -> None:
    monkeypatch.setattr(subprocess, "run", fake)


def _real_git(root: Path, *args: str) -> str:
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")}, **_IDENT}
    done = subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _repo(path: Path, text: str = "x = 1\n") -> tuple[Path, str]:
    path.mkdir()
    root = path.resolve()
    _real_git(root, "init", "-q")
    (root / "f.py").write_text(text, encoding="utf-8")
    _real_git(root, "add", "f.py")
    _real_git(root, "commit", "-qm", "one")
    return root, _real_git(root, "rev-parse", "HEAD")


# NFR01 ----------------------------------------------------------------------------------------------------


def test_git_runs_only_in_the_given_root_with_the_git_environment_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    fake = FakeGit(tmp_path)
    _install(monkeypatch, fake)

    assert ClaimTree(tmp_path).behind(CLAIM) is True
    assert checkout_head(tmp_path) == HEAD

    assert len(fake.calls) == 4, "HEAD lookup, commit check and ancestry check for the tree; one more for checkout_head"
    for argv, kwargs in fake.calls:
        assert argv[0] == "git" and argv[1:3] == ["--no-lazy-fetch", "--no-replace-objects"]
        assert argv[3:7] == ["-c", "core.fsmonitor=false", "-c", "core.commitGraph=false"]
        assert kwargs["env"]["GIT_GRAFT_FILE"] == os.devnull
        assert not any(k.startswith("GIT_") and k != "GIT_GRAFT_FILE" for k in kwargs["env"])
        for forbidden in ("-C", "--git-dir", "--work-tree"):
            assert forbidden not in argv
        assert kwargs["cwd"] == tmp_path
        assert kwargs["timeout"] == GIT_TIMEOUT_SECONDS
        assert not kwargs.get("shell")
        assert kwargs["capture_output"] is True
    assert fake.calls[1][0][-4:] == ["rev-parse", "--quiet", "--verify", f"{CLAIM}^{{commit}}"]
    assert fake.calls[2][0][-4:] == ["merge-base", "--is-ancestor", HEAD, CLAIM]


@pytest.mark.parametrize(
    "stored",
    [
        "--upload-pack=x",
        "HEAD",
        "a" * 39,
        "a" * 41,
        "a" * 64,
        ("a" * 39 + "A"),
        "a" * 40 + "\n",
        "",
        "-" * 40,
        None,
    ],
)
def test_a_stored_sha_that_is_not_forty_lowercase_hex_never_reaches_git(
    stored: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeGit(tmp_path)
    _install(monkeypatch, fake)

    assert ClaimTree(tmp_path).behind(stored) is False

    assert fake.calls == [], "not even the HEAD lookup starts for a value that is not a valid sha"


def test_a_root_inside_a_parent_repository_is_not_treated_as_a_checkout(tmp_path: Path) -> None:
    parent, head = _repo(tmp_path / "parent")
    below = parent / "sub"
    below.mkdir()
    (below / "g.py").write_text("y = 1\n", encoding="utf-8")
    _real_git(parent, "add", "sub/g.py")
    _real_git(parent, "commit", "-qm", "two")
    tip = _real_git(parent, "rev-parse", "HEAD")

    assert checkout_head(parent) == tip
    assert checkout_head(below) is None
    assert ClaimTree(below).behind(head) is False
    assert ClaimTree(below).head is None


def test_a_git_environment_pointing_elsewhere_does_not_redirect_the_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mine, my_head = _repo(tmp_path / "mine")
    other, other_head = _repo(tmp_path / "other", "different = 2\n")
    assert my_head != other_head
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))

    assert checkout_head(mine) == my_head
    assert ClaimTree(mine).head == my_head


def test_the_claim_tree_module_is_the_only_process_starter_in_lifecycle() -> None:
    lifecycle = Path(claim_tree_module.__file__).parent
    starters = re.compile(
        r"\bsubprocess\b|\bPopen\b|os\.system|os\.popen|os\.exec|os\.spawn|os\.posix_spawn|pty\.|create_subprocess"
    )
    offenders = [
        path.name
        for path in sorted(lifecycle.glob("*.py"))
        if path.name != "claim_tree.py" and starters.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []
    assert starters.search(Path(claim_tree_module.__file__).read_text(encoding="utf-8"))


# NFR04 ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("git", 5.0), FileNotFoundError("git")])
def test_a_hung_git_stops_further_git_calls(
    failure: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def hung(argv: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        raise failure

    _install(monkeypatch, hung)
    tree = ClaimTree(tmp_path)

    assert [tree.behind(CLAIM), tree.behind("c" * 40), tree.behind(CLAIM)] == [False, False, False]
    assert len(calls) == 1


def test_a_timeout_on_the_ancestry_check_stops_further_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    healthy = FakeGit(tmp_path)

    def hang_on_ancestry(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        if "merge-base" in argv:
            healthy.calls.append((argv, kw))
            raise subprocess.TimeoutExpired("git", 5.0)
        return healthy(argv, **kw)

    _install(monkeypatch, hang_on_ancestry)
    tree = ClaimTree(tmp_path)

    assert tree.behind(CLAIM) is False
    assert tree.behind("c" * 40) is False
    assert len(healthy.calls) == 3, "the HEAD lookup, the commit check and the one ancestry check that hung"


def test_git_time_is_bounded_per_claim_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    fake = FakeGit(tmp_path, on_call=lambda: now.__setitem__(0, now[0] + 2.0))
    _install(monkeypatch, fake)
    assert claim_tree_module.CLAIM_TREE_GIT_BUDGET_SECONDS == GIT_TIMEOUT_SECONDS == 5.0
    tree = ClaimTree(tmp_path, clock=lambda: now[0])

    assert tree.behind(CLAIM) is True, "HEAD (2s) and commit check (4s) leave budget; the ancestry check overspends it"
    assert tree.behind("c" * 40) is False, "the budget is spent, so a new sha starts no process"
    assert tree.behind(CLAIM) is True, "a cached answer is still served"
    assert len(fake.calls) == 3


# FR08 -----------------------------------------------------------------------------------------------------


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    yield store
    store.close()


def _rows(backend: SQLiteBackend, shas: list[str]) -> None:
    for index, sha in enumerate(shas):
        backend.store(
            MemoryEntry(
                id=f"L-{index:03d}",
                content=f"claim {index}",
                assertions=[Assertion(type=AssertionType.GREP_PRESENT, pattern="p", target="t.py", commit_hash=sha)],
            )
        )


def _sweep(backend: SQLiteBackend, root: Path, **kwargs: Any) -> verification_pass.MaintainVerifySummary:
    return verification_pass.run_maintain_verify(
        backend,
        assertion_failure_penalty=0.15,
        assertion_stale_threshold_days=7,
        anchor_validity_verified_floor=0.8,
        batch_limit=100,
        project_root=root,
        **kwargs,
    )


def test_one_ancestry_check_per_distinct_sha(
    tmp_path: Path, backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    _rows(backend, [CLAIM] * 50)
    fake = FakeGit(root)
    _install(monkeypatch, fake)

    summary = _sweep(backend, root)

    assert summary.tree_behind == 50
    assert len(fake.calls) == 3


def test_past_the_distinct_sha_cap_entries_are_evaluated(
    tmp_path: Path, backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    monkeypatch.setattr(claim_tree_module, "CLAIM_TREE_MAX_DISTINCT_SHAS", 3)
    _rows(backend, [f"{digit}" * 40 for digit in "bcdef"])
    fake = FakeGit(root)
    _install(monkeypatch, fake)

    summary = _sweep(backend, root)

    assert len(fake.calls) == 1 + 2 * 3, "one HEAD lookup, a commit check and an ancestry check per first three shas"
    assert summary.tree_behind == 3
    held = [backend.get(f"L-{i:03d}", namespace="default") for i in range(5)]
    assert [e.assertions[0].last_result for e in held if e is not None] == [None, None, None, False, False]


def test_daemon_slices_of_one_call_share_one_claim_tree(
    tmp_path: Path, backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.tools._maintain_sweep import MaintainRun, verify_slice

    root = tmp_path.resolve()
    _rows(backend, [CLAIM] * 4)
    fake = FakeGit(root)
    _install(monkeypatch, fake)
    seen: list[ClaimTree | None] = []
    real = verification_pass.run_maintain_verify

    def spy(*args: Any, **kwargs: Any) -> verification_pass.MaintainVerifySummary:
        seen.append(kwargs.get("claim_tree"))
        return real(*args, **kwargs)

    monkeypatch.setattr(verification_pass, "run_maintain_verify", spy)
    run = MaintainRun("default", MemoryConfig(project_root=str(root)), None, None, None)

    for _ in range(4):  # one row per slice
        verify_slice(run, backend, 0.0, 1)
        if run.after is None:
            break

    assert len(seen) >= 2
    assert isinstance(seen[0], ClaimTree) and all(tree is seen[0] for tree in seen)
    assert len(fake.calls) == 3, "the slices of one call share the HEAD lookup and the ancestry answer"


# round 2: findings of the security review --------------------------------------------------------------------


def test_a_head_that_names_a_tag_object_is_read_as_the_commit_it_sits_on(tmp_path: Path) -> None:
    """HEAD spelled as an annotated tag object id: the checkout is AT the claim's commit, not behind it."""
    root, x, tag, _tree, _blob = _commit_and_tag(tmp_path / "r")
    (root / ".git" / "HEAD").write_text(tag + "\n", encoding="utf-8")
    assert _real_git(root, "rev-parse", "HEAD") == tag, "the trap is set: HEAD names the tag object"
    trapped = subprocess.run(["git", "merge-base", "--is-ancestor", tag, x], cwd=root, capture_output=True)
    assert trapped.returncode == 0, "git answers yes for the tag against its own commit"

    assert checkout_head(root) == x
    assert ClaimTree(root).behind(x) is False


def _commit_and_tag(path: Path) -> tuple[Path, str, str, str, str]:
    root, x = _repo(path)
    _real_git(root, "tag", "-a", "-m", "annotated", "T", x)
    tag = _real_git(root, "rev-parse", "T")
    tree = _real_git(root, "rev-parse", "HEAD^{tree}")
    blob = _real_git(root, "rev-parse", "HEAD:f.py")
    return root, x, tag, tree, blob


def test_a_tag_tree_or_blob_id_is_never_held_even_when_it_peels_to_the_checkout(tmp_path: Path) -> None:
    root, x, tag, tree, blob = _commit_and_tag(tmp_path / "r")
    assert tag != x and _real_git(root, "rev-parse", "T^{commit}") == x
    # control: git itself says HEAD is an ancestor of the tag object, which is why the id must be a commit
    assert subprocess.run(["git", "merge-base", "--is-ancestor", x, tag], cwd=root).returncode == 0

    tree_ = ClaimTree(root)
    assert [tree_.behind(tag), tree_.behind(tree), tree_.behind(blob)] == [False, False, False]


def test_a_tag_stamped_assertion_is_evaluated_not_held(tmp_path: Path, backend: SQLiteBackend) -> None:
    root, _x, tag, _tree, _blob = _commit_and_tag(tmp_path / "r")
    _rows(backend, [tag])

    summary = _sweep(backend, root)

    assert summary.tree_behind == 0
    entry = backend.get("L-000", namespace="default")
    assert entry is not None and entry.assertions[0].last_result is False  # t.py is absent: evaluated on the files


def test_a_replace_ref_cannot_manufacture_ancestry(tmp_path: Path) -> None:
    root, a = _repo(tmp_path / "r")
    _real_git(root, "checkout", "-q", "--orphan", "other")
    (root / "u.py").write_text("u = 1\n", encoding="utf-8")
    _real_git(root, "add", "u.py")
    _real_git(root, "commit", "-qm", "unrelated root")
    u = _real_git(root, "rev-parse", "HEAD")
    _real_git(root, "checkout", "-q", "--force", a)
    # a replace ref that makes U look like a child of A (what `git replace --graft U A` writes)
    _real_git(root, "replace", "--graft", u, a)
    assert subprocess.run(["git", "merge-base", "--is-ancestor", a, u], cwd=root).returncode == 0, "the trap is set"

    assert ClaimTree(root).behind(u) is False


def test_a_graft_file_cannot_manufacture_ancestry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, a = _repo(tmp_path / "r")
    _real_git(root, "checkout", "-q", "--orphan", "other")
    (root / "u.py").write_text("u = 1\n", encoding="utf-8")
    _real_git(root, "add", "u.py")
    _real_git(root, "commit", "-qm", "unrelated root")
    u = _real_git(root, "rev-parse", "HEAD")
    _real_git(root, "checkout", "-q", "--force", a)
    graft = tmp_path / "grafts"
    graft.write_text(f"{u} {a}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_GRAFT_FILE", str(graft))

    assert ClaimTree(root).behind(u) is False


def test_the_repositorys_own_grafts_file_cannot_manufacture_ancestry(tmp_path: Path) -> None:
    """The repository-controlled form: ``.git/info/grafts`` is still read by git, ``--no-replace-objects`` does
    not switch it off, and only pointing ``GIT_GRAFT_FILE`` at the null device does."""
    root, a = _repo(tmp_path / "r")
    _real_git(root, "checkout", "-q", "--orphan", "other")
    (root / "u.py").write_text("u = 1\n", encoding="utf-8")
    _real_git(root, "add", "u.py")
    _real_git(root, "commit", "-qm", "unrelated root")
    u = _real_git(root, "rev-parse", "HEAD")
    _real_git(root, "checkout", "-q", "--force", a)
    (root / ".git" / "info").mkdir(exist_ok=True)
    (root / ".git" / "info" / "grafts").write_text(f"{u} {a}\n", encoding="utf-8")
    trapped = subprocess.run(
        ["git", "--no-replace-objects", "merge-base", "--is-ancestor", a, u], cwd=root, capture_output=True
    )
    assert trapped.returncode == 0, "the trap is set: git itself answers yes through the grafts file"

    assert ClaimTree(root).behind(u) is False


def test_undecodable_git_output_is_no_answer_and_the_assertion_is_evaluated(
    tmp_path: Path, backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    def garbage(argv: list[str], **_kw: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b"\xff\xfe" + str(tmp_path).encode() + b"\n" + HEAD.encode(), b"")

    _install(monkeypatch, garbage)
    assert checkout_head(tmp_path) is None
    _rows(backend, [CLAIM])

    summary = _sweep(backend, tmp_path.resolve())

    assert summary.entry_failures == 0 and summary.tree_behind == 0
    entry = backend.get("L-000", namespace="default")
    assert entry is not None and entry.assertions[0].last_result is False
