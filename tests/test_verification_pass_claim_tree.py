"""PRD-CORE-362: an assertion stamped with a commit the checkout is strictly behind is unknown, not failed.

Real git repositories in ``tmp_path`` and a real ``SQLiteBackend``; nothing about the rule is mocked.
Commits: A (no ``source.py``), B (child of A, adds ``def my_func``), D (child of A on a side branch).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from trw_memory.lifecycle import verification as verification_module
from trw_memory.lifecycle.claim_tree import ClaimTree
from trw_memory.lifecycle.verification_pass import MaintainVerifySummary, run_maintain_verify
from trw_memory.models.memory import Anchor, Assertion, AssertionResult, AssertionType, MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools._maintain_sweep import add_sweep_counts

_IDENT = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(root: Path, *args: str) -> str:
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")}, **_IDENT}
    done = subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True, text=True)
    return done.stdout.strip()


@dataclass
class Repo:
    root: Path
    a: str
    b: str
    d: str

    def checkout(self, sha: str) -> None:
        _git(self.root, "checkout", "-q", sha)


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    _git(root, "init", "-q")
    (root / "other.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "other.py")
    _git(root, "commit", "-qm", "a")
    a = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-qb", "side")
    (root / "side.py").write_text("y = 1\n", encoding="utf-8")
    _git(root, "add", "side.py")
    _git(root, "commit", "-qm", "d")
    d = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", a)
    _git(root, "checkout", "-qb", "main2")
    (root / "source.py").write_text("def my_func(): ...\n", encoding="utf-8")
    _git(root, "add", "source.py")
    _git(root, "commit", "-qm", "b")
    b = _git(root, "rev-parse", "HEAD")
    return Repo(root, a, b, d)


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    yield store
    store.close()


def _store(backend: SQLiteBackend, entry_id: str, *commits: str | None, anchor: bool = False) -> None:
    backend.store(
        MemoryEntry(
            id=entry_id,
            content="claim " + entry_id,
            assertions=[
                Assertion(type=AssertionType.GREP_PRESENT, pattern="my_func", target="source.py", commit_hash=c)
                for c in commits
            ],
            anchors=[Anchor(file="source.py", symbol_name="my_func")] if anchor else [],
            anchor_validity=1.0 if anchor else None,
        )
    )


def _sweep(backend: SQLiteBackend, root: Path | None, **kwargs: object) -> MaintainVerifySummary:
    return run_maintain_verify(
        backend,
        assertion_failure_penalty=0.15,
        assertion_stale_threshold_days=7,
        anchor_validity_verified_floor=0.8,
        batch_limit=10,
        project_root=root,
        **kwargs,  # type: ignore[arg-type]
    )


def _row(backend: SQLiteBackend, entry_id: str = "L-1") -> MemoryEntry:
    entry = backend.get(entry_id, namespace="default")
    assert entry is not None
    return entry


# FR04 -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head", "claim", "expected"),
    [
        ("a", "b", True),  # HEAD is a strict ancestor of the claim commit
        ("b", "b", False),  # equal
        ("b", "a", False),  # HEAD ahead of the claim
        ("d", "b", False),  # diverged
        ("a", "unknown", False),  # an object this repository lacks
        ("a", "UPPER", False),  # not 40 lowercase hex
    ],
)
def test_behind_is_true_only_for_a_strict_ancestor_head(repo: Repo, head: str, claim: str, expected: bool) -> None:
    shas = {
        "a": repo.a,
        "b": repo.b,
        "d": repo.d,
        "unknown": "1" * 40,
        "UPPER": repo.b.upper(),
    }
    repo.checkout(shas[head])

    assert ClaimTree(repo.root).behind(shas[claim]) is expected


# FR05 -----------------------------------------------------------------------------------------------------


def test_a_checkout_behind_the_claim_records_unknown_not_failed(repo: Repo, backend: SQLiteBackend) -> None:
    repo.checkout(repo.a)
    _store(backend, "L-1", repo.b)

    summary = _sweep(backend, repo.root)

    (assertion,) = _row(backend).assertions
    assert assertion.last_result is None and assertion.first_failed_at is None
    assert assertion.last_verified_at is not None
    assert repo.a[:12] in assertion.last_evidence and repo.b[:12] in assertion.last_evidence
    assert summary.entries_processed == 1


def test_no_file_is_walked_for_a_held_assertion(
    repo: Repo, backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.checkout(repo.a)
    _store(backend, "L-1", repo.b)
    seen: list[int] = []
    real = verification_module.verify_assertions

    def spy(assertions: list[Assertion], root: Path | None, **kw: object) -> list[AssertionResult]:
        seen.append(len(assertions))
        return list(real(assertions, root, **kw))  # type: ignore[arg-type]

    monkeypatch.setattr(verification_module, "verify_assertions", spy)

    _sweep(backend, repo.root)

    assert seen == [0], "the held assertion must not be handed to the file check"


def test_a_held_entry_keeps_its_stored_anchor_validity(repo: Repo, backend: SQLiteBackend) -> None:
    repo.checkout(repo.a)  # source.py is absent here, so a rescore would give 0.0
    _store(backend, "L-1", repo.b, anchor=True)

    _sweep(backend, repo.root)

    entry = _row(backend)
    assert entry.anchor_validity == 1.0
    assert entry.verification_status is None


def test_an_entry_with_one_held_assertion_still_evaluates_the_others(repo: Repo, backend: SQLiteBackend) -> None:
    repo.checkout(repo.a)
    _store(backend, "L-1", repo.b, None)  # the second assertion carries no stamp: it is evaluated, and fails

    _sweep(backend, repo.root)

    held, evaluated = _row(backend).assertions
    assert held.last_result is None and held.first_failed_at is None
    assert evaluated.last_result is False and evaluated.first_failed_at is not None
    assert _row(backend).verification_status != "verified"


# FR06 -----------------------------------------------------------------------------------------------------


def test_a_stored_failure_becomes_unknown_when_the_tree_is_behind(repo: Repo, backend: SQLiteBackend) -> None:
    repo.checkout(repo.a)
    _store(backend, "L-1", repo.b)
    # the old sweep: no claim tree, so the missing file is recorded as a failure
    _sweep(backend, repo.root, claim_tree=_NeverBehind())
    (failed,) = _row(backend).assertions
    assert failed.last_result is False and failed.first_failed_at is not None

    _sweep(backend, repo.root)

    (held,) = _row(backend).assertions
    assert held.last_result is None
    assert held.first_failed_at is None
    assert repo.a[:12] in held.last_evidence and repo.b[:12] in held.last_evidence
    assert held.last_verified_at is not None and held.last_verified_at > failed.last_verified_at  # type: ignore[operator]


class _NeverBehind(ClaimTree):
    def __init__(self) -> None:
        super().__init__(Path("."))

    def behind(self, sha: object) -> bool:
        return False


def test_an_entry_whose_assertions_are_all_held_is_still_persisted(repo: Repo, backend: SQLiteBackend) -> None:
    repo.checkout(repo.b)
    _store(backend, "L-1", repo.b)
    _sweep(backend, repo.root)
    assert _row(backend).assertions[0].last_result is True
    assert _row(backend).verification_status == "verified"

    repo.checkout(repo.a)
    _sweep(backend, repo.root)

    entry = _row(backend)
    assert entry.assertions[0].last_result is None
    assert entry.assertions[0].last_evidence.startswith("checkout HEAD ")
    assert entry.verification_status is None, "a verified verdict needs current evidence"


def test_a_second_sweep_while_behind_is_stable(repo: Repo, backend: SQLiteBackend) -> None:
    repo.checkout(repo.a)
    _store(backend, "L-1", repo.b)

    _sweep(backend, repo.root)
    first = _row(backend).assertions[0].model_dump(mode="json")
    _sweep(backend, repo.root)
    second = _row(backend).assertions[0].model_dump(mode="json")

    for key in ("last_result", "first_failed_at", "last_evidence"):
        assert first[key] == second[key]


# FR07 -----------------------------------------------------------------------------------------------------


def test_an_up_to_date_checkout_still_fails_a_false_claim(repo: Repo, backend: SQLiteBackend) -> None:
    # B's own tree lacks "absent_symbol", so the claim is false there and at a descendant of B
    backend.store(
        MemoryEntry(
            id="L-1",
            content="false claim",
            assertions=[
                Assertion(
                    type=AssertionType.GREP_PRESENT, pattern="absent_symbol", target="source.py", commit_hash=repo.b
                )
            ],
        )
    )
    (repo.root / "child.py").write_text("z = 1\n", encoding="utf-8")
    _git(repo.root, "add", "child.py")
    _git(repo.root, "commit", "-qm", "child")
    child = _git(repo.root, "rev-parse", "HEAD")

    for at in (repo.b, child):
        repo.checkout(at)
        _sweep(backend, repo.root)
        (assertion,) = _row(backend).assertions
        assert assertion.last_result is False, f"at {at[:7]}"


def test_a_diverged_or_squashed_claim_commit_is_still_evaluated(repo: Repo, backend: SQLiteBackend) -> None:
    repo.checkout(repo.d)  # a sibling of B: not an ancestor of it
    _store(backend, "L-1", repo.b)

    summary = _sweep(backend, repo.root)

    (assertion,) = _row(backend).assertions
    assert assertion.last_result is False
    assert summary.tree_behind == 0


def test_an_unknown_sha_no_sha_and_no_git_are_evaluated(
    repo: Repo, backend: SQLiteBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.checkout(repo.a)
    _store(backend, "L-missing", "1" * 40)
    _store(backend, "L-none", None)
    _sweep(backend, repo.root)
    assert _row(backend, "L-missing").assertions[0].last_result is False
    assert _row(backend, "L-none").assertions[0].last_result is False

    # a root that is not a repository, with a real file so a verdict is possible
    plain = tmp_path / "plain"
    plain.mkdir()
    _store(backend, "L-plain", repo.b)
    _sweep(backend, plain.resolve())
    assert _row(backend, "L-plain").assertions[0].last_result is False

    # git absent from PATH
    _store(backend, "L-nogit", repo.b)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    _sweep(backend, repo.root)
    assert _row(backend, "L-nogit").assertions[0].last_result is False


def test_moving_the_checkout_forward_evaluates_the_claim_again(repo: Repo, backend: SQLiteBackend) -> None:
    repo.checkout(repo.a)
    _store(backend, "L-1", repo.b, anchor=True)
    _sweep(backend, repo.root)
    assert _row(backend).assertions[0].last_result is None

    repo.checkout(repo.b)
    _sweep(backend, repo.root)

    entry = _row(backend)
    assert entry.assertions[0].last_result is True
    assert entry.verification_status == "verified"
    assert entry.anchor_validity == 1.0


# FR09 -----------------------------------------------------------------------------------------------------


def test_the_summary_counts_entries_held_as_tree_behind(repo: Repo, backend: SQLiteBackend) -> None:
    repo.checkout(repo.a)
    _store(backend, "L-1", repo.b, repo.b)  # two held assertions still count one entry
    _store(backend, "L-2", repo.b)
    _store(backend, "L-3", None)

    summary = _sweep(backend, repo.root)

    assert summary.tree_behind == 2
    assert summary.entries_processed == 3
    assert summary.as_dict()["tree_behind"] == 2


def test_slice_counts_of_tree_behind_are_added() -> None:
    merged = add_sweep_counts({"tree_behind": 2, "entries_processed": 3}, {"tree_behind": 1, "entries_processed": 4})

    assert merged["tree_behind"] == 3
    assert merged["entries_processed"] == 7
