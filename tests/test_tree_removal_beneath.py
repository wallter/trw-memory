"""``remove_tree_beneath`` never removes through a symlinked ancestor, and ``append_beneath(sync=True)`` syncs (REMOVE-TREE-ANCESTOR-SYMLINK)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import structlog

from trw_memory import safe_fs
from trw_memory._tree_removal import remove_tree_beneath

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor-anchored removal is the POSIX branch")


@pytest.fixture()
def events() -> list[dict[str, object]]:
    with structlog.testing.capture_logs() as captured:
        yield captured


def _failures(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [e for e in events if e["event"] == "tree_removal_failed"]


def test_a_tree_beneath_the_root_is_removed(tmp_path: Path, events: list[dict[str, object]]) -> None:
    (tmp_path / "a" / "b" / "leaf").mkdir(parents=True)
    (tmp_path / "a" / "b" / "leaf" / "f").write_text("x")

    remove_tree_beneath(tmp_path, Path("a/b/leaf"), purpose="test")

    assert not (tmp_path / "a" / "b" / "leaf").exists() and (tmp_path / "a" / "b").is_dir()
    assert _failures(events) == []


def test_a_symlinked_ancestor_is_refused_and_its_target_survives(
    tmp_path: Path, events: list[dict[str, object]]
) -> None:
    outside = tmp_path / "outside"
    (outside / "leaf").mkdir(parents=True)
    (outside / "leaf" / "keep").write_text("x")
    root = tmp_path / "root"
    root.mkdir()
    (root / "a").symlink_to(outside, target_is_directory=True)

    remove_tree_beneath(root, Path("a/leaf"), purpose="test")

    assert (outside / "leaf" / "keep").read_text() == "x"
    [failure] = _failures(events)
    assert "symlink_component" in str(failure["first_error"])


def test_a_symlink_leaf_is_unlinked_not_followed(tmp_path: Path, events: list[dict[str, object]]) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("x")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    remove_tree_beneath(root, Path("link"), purpose="test")

    assert not (root / "link").is_symlink() and (outside / "keep").read_text() == "x"
    assert _failures(events) == []


def test_a_symlink_inside_the_tree_is_unlinked_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("x")
    root = tmp_path / "root"
    (root / "t").mkdir(parents=True)
    (root / "t" / "inner").symlink_to(outside, target_is_directory=True)

    remove_tree_beneath(root, Path("t"), purpose="test")

    assert not (root / "t").exists() and (outside / "keep").read_text() == "x"


def test_an_absent_leaf_or_ancestor_is_a_quiet_no_op(tmp_path: Path, events: list[dict[str, object]]) -> None:
    remove_tree_beneath(tmp_path, Path("never/created"), purpose="test")
    (tmp_path / "there").mkdir()
    remove_tree_beneath(tmp_path, Path("there/never"), purpose="test")

    assert _failures(events) == [] and (tmp_path / "there").is_dir()


def test_an_ancestor_swapped_for_a_link_after_the_walk_cannot_redirect_the_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    (outside / "leaf").mkdir(parents=True)
    (outside / "leaf" / "keep").write_text("x")
    root = tmp_path / "root"
    (root / "a" / "leaf").mkdir(parents=True)
    (root / "a" / "leaf" / "mine").write_text("x")
    real_open_parent = safe_fs.open_parent_beneath

    def swap_after_walk(r: Path, rel: object) -> tuple[int, str]:
        fd_and_leaf = real_open_parent(r, rel)  # type: ignore[arg-type]
        (root / "a").rename(tmp_path / "moved")  # the attacker swaps the ancestor for a link once it is opened
        (root / "a").symlink_to(outside, target_is_directory=True)
        return fd_and_leaf

    monkeypatch.setattr(safe_fs, "open_parent_beneath", swap_after_walk)
    remove_tree_beneath(root, Path("a/leaf"), purpose="test")

    assert (outside / "leaf" / "keep").read_text() == "x", "the removal followed the swapped-in link"
    assert not (tmp_path / "moved" / "leaf").exists(), "the descriptor still names the original directory"


def test_append_beneath_syncs_only_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])

    safe_fs.append_beneath(tmp_path, "f", b"a", mode=0o600)
    assert synced == []
    safe_fs.append_beneath(tmp_path, "f", b"b", mode=0o600, sync=True)

    assert len(synced) == 1 and (tmp_path / "f").read_bytes() == b"ab"


def test_a_path_with_an_embedded_nul_is_logged_not_raised(tmp_path: Path, events: list[dict[str, object]]) -> None:
    remove_tree_beneath(tmp_path, Path("a\x00b"), purpose="test")

    [failure] = _failures(events)
    assert "null" in str(failure["first_error"]).lower()
