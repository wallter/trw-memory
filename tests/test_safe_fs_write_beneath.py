"""PRD-CORE-337 FR01-FR03 / NFR03: ``trw_memory.safe_fs`` writes beneath a root and never follows a symlink.

Every test builds a hostile layout under ``tmp_path``: a ``project`` root to write beneath and an
``outside`` directory standing in for everything the write must never reach. The race tests wrap
``os.open``/``os.replace`` to swap a symlink in at the exact step a by-name implementation would lose.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from trw_memory import safe_fs
from trw_memory.exceptions import UnsafeWriteError, UntrustedDirectoryError
from trw_memory.safe_fs import append_beneath, write_beneath

pytestmark = pytest.mark.skipif(safe_fs._BRANCH != "posix", reason="the race-safe contract is the POSIX branch")


@pytest.fixture
def layout(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "target.txt").write_bytes(b"outside-original")
    return project, outside


def _outside_snapshot(outside: Path) -> dict[str, bytes]:
    return {path.relative_to(outside).as_posix(): path.read_bytes() for path in outside.rglob("*") if path.is_file()}


def _no_temp_files(directory: Path) -> bool:
    return not [name for name in os.listdir(directory) if name.endswith(".tmp")]


# --- FR01: write_beneath -------------------------------------------------------------------------


def test_a_symlinked_parent_component_is_refused_not_followed(layout: tuple[Path, Path]) -> None:
    project, outside = layout
    (project / "a").mkdir()
    (project / "a" / "b").symlink_to(outside, target_is_directory=True)
    before = _outside_snapshot(outside)

    with pytest.raises(UnsafeWriteError) as refused:
        write_beneath(project, "a/b/c/target.txt", b"x", mode=0o644)

    assert refused.value.reason == "symlink_component"
    assert refused.value.path == str(project / "a" / "b")
    assert _outside_snapshot(outside) == before
    assert not (outside / "c").exists()

    (project / "a" / "b").unlink()
    write_beneath(project, "a/b/c/target.txt", b"x", mode=0o644)
    assert (project / "a" / "b" / "c" / "target.txt").read_bytes() == b"x"


@pytest.mark.parametrize("rel_path", ["new.txt", "deep/er/new.txt"])
def test_write_beneath_creates_missing_parents_and_the_file(layout: tuple[Path, Path], rel_path: str) -> None:
    project, _outside = layout
    write_beneath(project, rel_path, b"payload", mode=0o644)
    assert (project / rel_path).read_bytes() == b"payload"
    assert _no_temp_files((project / rel_path).parent)


def test_write_beneath_replaces_an_existing_file_whole(layout: tuple[Path, Path]) -> None:
    project, _outside = layout
    (project / "cfg.json").write_bytes(b"a much longer previous content")
    write_beneath(project, "cfg.json", b"short", mode=0o644)
    assert (project / "cfg.json").read_bytes() == b"short"


def test_a_symlinked_leaf_is_refused_for_write(layout: tuple[Path, Path]) -> None:
    project, outside = layout
    (project / "hooks.json").symlink_to(outside / "target.txt")

    with pytest.raises(UnsafeWriteError) as refused:
        write_beneath(project, "hooks.json", b"generated", mode=0o644)

    assert refused.value.reason == "symlink_leaf"
    assert (outside / "target.txt").read_bytes() == b"outside-original"
    assert (project / "hooks.json").is_symlink()
    assert _no_temp_files(project)


def test_a_hardlinked_leaf_is_replaced_not_written_through(layout: tuple[Path, Path]) -> None:
    """``os.replace`` swaps the directory entry, so a second name for an outside inode is left intact."""
    project, outside = layout
    os.link(outside / "target.txt", project / "linked.txt")

    write_beneath(project, "linked.txt", b"generated", mode=0o644)

    assert (project / "linked.txt").read_bytes() == b"generated"
    assert (outside / "target.txt").read_bytes() == b"outside-original"


# --- FR02: append_beneath ------------------------------------------------------------------------


def test_append_beneath_refuses_a_symlinked_leaf(layout: tuple[Path, Path]) -> None:
    project, outside = layout
    (project / "events.jsonl").symlink_to(outside / "target.txt")

    with pytest.raises(UnsafeWriteError) as refused:
        append_beneath(project, "events.jsonl", b"line\n", mode=0o644)

    assert refused.value.reason == "symlink_leaf"
    assert (outside / "target.txt").read_bytes() == b"outside-original"

    (project / "events.jsonl").unlink()
    append_beneath(project, "logs/events.jsonl", b"one\n", mode=0o640)
    append_beneath(project, "logs/events.jsonl", b"two\n", mode=0o640)
    created = project / "logs" / "events.jsonl"
    assert created.read_bytes() == b"one\ntwo\n"
    assert stat.S_IMODE(created.stat().st_mode) == 0o640 & ~_umask()


def test_append_beneath_refuses_a_symlinked_parent(layout: tuple[Path, Path]) -> None:
    project, outside = layout
    (project / ".codex").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeWriteError) as refused:
        append_beneath(project, ".codex/target.txt", b"appended", mode=0o644)

    assert refused.value.reason == "symlink_component"
    assert (outside / "target.txt").read_bytes() == b"outside-original"


# --- FR03 / NFR03: one typed refusal -------------------------------------------------------------


def _symlinked_parent(project: Path, outside: Path) -> str:
    (project / "dir").symlink_to(outside, target_is_directory=True)
    return "dir/target.txt"


def _symlinked_leaf(project: Path, outside: Path) -> str:
    (project / "target.txt").symlink_to(outside / "target.txt")
    return "target.txt"


def _dangling_symlinked_leaf(project: Path, outside: Path) -> str:
    (project / "target.txt").symlink_to(outside / "not-there-yet.txt")
    return "target.txt"


def _file_where_a_directory_belongs(project: Path, _outside: Path) -> str:
    (project / "dir").write_bytes(b"i am a file")
    return "dir/target.txt"


def _directory_at_the_leaf(project: Path, _outside: Path) -> str:
    (project / "target.txt").mkdir()
    return "target.txt"


def _fifo_at_the_leaf(project: Path, _outside: Path) -> str:
    os.mkfifo(project / "target.txt")
    return "target.txt"


def _escaping_parent_reference(_project: Path, _outside: Path) -> str:
    return "../outside/target.txt"


def _absolute_path(_project: Path, outside: Path) -> str:
    return str(outside / "target.txt")


def _symlinked_root(project: Path, outside: Path) -> str:
    project.rmdir()
    project.symlink_to(outside, target_is_directory=True)
    return "target.txt"


_Setup = Callable[[Path, Path], str]

_REFUSALS: list[tuple[_Setup, str]] = [
    (_symlinked_parent, "symlink_component"),
    (_symlinked_leaf, "symlink_leaf"),
    (_dangling_symlinked_leaf, "symlink_leaf"),
    (_file_where_a_directory_belongs, "not_a_directory"),
    (_directory_at_the_leaf, "leaf_not_regular_file"),
    (_fifo_at_the_leaf, "leaf_not_regular_file"),
    (_escaping_parent_reference, "escapes_root"),
    (_absolute_path, "escapes_root"),
    (_symlinked_root, "root_untrusted"),
]


@pytest.mark.parametrize("operation", [write_beneath, append_beneath], ids=["write", "append"])
@pytest.mark.parametrize(("setup", "reason"), _REFUSALS, ids=[setup.__name__.lstrip("_") for setup, _ in _REFUSALS])
def test_every_refusal_path_raises_the_typed_error(
    layout: tuple[Path, Path], setup: _Setup, reason: str, operation: Callable[..., None]
) -> None:
    project, outside = layout
    rel_path = setup(project, outside)
    before = _outside_snapshot(outside)

    with pytest.raises(UnsafeWriteError) as refused:
        operation(project, rel_path, b"secret-payload", mode=0o600)

    error = refused.value
    assert isinstance(error, UntrustedDirectoryError)
    assert not isinstance(error, OSError)
    assert error.reason == reason
    assert error.path
    assert "secret-payload" not in str(error)
    assert _outside_snapshot(outside) == before


@pytest.mark.parametrize(
    ("os_name", "dir_fd", "nofollow", "branch"),
    [
        ("posix", True, True, "posix"),
        ("posix", False, True, "unsupported"),
        ("posix", True, False, "unsupported"),
        ("nt", False, False, "windows"),
    ],
)
def test_the_platform_branch_is_selected_from_the_capabilities(
    os_name: str, dir_fd: bool, nofollow: bool, branch: str
) -> None:
    assert safe_fs._select_branch(os_name, dir_fd, nofollow) == branch


@pytest.mark.parametrize("missing", [(False, True), (True, False)], ids=["no-dir_fd", "no-O_NOFOLLOW"])
@pytest.mark.parametrize("operation", [write_beneath, append_beneath], ids=["write", "append"])
def test_a_posix_platform_without_the_capabilities_refuses_rather_than_downgrades(
    layout: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    missing: tuple[bool, bool],
    operation: Callable[..., None],
) -> None:
    project, _outside = layout
    monkeypatch.setattr(safe_fs, "_BRANCH", safe_fs._select_branch("posix", *missing))

    with pytest.raises(UnsafeWriteError) as refused:
        operation(project, "sub/target.txt", b"x", mode=0o644)

    assert refused.value.reason == "unsupported_platform"
    assert list(project.iterdir()) == []


# --- mode at creation, atomic publish ------------------------------------------------------------


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


def test_the_file_is_created_at_its_final_mode_with_no_chmod_window(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every mode the new file is ever observable at is 0600: at creation, before publish, after publish."""
    project, _outside = layout
    (project / "credentials.yaml").write_bytes(b"old")
    os.chmod(project / "credentials.yaml", 0o644)
    observed: list[int] = []
    real_open, real_replace = os.open, os.replace

    def spying_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        fd = real_open(path, flags, *args, **kwargs)
        if flags & os.O_CREAT:
            observed.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return fd

    def spying_replace(src: str, dst: str, **kwargs: Any) -> None:
        observed.append(stat.S_IMODE(os.stat(src, dir_fd=kwargs["src_dir_fd"]).st_mode))
        real_replace(src, dst, **kwargs)

    def no_chmod(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("safe_fs must set the mode at creation, never chmod afterwards")

    monkeypatch.setattr(os, "open", spying_open)
    monkeypatch.setattr(os, "replace", spying_replace)
    monkeypatch.setattr(os, "chmod", no_chmod)
    monkeypatch.setattr(os, "fchmod", no_chmod)
    previous_umask = os.umask(0)  # no umask narrowing: the mode observed is exactly the mode requested
    try:
        write_beneath(project, "credentials.yaml", b"platform_api_key: s3cret\n", mode=0o600)
    finally:
        os.umask(previous_umask)

    observed.append(stat.S_IMODE((project / "credentials.yaml").stat().st_mode))
    assert observed == [0o600, 0o600, 0o600]
    assert (project / "credentials.yaml").read_bytes() == b"platform_api_key: s3cret\n"


def test_a_failed_write_leaves_the_previous_file_whole_and_no_temp_behind(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _outside = layout
    (project / "state.json").write_bytes(b"previous")
    real_write = os.write
    seen_during_write: list[bytes] = []

    def failing_write(fd: int, data: Any) -> int:
        real_write(fd, bytes(data)[:3])
        seen_during_write.append((project / "state.json").read_bytes())
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", failing_write)

    with pytest.raises(OSError, match="No space left"):
        write_beneath(project, "state.json", b"replacement", mode=0o644)

    assert seen_during_write == [b"previous"]
    assert (project / "state.json").read_bytes() == b"previous"
    assert _no_temp_files(project)


# --- races: a symlink swapped in between two steps -----------------------------------------------


def test_a_parent_swapped_for_a_symlink_mid_walk_cannot_redirect_the_write(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """After ``a`` is opened, ``a`` is renamed away and replaced by a symlink to ``outside``.

    The walk holds ``a``'s descriptor, so ``b`` and the file land in the renamed real directory;
    a by-name implementation would write into ``outside/b/target.txt``.
    """
    project, outside = layout
    (project / "a" / "b").mkdir(parents=True)
    real_open = os.open
    swapped: list[bool] = []

    def swapping_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == "b" and not swapped:
            os.rename(project / "a", project / "a-real")
            (project / "a").symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)
    write_beneath(project, "a/b/target.txt", b"generated", mode=0o644)

    assert swapped == [True]
    assert (project / "a-real" / "b" / "target.txt").read_bytes() == b"generated"
    assert (outside / "target.txt").read_bytes() == b"outside-original"
    assert not (outside / "b").exists()


def test_a_component_swapped_for_a_symlink_just_before_its_open_is_refused(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, outside = layout
    (project / "a" / "b").mkdir(parents=True)
    real_open = os.open

    def swapping_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == "b" and (project / "a" / "b").is_dir() and not (project / "a" / "b").is_symlink():
            os.rename(project / "a" / "b", project / "a" / "b-real")
            (project / "a" / "b").symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)
    with pytest.raises(UnsafeWriteError) as refused:
        write_beneath(project, "a/b/target.txt", b"generated", mode=0o644)

    assert refused.value.reason == "symlink_component"
    assert (outside / "target.txt").read_bytes() == b"outside-original"


def test_a_leaf_symlink_planted_just_before_publish_is_replaced_not_followed(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, outside = layout
    real_replace = os.replace

    def planting_replace(src: str, dst: str, **kwargs: Any) -> None:
        (project / "hooks.json").symlink_to(outside / "target.txt")
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", planting_replace)
    write_beneath(project, "hooks.json", b"generated", mode=0o644)

    assert not (project / "hooks.json").is_symlink()
    assert (project / "hooks.json").read_bytes() == b"generated"
    assert (outside / "target.txt").read_bytes() == b"outside-original"


# --- FR04: the Windows branch, exercised by selecting it on POSIX -------------------------------


@pytest.mark.parametrize("operation", [write_beneath, append_beneath], ids=["write", "append"])
def test_the_best_effort_branch_refuses_a_detected_symlink_and_otherwise_writes(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, operation: Callable[..., None]
) -> None:
    """Not a Windows test: it pins that the lstat branch refuses only what lstat detects, and writes."""
    project, outside = layout
    monkeypatch.setattr(safe_fs, "_BRANCH", "windows")
    (project / "linked").symlink_to(outside, target_is_directory=True)
    (project / "leaf.txt").symlink_to(outside / "target.txt")

    with pytest.raises(UnsafeWriteError, match="symlink_component"):
        operation(project, "linked/target.txt", b"x", mode=0o644)
    with pytest.raises(UnsafeWriteError, match="symlink_leaf"):
        operation(project, "leaf.txt", b"x", mode=0o644)
    operation(project, "fresh/dir/out.txt", b"written", mode=0o644)

    assert (outside / "target.txt").read_bytes() == b"outside-original"
    assert (project / "fresh" / "dir" / "out.txt").read_bytes() == b"written"


# --- review round 1 P1: the errno alone decides a refusal, not a follow-up stat ------------------


def _vanished(_dir_fd: int, _name: str) -> int | None:
    """``_kind_at`` after the symlink that failed the open was removed again: it reports "absent"."""
    return None


def test_an_eloop_append_open_is_a_refusal_even_when_the_symlink_vanished(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _outside = layout

    def eloop(*_args: Any, **_kwargs: Any) -> int:
        raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), "events.jsonl")

    monkeypatch.setattr(safe_fs, "open_or_create_at", eloop)
    monkeypatch.setattr(safe_fs, "_kind_at", _vanished)

    with pytest.raises(UnsafeWriteError) as refused:
        append_beneath(project, "events.jsonl", b"line\n", mode=0o644)

    assert refused.value.reason == "symlink_leaf"
    assert refused.value.path == str(project / "events.jsonl")


@pytest.mark.parametrize(
    ("error_number", "reason"),
    [(errno.ELOOP, "symlink_component"), (errno.ENOTDIR, "not_a_directory")],
    ids=["ELOOP", "ENOTDIR"],
)
@pytest.mark.parametrize("operation", [write_beneath, append_beneath], ids=["write", "append"])
def test_a_failed_no_follow_directory_open_is_a_refusal_even_when_the_entry_vanished(
    layout: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    reason: str,
    operation: Callable[..., None],
) -> None:
    project, _outside = layout
    real_open = os.open

    def failing_component_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == "sub" and kwargs.get("dir_fd") is not None:
            raise OSError(error_number, os.strerror(error_number), path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", failing_component_open)
    monkeypatch.setattr(safe_fs, "_kind_at", _vanished)

    with pytest.raises(UnsafeWriteError) as refused:
        operation(project, "sub/target.txt", b"x", mode=0o644)

    assert refused.value.reason == reason
    assert refused.value.path == str(project / "sub")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_exact_mode_publishes_the_requested_bits_unnarrowed_and_before_any_byte(tmp_path: Path) -> None:
    """``exact_mode`` is for a caller keeping the bits of the file it replaces: the umask must not narrow them again.

    The default (no ``exact_mode``) still lets the umask narrow the mode, as ``open()`` does.
    """
    project = tmp_path / "project"
    project.mkdir()
    previous_umask = os.umask(0o022)
    try:
        write_beneath(project, "kept.json", b"{}\n", mode=0o664, exact_mode=True)
        write_beneath(project, "default.json", b"{}\n", mode=0o664)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((project / "kept.json").stat().st_mode) == 0o664
    assert stat.S_IMODE((project / "default.json").stat().st_mode) == 0o644
