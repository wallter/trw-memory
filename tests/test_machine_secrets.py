"""``trw_memory.machine_secrets``: the shared owner-only reader/writer for ``~/.trw`` secret files."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from trw_memory.machine_secrets import machine_secret_path, read_private_file, write_private_file


def _lines(text: str) -> list[str]:
    return text.splitlines()


def test_write_creates_0600_file_and_0700_dir_then_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "home" / ".trw" / "credentials.yaml"
    (tmp_path / "home").mkdir()

    write_private_file(path, "platform_api_key: abc\n")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    read = read_private_file(path, _lines)
    assert (read.value, read.present, read.problem) == (["platform_api_key: abc"], True, "")


def test_absent_file_is_not_a_problem(tmp_path: Path) -> None:
    read = read_private_file(tmp_path / "nope", _lines)
    assert (read.value, read.present, read.problem) == (None, False, "")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_group_or_world_bits_are_refused_without_calling_the_parser(tmp_path: Path) -> None:
    path = tmp_path / "s.env"
    path.write_text("K=v\n", encoding="utf-8")
    path.chmod(0o640)

    def _never(_text: str) -> object:
        raise AssertionError("parser saw a refused file")

    read = read_private_file(path, _never, shown="~/.trw/s.env")

    assert read.value is None and "looser than 0600 (fix: chmod 600 ~/.trw/s.env)" in read.problem


def test_symlink_and_directory_are_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.write_text("K=v\n", encoding="utf-8")
    real.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(real)
    (tmp_path / "dir").mkdir()

    assert read_private_file(link, _lines).problem == "is a symlink"
    assert read_private_file(tmp_path / "dir", _lines).problem == "is not a regular file"


def test_write_refuses_a_symlinked_leaf(tmp_path: Path) -> None:
    from trw_memory.exceptions import UnsafeWriteError

    target = tmp_path / "victim"
    target.write_text("keep\n", encoding="utf-8")
    (tmp_path / "s.env").symlink_to(target)

    with pytest.raises(UnsafeWriteError):
        write_private_file(tmp_path / "s.env", "K=v\n")
    assert target.read_text(encoding="utf-8") == "keep\n"


def test_machine_secret_path_is_under_home_trw_and_rejects_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert machine_secret_path("jev.env") == tmp_path / ".trw" / "jev.env"
    with pytest.raises(ValueError):
        machine_secret_path("../x")
