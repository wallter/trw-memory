"""The launcher record: a store's daemon starts from the interpreter its record names, or not at all.

DAEMON-AUTOSTART-VERSION-RACE: a client on an older interpreter must never publish an older daemon beside the
one the store is pointed at. The probe runs a REAL interpreter; only the spawn itself is captured.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

import trw_memory
from trw_memory import __version__
from trw_memory.daemon import DaemonClient, DaemonPaths
from trw_memory.daemon import _launcher_record as record_module
from trw_memory.daemon._launcher_record import launch_from_record, read_launcher_record, write_launcher_record
from trw_memory.daemon._paths import write_secret_file
from trw_memory.exceptions import DaemonUnreachableError
from trw_memory.models.config import MemoryConfig

#: The source tree under test: the probe strips the caller's PYTHONPATH, so the record must carry it.
_SRC = str(Path(trw_memory.__file__).resolve().parents[1])


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("TRW_USER_DIR", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    return tmp_path / "home"


class _Start:
    def __init__(self) -> None:
        self.calls: list[tuple[DaemonPaths, dict[str, Any]]] = []

    def __call__(self, paths: DaemonPaths, **kwargs: Any) -> str:
        self.calls.append((paths, kwargs))
        return "spawned"


@pytest.fixture
def start(monkeypatch: pytest.MonkeyPatch) -> _Start:
    spawn = _Start()
    monkeypatch.setattr(record_module, "start_daemon_detached", spawn)
    return spawn


def _paths(directory: Path) -> DaemonPaths:
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    return DaemonPaths(user_memory_dir=directory)


def test_a_store_with_no_record_keeps_the_clients_own_start(tmp_path: Path, start: _Start) -> None:
    assert launch_from_record(_paths(tmp_path / "memory")) is None
    assert start.calls == []


def test_the_record_round_trips_and_names_the_interpreter(tmp_path: Path) -> None:
    paths = _paths(tmp_path / "memory")
    write_launcher_record(paths, Path(sys.executable), __version__, pythonpath="/src/a:/src/b")

    record = read_launcher_record(paths)

    assert record is not None
    assert (record.python, record.version, record.pythonpath) == (sys.executable, __version__, "/src/a:/src/b")
    assert record.written_at
    assert oct(paths.launcher.stat().st_mode & 0o777) == "0o600"


def test_a_current_record_starts_the_daemon_from_its_interpreter_and_replaces_the_clients_selectors(
    home: Path, start: _Start, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/the/clients/own")
    paths = _paths(home / ".trw" / "memory")
    write_launcher_record(paths, Path(sys.executable), __version__, pythonpath=_SRC)

    assert launch_from_record(paths) == "spawned"

    ((called_paths, kwargs),) = start.calls
    assert called_paths == paths
    assert kwargs["python"] == sys.executable
    assert kwargs["environ"]["PYTHONPATH"] == _SRC, "the record's PYTHONPATH replaces the client's own"
    assert "TRW_USER_DIR" not in kwargs["environ"], "the default store needs no TRW_USER_DIR"


def test_another_stores_record_sets_that_stores_user_dir(tmp_path: Path, home: Path, start: _Start) -> None:
    paths = _paths(tmp_path / "scratch-user" / "memory")
    write_launcher_record(paths, Path(sys.executable), __version__, pythonpath=_SRC)

    launch_from_record(paths)

    environ = start.calls[0][1]["environ"]
    assert environ["TRW_USER_DIR"] == str(tmp_path / "scratch-user")
    assert environ["PYTHONPATH"] == _SRC


def test_a_record_whose_interpreter_serves_an_older_version_refuses(tmp_path: Path, start: _Start) -> None:
    paths = _paths(tmp_path / "memory")
    write_launcher_record(paths, Path(sys.executable), "99.0.0", pythonpath=_SRC)

    with pytest.raises(DaemonUnreachableError, match="serves " + __version__.replace(".", r"\.")):
        launch_from_record(paths)

    assert start.calls == [], "falling back to the client's own interpreter is the bug being closed"
    record = read_launcher_record(paths)
    assert record is not None and record.version == "99.0.0"


def test_a_recorded_interpreter_upgraded_in_place_starts_and_the_record_follows(tmp_path: Path, start: _Start) -> None:
    """`pip install -U` (or a version bump in a `swap --src` tree) under a live record: 2026-10-08, recall was down
    for every session on the machine from the daemon's next exit until a manual `swap`."""
    paths = _paths(tmp_path / "memory")
    write_launcher_record(paths, Path(sys.executable), "0.0.1", pythonpath=_SRC)

    assert launch_from_record(paths) == "spawned"

    (_, kwargs), record = start.calls[0], read_launcher_record(paths)
    assert kwargs["python"] == sys.executable, "the RECORD's interpreter, never the client's own"
    assert kwargs["environ"]["PYTHONPATH"] == _SRC
    assert record is not None and (record.python, record.version, record.pythonpath) == (
        sys.executable,
        __version__,
        _SRC,
    )


def test_a_record_whose_interpreter_is_gone_refuses(tmp_path: Path, start: _Start) -> None:
    paths = _paths(tmp_path / "memory")
    write_launcher_record(paths, tmp_path / "gone" / "python", __version__)

    with pytest.raises(DaemonUnreachableError, match="does not run"):
        launch_from_record(paths)

    assert start.calls == []


@pytest.mark.parametrize("content", ["{not json", "[]", '{"python": "", "version": "1", "written_at": "x"}'])
def test_an_unreadable_record_refuses_instead_of_falling_back(tmp_path: Path, start: _Start, content: str) -> None:
    paths = _paths(tmp_path / "memory")
    write_secret_file(paths.launcher, content)

    with pytest.raises(DaemonUnreachableError, match="cannot be read"):
        launch_from_record(paths)

    assert start.calls == []


def test_a_client_autostarts_through_the_record_not_its_own_interpreter(
    tmp_path: Path, start: _Start, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring: DaemonClient._attach consults the record (so the trw-memory CLI and trw-mcp both do)."""
    from trw_memory.daemon import client as client_module

    own: list[DaemonPaths] = []
    monkeypatch.setattr(client_module, "start_daemon_detached", own.append)
    paths = _paths(tmp_path / "memory")
    write_launcher_record(paths, Path(sys.executable), __version__, pythonpath=_SRC)

    with pytest.raises(DaemonUnreachableError, match="did not publish"):
        DaemonClient(
            "g",
            config=MemoryConfig(memory_daemon_autostart=True, memory_daemon_startup_timeout_seconds=0.2),
            paths=paths,
        )._attach()

    assert len(start.calls) == 1
    assert own == []


def test_a_client_with_a_stale_record_refuses_and_starts_nothing(
    tmp_path: Path, start: _Start, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.daemon import client as client_module

    own: list[DaemonPaths] = []
    monkeypatch.setattr(client_module, "start_daemon_detached", own.append)
    paths = _paths(tmp_path / "memory")
    write_launcher_record(paths, Path(sys.executable), "99.0.0", pythonpath=_SRC)

    with pytest.raises(DaemonUnreachableError, match="so no memory daemon was started"):
        DaemonClient(
            "g",
            config=MemoryConfig(memory_daemon_autostart=True, memory_daemon_startup_timeout_seconds=0.2),
            paths=paths,
        )._attach()

    assert start.calls == [] and own == []


def test_a_server_registers_itself_when_the_store_has_no_record(tmp_path: Path) -> None:
    from trw_memory.daemon._launcher_record import register_launcher_record

    paths = _paths(tmp_path / "memory")

    assert register_launcher_record(paths, Path(sys.executable), "8.0.0", pythonpath="/src/a") is True

    record = read_launcher_record(paths)
    assert record is not None and (record.python, record.version, record.pythonpath) == (
        sys.executable,
        "8.0.0",
        "/src/a",
    )


@pytest.mark.parametrize(
    ("current", "writes"), [("7.9.9", True), ("8.0.0", False), ("8.0.1", False), ("unknown", False)]
)
def test_a_server_refreshes_only_an_older_record(tmp_path: Path, current: str, writes: bool) -> None:
    from trw_memory.daemon._launcher_record import register_launcher_record

    paths = _paths(tmp_path / "memory")
    write_launcher_record(paths, Path("/some/other/python"), current)

    assert register_launcher_record(paths, Path(sys.executable), "8.0.0") is writes

    record = read_launcher_record(paths)
    assert record is not None
    assert record.python == (sys.executable if writes else "/some/other/python")


def test_a_server_replaces_a_record_it_cannot_read(tmp_path: Path) -> None:
    from trw_memory.daemon._launcher_record import register_launcher_record

    paths = _paths(tmp_path / "memory")
    write_secret_file(paths.launcher, "{not json")

    assert register_launcher_record(paths, Path(sys.executable), "8.0.0") is True
    assert read_launcher_record(paths) is not None
