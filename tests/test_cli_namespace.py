"""PRD-CORE-253 FR05 — the ``trw-memory namespace`` verbs, over a real daemon.

The CLI process must open **no** SQLite connection: an invocation that did
would be an extra writer on a store the daemon is meant to own alone. These
tests run the verbs against a real subprocess daemon and assert on the store's
state afterwards, so "it went over the daemon" is proved by the rows moving in
the daemon's store rather than by a mock.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.cli_namespace import handle_namespace
from trw_memory.daemon import DaemonInfo, DaemonPaths, read_discovery
from trw_memory.daemon.client import DaemonClient
from trw_memory.models.config import MemoryConfig

pytest.importorskip("fastmcp")

_START_DEADLINE_SECONDS = 60.0
_TEST_IDLE_SECONDS = 20.0

OLD = "project:moved-11111111"
NEW = "project:moved-22222222"


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DaemonPaths:
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    return DaemonPaths.resolve()


@pytest.fixture
def daemon(paths: DaemonPaths) -> Iterator[DaemonInfo]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "trw_memory.server",
            "serve",
            "http",
            "--idle-shutdown-seconds",
            str(_TEST_IDLE_SECONDS),
        ],
        env={**os.environ, "TRW_USER_DIR": str(paths.user_memory_dir.parent)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + _START_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail("daemon exited during startup")
            info = read_discovery(paths)
            if info is not None:
                yield info
                return
            time.sleep(0.05)
        pytest.fail("daemon never published a discovery file")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


@pytest.fixture
def client(paths: DaemonPaths) -> DaemonClient:
    return DaemonClient(config=MemoryConfig(memory_daemon_startup_timeout_seconds=1.0), paths=paths)


def _args(action: str, **fields: str) -> argparse.Namespace:
    return argparse.Namespace(namespace_action=action, **fields)


async def test_cli_namespace_rename_merge_and_doctor_use_the_daemon_client(
    paths: DaemonPaths,
    daemon: DaemonInfo,
    client: DaemonClient,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FR05: each verb completes over the daemon, and the CLI opens no store."""
    await client.store("a learning stranded under the old identity", OLD)

    # Count every SQLite open THIS process performs from here on. The verbs
    # below must not add one: an invocation that opened the store would be an
    # extra writer on a store the daemon is meant to own alone.
    opened: list[str] = []
    real_connect = sqlite3.connect

    def _counting_connect(*args: object, **kwargs: object) -> object:
        opened.append(str(args[0]) if args else "")
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", _counting_connect)

    # doctor reports the move and names the repair, without performing it.
    assert await handle_namespace(_args("doctor", namespace=NEW), client=client) == 0
    reported = capsys.readouterr().out
    assert "looks moved or renamed" in reported
    assert f"trw-memory namespace rename {OLD} {NEW}" in reported

    # rename carries the row forward.
    assert await handle_namespace(_args("rename", source=OLD, destination=NEW), client=client) == 0
    assert "renamed" in capsys.readouterr().out

    # doctor is now quiet, which is the observable proof the rename landed.
    assert await handle_namespace(_args("doctor", namespace=NEW), client=client) == 0
    assert "no moved-checkout signal" in capsys.readouterr().out

    # merge is the deliberate two-clones gesture.
    assert await handle_namespace(_args("merge", source=NEW, destination=OLD), client=client) == 0
    assert "merged" in capsys.readouterr().out

    assert opened == [], f"the CLI opened SQLite connections instead of using the daemon: {opened}"


async def test_a_populated_rename_destination_is_reported_without_a_stack_trace(
    daemon: DaemonInfo,
    client: DaemonClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A person running a repair must not have to read a traceback."""
    await client.store("row in old", OLD)
    await client.store("row in new", NEW)

    exit_code = await handle_namespace(_args("rename", source=OLD, destination=NEW), client=client)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Use merge" in captured.err
    assert "Traceback" not in captured.err


async def test_an_unreachable_daemon_is_reported_with_the_remedy(
    paths: DaemonPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    """FR08 reaches the CLI: one line naming the discovery file and the command."""
    import socket

    from trw_memory.daemon import DaemonInfo as Info

    paths.user_memory_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
    paths.discovery.write_text(
        Info(
            pid=os.getpid(),
            url=f"http://127.0.0.1:{dead_port}/mcp",
            token="t",
            started_at="2026-09-03T00:00:00+00:00",
            version="test",
        ).model_dump_json(),
        encoding="utf-8",
    )
    offline = DaemonClient(config=MemoryConfig(memory_daemon_startup_timeout_seconds=1.0), paths=paths)

    exit_code = await handle_namespace(_args("doctor", namespace=NEW), client=offline)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "trw-memory-server serve http" in captured.err
    assert str(paths.discovery) in captured.err
    assert "Traceback" not in captured.err


def test_the_namespace_verbs_are_registered_on_the_cli() -> None:
    """The parser has to reach them, or the tools are unreachable from a terminal."""
    from trw_memory.cli_parser import build_parser

    parser = build_parser()

    renamed = parser.parse_args(["namespace", "rename", OLD, NEW])
    assert (renamed.namespace_action, renamed.source, renamed.destination) == ("rename", OLD, NEW)

    merged = parser.parse_args(["namespace", "merge", OLD, NEW])
    assert merged.namespace_action == "merge"

    doctored = parser.parse_args(["namespace", "doctor"])
    assert (doctored.namespace_action, doctored.namespace) == ("doctor", "")
