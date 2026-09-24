"""PRD-CORE-298 FR02 -- the admin CLI has no exemption: its verbs go through the daemon.

A checkout pinned to one namespace runs ``store``, ``recall``, ``export``,
``status``, ``forget`` and ``consolidate`` against a real daemon that also holds
another project's rows, with the SQLite connection factories patched to raise.
Each verb presents only the checkout grant and defaults to the checkout's pin,
not the identity of the directory it now sits in; naming the other project is
refused in one line; and the verbs that still write a store directly --
``import``, ``reembed`` and ``restore`` -- and ``snapshot create``, which copies
every namespace in the file, refuse while the daemon serves.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from trw_memory.cli import main
from trw_memory.daemon import DaemonPaths, mint_grant, write_checkout_grant

from .test_daemon_server import _await_discovery, _spawn_daemon

pytest.importorskip("fastmcp")

_MINE = "project:a-11111111"
_THEIRS = "project:b-22222222"


def _refuse(*_args: object, **_kwargs: object) -> sqlite3.Connection:
    raise AssertionError("the CLI opened a SQLite connection")


def test_the_cli_verbs_reach_only_the_checkout_grant_over_the_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    user_dir = tmp_path / "userhome"
    monkeypatch.setenv("TRW_USER_DIR", str(user_dir))
    paths = DaemonPaths.resolve()
    proc = _spawn_daemon(user_dir)
    try:
        _await_discovery(paths, proc)
        # A moved checkout: its pin, not this directory's identity, names its rows.
        checkout = tmp_path / "moved" / "repo"
        write_checkout_grant(checkout / ".trw", mint_grant(paths, [_MINE]))
        (checkout / ".trw" / "config.yaml").write_text(f"project_namespace: {_MINE}\n", encoding="utf-8")
        monkeypatch.chdir(checkout / ".trw")
        monkeypatch.setattr(sqlite3, "connect", _refuse)
        monkeypatch.setattr("trw_memory.storage._connection.connect", _refuse)

        assert main(["store", "--summary", "daemon row one"]) == 0
        memory_id = capsys.readouterr().out.split("Stored: ")[1].split()[0]
        assert main(["recall", "daemon row", "--namespace", _MINE, "--format", "compact"]) == 0
        assert memory_id in capsys.readouterr().out
        assert main(["export", "--format", "json"]) == 0
        assert [row["id"] for row in json.loads(capsys.readouterr().out)] == [memory_id]
        assert main(["status", "--format", "json"]) == 0
        assert json.loads(capsys.readouterr().out)["entry_count"] == 1
        assert main(["consolidate", "--dry-run", "--namespace", _MINE]) == 0
        assert main(["forget", memory_id, "--namespace", _MINE]) == 0
        capsys.readouterr()

        verbs = (
            ["store", "--summary", "x"],
            ["recall", "x"],
            ["forget", "M-1"],
            ["consolidate"],
            ["export"],
            ["status"],
        )
        for verb in verbs:
            assert main([*verb, "--namespace", _THEIRS]) == 1
            err = capsys.readouterr().err
            assert _THEIRS in err
            assert "Traceback" not in err

        (checkout / "rows.json").write_text("[]", encoding="utf-8")
        direct_writers = (
            ["import", "rows.json"],
            ["reembed"],
            ["restore", "--from-cold", "--db", str(paths.store)],
            ["snapshot", "create", "--db", str(paths.store)],
        )
        for verb in direct_writers:
            assert main(verb) == 1
            assert f"pid {proc.pid}" in capsys.readouterr().err
        assert not [path for path in tmp_path.rglob("*") if "snapshot" in path.name], "a refused snapshot wrote a file"
    finally:
        proc.kill()
        proc.wait(timeout=30)
