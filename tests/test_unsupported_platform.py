"""C12 (7.0 freeze): native Windows is refused with a named error, never an obscure crash.

The directory and secret-file protections open directories as no-follow descriptors,
which native Windows cannot do. Both entries every client and daemon start pass through
refuse first, before any directory is opened or created.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from trw_memory.daemon import DaemonPaths
from trw_memory.daemon._serve import DaemonServeOptions, serve_loopback
from trw_memory.exceptions import ConfigError, UnsupportedPlatformError
from trw_memory.user_paths import UNSUPPORTED_PLATFORM_MESSAGE, resolve_user_memory_dir

_MESSAGE = "trw-memory 4.0 supports macOS and Linux (glibc); native Windows is not supported in this release; use WSL2"


def test_the_message_names_the_supported_platforms_and_the_workaround() -> None:
    assert UNSUPPORTED_PLATFORM_MESSAGE == _MESSAGE
    assert issubclass(UnsupportedPlatformError, ConfigError)


def test_resolving_the_user_memory_dir_on_windows_refuses_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_dir = tmp_path / "user"
    monkeypatch.setenv("TRW_USER_DIR", str(user_dir))
    monkeypatch.setattr(os, "name", "nt")

    with pytest.raises(UnsupportedPlatformError) as refused:
        resolve_user_memory_dir(create=True)

    monkeypatch.undo()
    assert str(refused.value) == _MESSAGE
    assert not user_dir.exists()


def test_a_daemon_start_with_explicit_paths_on_windows_refuses_before_opening_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    paths = DaemonPaths(user_memory_dir=memory)
    options = DaemonServeOptions(port=0, idle_shutdown_seconds=1.0)
    monkeypatch.setattr(os, "name", "nt")

    with pytest.raises(UnsupportedPlatformError) as refused:
        asyncio.run(serve_loopback(options, paths=paths))

    monkeypatch.undo()
    assert str(refused.value) == _MESSAGE
    assert not memory.exists()


def test_a_daemon_client_with_explicit_paths_on_windows_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.daemon.client import DaemonClient

    paths = DaemonPaths(user_memory_dir=tmp_path / "memory")
    monkeypatch.setattr(os, "name", "nt")

    with pytest.raises(UnsupportedPlatformError) as refused:
        DaemonClient("token", paths=paths)

    monkeypatch.undo()
    assert str(refused.value) == _MESSAGE


def test_the_server_entry_point_on_windows_refuses_before_parsing_or_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory import server

    monkeypatch.setattr(server.mcp, "run", lambda *_a, **_k: pytest.fail("served on an unsupported platform"))
    monkeypatch.setattr(os, "name", "nt")

    with pytest.raises(UnsupportedPlatformError) as refused:
        server.main([])

    monkeypatch.undo()
    assert str(refused.value) == _MESSAGE
