"""`trw-memory-server serve stdio` will not serve the machine-wide user store (MEMORY-INPROC-SERVER-NO-ACCESS-CHECK).

A stdio server has no daemon access token, so ``require_namespace_permission`` skips the grant step: every namespace is
reachable, and it would open the user store beside the daemon as a second writer. It is therefore refused when its
resolved store IS the user store the daemon serves. A project-local store (the default ``.memory`` beside the project's
``.trw``) and any explicit non-user store stay allowed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory import server


def _serve(monkeypatch: pytest.MonkeyPatch, argv: list[str] | None = None) -> list[str]:
    ran: list[str] = []
    monkeypatch.setattr(server.mcp, "run", lambda **_k: ran.append("run"))
    server.main(argv or [])
    return ran


def test_stdio_on_the_user_store_is_refused_and_names_the_daemon_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_store = tmp_path / "user" / "memory.db"
    monkeypatch.setattr("trw_memory.daemon.served_store_path", lambda *_a, **_k: user_store)
    monkeypatch.setenv("MEMORY_SINGLE_STORE_PATH", str(user_store))
    monkeypatch.setattr(server.mcp, "run", lambda **_k: pytest.fail("the server must not start"))

    with pytest.raises(SystemExit) as refused:
        server.main([])

    message = str(refused.value.code)  # sys.exit(<text>): the interpreter prints it to stderr and exits 1
    assert "serve http" in message and "daemon" in message and str(user_store) in message


def test_stdio_on_a_project_local_store_still_serves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trw_memory.daemon.served_store_path", lambda *_a, **_k: tmp_path / "user" / "memory.db")
    monkeypatch.setenv("MEMORY_SINGLE_STORE_PATH", str(tmp_path / "project" / "memory.db"))

    assert _serve(monkeypatch) == ["run"]


def test_stdio_with_no_single_store_path_still_serves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trw_memory.daemon.served_store_path", lambda *_a, **_k: tmp_path / "user" / "memory.db")
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)

    assert _serve(monkeypatch) == ["run"]


def test_a_symlink_to_the_user_store_is_still_the_user_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_store = tmp_path / "user" / "memory.db"
    user_store.parent.mkdir()
    user_store.write_bytes(b"")
    alias = tmp_path / "alias.db"
    alias.symlink_to(user_store)
    monkeypatch.setattr("trw_memory.daemon.served_store_path", lambda *_a, **_k: user_store)
    monkeypatch.setenv("MEMORY_SINGLE_STORE_PATH", str(alias))
    monkeypatch.setattr(server.mcp, "run", lambda **_k: pytest.fail("the server must not start"))

    with pytest.raises(SystemExit):
        server.main([])


def test_http_mode_is_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_store = tmp_path / "user" / "memory.db"
    monkeypatch.setattr("trw_memory.daemon.served_store_path", lambda *_a, **_k: user_store)
    monkeypatch.setenv("MEMORY_SINGLE_STORE_PATH", str(user_store))
    started: list[tuple[int | None, float | None]] = []
    monkeypatch.setattr(server, "_serve_http", lambda port, idle: started.append((port, idle)))

    server.main(["serve", "http"])

    assert started == [(None, None)]


def test_a_store_that_cannot_be_resolved_does_not_stop_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A standalone install with no project anchor has no resolvable default store; that is not the user store."""

    from trw_memory.exceptions import StorageRootUnresolvableError

    def no_anchor(*_a: object, **_k: object) -> Path:
        raise StorageRootUnresolvableError("needs a project anchor")

    monkeypatch.setattr("trw_memory.integrations._backend.resolve_backend_db_path", no_anchor)
    monkeypatch.setattr("trw_memory.daemon.served_store_path", lambda *_a, **_k: tmp_path / "user" / "memory.db")

    assert _serve(monkeypatch) == ["run"]


def test_any_other_resolution_failure_does_not_start_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the no-anchor case is waved through; an unexpected failure must never turn into serving (fail closed)."""

    def broken(*_a: object, **_k: object) -> Path:
        raise RuntimeError("config reload failed")

    monkeypatch.setattr("trw_memory.integrations._backend.resolve_backend_db_path", broken)
    monkeypatch.setattr(server.mcp, "run", lambda **_k: pytest.fail("the server must not start"))

    with pytest.raises(RuntimeError):
        server.main([])
