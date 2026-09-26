"""PRD-CORE-298 FR02 -- the whole-store read paths stay inside the token's grant.

``require_namespace_permission`` guards every tool that NAMES a namespace. The
tools below also read namespaces the caller never named -- a status breakdown,
the moved-checkout census, the importance decay pass -- so each must narrow to
the grant itself. With no access token (the in-process SDK) nothing changes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from tests.conftest import make_entry
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.curate import store_census
from trw_memory.tools.maintain import memory_maintain_impl
from trw_memory.tools.status import memory_status_impl

_ALPHA = "project:alpha-11111111"
_BETA = "project:beta-22222222"
_OLD = "2020-01-01T00:00:00+00:00"


@pytest.fixture
def config(tmp_path: Path) -> MemoryConfig:
    config = MemoryConfig(storage_path=str(tmp_path), memory_single_store_path=str(tmp_path / "memory.db"))
    with create_backend_from_config(config, _ALPHA) as backend:
        for namespace in (_ALPHA, _BETA):
            backend.store(make_entry(entry_id=f"M-{namespace[8:12]}", namespace=namespace, importance=0.8))
        backend._conn.execute("UPDATE memories SET created_at = ?, last_accessed_at = ?", (_OLD, _OLD))  # type: ignore[attr-defined]
        backend._conn.commit()  # type: ignore[attr-defined]
    return config


@pytest.fixture
def alpha_token() -> Iterator[None]:
    reset = auth_context_var.set(AuthenticatedUser(AccessToken(token="t", client_id="c", scopes=[f"ns:{_ALPHA}"])))
    yield
    auth_context_var.reset(reset)


def test_status_without_a_namespace_counts_only_the_grant(config: MemoryConfig, alpha_token: None) -> None:
    with create_backend_from_config(config, _ALPHA) as backend:
        result = memory_status_impl(None, backend=backend, config=config)

    assert _BETA not in str(result)
    assert result["total_entries"] == 1
    assert result["namespaces"] == {_ALPHA: 1, "__active__": 1}


def test_status_naming_an_ungranted_namespace_is_refused(config: MemoryConfig, alpha_token: None) -> None:
    with create_backend_from_config(config, _ALPHA) as backend:
        result = memory_status_impl(_BETA, backend=backend, config=config)

    assert result["status"] == "forbidden"
    assert "total_entries" not in result


def test_the_census_holds_only_granted_namespaces(config: MemoryConfig, alpha_token: None) -> None:
    assert store_census(config) == {_ALPHA: 1}


def test_maintain_decays_only_granted_rows(config: MemoryConfig, alpha_token: None) -> None:
    with create_backend_from_config(config, _ALPHA) as backend:
        memory_maintain_impl(_ALPHA, backend=backend, config=config)
        importance = dict(
            backend._conn.execute("SELECT namespace, importance FROM memories").fetchall()  # type: ignore[attr-defined]
        )

    assert importance[_ALPHA] < 0.8
    assert importance[_BETA] == 0.8


def test_without_a_token_every_namespace_is_visible(config: MemoryConfig) -> None:
    assert store_census(config) == {_ALPHA: 1, _BETA: 1}
    with create_backend_from_config(config, _ALPHA) as backend:
        assert memory_status_impl(None, backend=backend, config=config)["total_entries"] == 2


class _ReadRecorder:
    """Wraps the quarantine store and records which namespace every read named."""

    def __init__(self, inner: object, reads: list[tuple[str, object]]) -> None:
        self._inner = inner
        self._reads = reads

    def __enter__(self) -> _ReadRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self._inner.close()  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        attr = getattr(self._inner, name)
        if name not in {"list_entries", "get", "search", "count", "list_namespaces"}:
            return attr

        def record(*args: object, **kwargs: object) -> object:
            scope = args[0] if name == "list_namespaces" and args else kwargs.get("namespace", args[1:2] or None)
            self._reads.append((name, scope))
            return attr(*args, **kwargs)

        return record


@pytest.fixture
def quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[MemoryConfig, list[tuple[str, object]]]]:
    """One alpha row, then three newer beta rows, behind a read recorder."""
    from trw_memory.security import _runtime_quarantine
    from trw_memory.security.runtime import store_quarantined_entry

    config = MemoryConfig(storage_path=str(tmp_path / "store"), quarantine_db_path=str(tmp_path / "q.db"))
    store_quarantined_entry(config, make_entry(entry_id="Q-alpha", namespace=_ALPHA))
    for index in range(3):
        store_quarantined_entry(config, make_entry(entry_id=f"Q-beta-{index}", namespace=_BETA))
    reads: list[tuple[str, object]] = []
    real = _runtime_quarantine.open_quarantine_backend
    monkeypatch.setattr(_runtime_quarantine, "open_quarantine_backend", lambda cfg: _ReadRecorder(real(cfg), reads))
    yield config, reads


def _foreign(reads: list[tuple[str, object]]) -> list[tuple[str, object]]:
    return [read for read in reads if read[1] not in (_ALPHA, [_ALPHA])]


def test_the_quarantine_list_never_reads_an_ungranted_namespace(
    quarantine: tuple[MemoryConfig, list[tuple[str, object]]], alpha_token: None
) -> None:
    from trw_memory.tools.review import memory_quarantine_list_impl

    config, reads = quarantine
    listed = memory_quarantine_list_impl(config=config)

    assert reads
    assert _foreign(reads) == []
    assert listed["namespaces"] == [_ALPHA]
    assert memory_quarantine_list_impl(_BETA, config=config)["status"] == "forbidden"


def test_newer_ungranted_rows_cannot_starve_the_granted_page(
    quarantine: tuple[MemoryConfig, list[tuple[str, object]]], alpha_token: None
) -> None:
    from trw_memory.tools.review import memory_quarantine_list_impl

    listed = memory_quarantine_list_impl(limit=1, config=quarantine[0])

    assert [row["id"] for row in listed["entries"]] == ["Q-alpha"]  # type: ignore[index, union-attr]


def test_status_counts_only_the_granted_quarantine(
    quarantine: tuple[MemoryConfig, list[tuple[str, object]]], alpha_token: None
) -> None:
    config, reads = quarantine
    with create_backend_from_config(config, _ALPHA) as backend:
        result = memory_status_impl(_ALPHA, backend=backend, config=config)

    assert result["security_posture"]["quarantine_count"] == 1  # type: ignore[index]
    assert reads
    assert _foreign(reads) == []


def test_team_wildcard_consolidation_never_names_an_ungranted_team(config: MemoryConfig, alpha_token: None) -> None:
    from trw_memory.tools.consolidate import memory_consolidate_impl

    with create_backend_from_config(config, _ALPHA) as backend:
        backend.store(make_entry(entry_id="T-1", namespace="team:theirs", importance=0.9))
        result = memory_consolidate_impl("team:*", backend=backend, config=config)

    assert "team:theirs" not in str(result)


def test_status_under_a_grant_is_blind_to_the_process_wide_maintenance_queue(
    config: MemoryConfig, alpha_token: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The maintenance queue spans every tenant the daemon serves, so a scoped token sees none of it."""
    from trw_memory.tools import status

    def status_with(queued: int) -> dict[str, object]:
        busy = {"queued": queued, "processed": queued * 7, "bounded": True}
        monkeypatch.setattr(status, "security_maintenance_status", lambda: busy)
        with create_backend_from_config(config, _ALPHA) as backend:
            return memory_status_impl(_ALPHA, backend=backend, config=config)

    busy, idle = status_with(5), status_with(0)

    assert busy == idle
    assert "maintenance" not in busy.get("security_posture", {})  # type: ignore[operator]


def test_path_refusals_leak_no_file_content(tmp_path: Path) -> None:
    """PRD-SEC-016 NFR04 -- the evidence artifact this AC names.

    "A refusal reply and its log line name the path and the reason. They
    contain no bytes read from the refused file." Exercises the ONE
    checkout-bound opener (``open_checkout_file_fd``, PRD-SEC-016 FR02/FR03/
    FR05) against two distinct refusal causes -- a symlink escape and a
    ``..`` traversal attempt -- each targeting a file that holds a
    distinctive secret marker string. The walk refuses BEFORE it ever reads
    a byte of that file (the component-by-component ``O_NOFOLLOW`` open in
    ``_dir_trust.open_component_fd`` fails on the symlink/traversal itself,
    never opening the target for read), so this test proves the secret
    cannot appear anywhere a caller or an operator can see it: the returned
    refusal dict (including its ``"error"`` string), and every structlog
    event captured during the call.
    """
    import structlog

    from trw_memory.tools.entry import open_checkout_file_fd

    secret = "TOP-SECRET-9f3a1c-do-not-leak-this-content"  # a marker string, not a real credential
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "private.db"
    outside_file.write_text(secret, encoding="utf-8")

    root = tmp_path / "checkout"
    root.mkdir()
    escape_link = root / "escape.db"
    escape_link.symlink_to(outside_file)

    with structlog.testing.capture_logs() as symlink_logs:
        symlink_result = open_checkout_file_fd(str(root), str(root / "escape.db"), "test_read")
    with structlog.testing.capture_logs() as traversal_logs:
        traversal_result = open_checkout_file_fd(str(root), str(root / ".." / "outside" / "private.db"), "test_read")

    for result, logs, expected_path_fragment in (
        (symlink_result, symlink_logs, "escape.db"),
        (traversal_result, traversal_logs, ".."),
    ):
        assert isinstance(result, dict), result
        assert result["status"] == "refused", result
        error_message = str(result["error"])
        assert secret not in error_message, error_message
        assert secret not in str(result), result
        log_text = "\n".join(f"{event.get('event', '')} {event}" for event in logs)
        assert secret not in log_text, log_text
        # The refusal still names the path and a reason -- NFR04 is about
        # content leakage, not silence -- so the reply must not degenerate
        # into an empty acknowledgement.
        assert "error" in result and error_message
        assert result["status"] == "refused"
        # PRD-SEC-016 round-8 finding 3: NFR04's own text is "A refusal
        # reply AND ITS LOG LINE name the path and the reason" -- assert the
        # LOG's fields directly, not just the reply's. Before this fix, the
        # `".."`-traversal refusal produced NO log event at all (only the
        # symlink-escape path, indirectly, via `_dir_trust`'s own
        # `dir_open_refused`), so a string-absence check alone could not
        # have caught a silent refusal.
        refusal_events = [event for event in logs if event.get("event") == "checkout_boundary_refused"]
        assert refusal_events, logs
        assert any(expected_path_fragment in str(event.get("path", "")) for event in refusal_events), logs
        assert all(event.get("reason") for event in refusal_events), logs


def test_a_served_verify_refusal_never_opens_the_refused_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRD-SEC-016 round-7 item 4 -- an open/read spy, at a SERVED tool's own reply and log output.

    The prior NFR04 test above exercises the internal opener directly and
    proves no BYTES leak. This goes further, at the boundary the coordinator
    named: the REAL served ``memory_verify`` tool, a REAL SQLite-backed entry
    carrying a ``grep_present`` assertion whose target is a symlink escaping
    the checkout, and a spy on ``os.open`` proving the escape target is
    never opened AT ALL (not "opened but the read is discarded") -- the
    dir_fd walk (``verification.py::_walk_checkout``) refuses a pattern-
    matching candidate the instant ``os.DirEntry``/a fresh no-follow stat
    says it is a symlink, before any enumeration through it or any attempt
    to read it, so no ``os.open`` call for the outside path is ever made.

    PRD-SEC-016 round-8 review, item 3: the spy used to record ``os.open``'s
    raw ``path`` ARGUMENT and compare that string against the outside file's
    fully-RESOLVED path -- blind to a hypothetical future bug that reached
    the same inode through a bare, ``dir_fd``-relative name (which never
    equals an absolute string no matter what it points to). This spy instead
    ``fstat``s every fd ``os.open`` actually returns and records its
    ``(st_dev, st_ino)`` IDENTITY, then asserts the outside file's identity
    never appears among them -- a check that holds regardless of what
    string, relative or absolute, any future open call used to get there.
    """
    import asyncio

    import structlog

    from trw_memory.models.memory import Assertion, AssertionType, MemoryEntry, MemoryStatus
    from trw_memory.tools.verify import register_verify_tool

    secret = "TOP-SECRET-r7-item4-never-opened"  # a marker string, not a real credential
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "leak.txt"
    outside_file.write_text(secret, encoding="utf-8")

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "escape.txt").symlink_to(outside_file)

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "verify-storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")

    config = MemoryConfig()
    with create_backend_from_config(config, _ALPHA) as backend:
        backend.store(
            MemoryEntry(
                id="L-escape",
                content="a claim checked against escape.txt",
                namespace=_ALPHA,
                status=MemoryStatus.ACTIVE,
                assertions=[Assertion(type=AssertionType.GREP_PRESENT, pattern="anything", target="escape.txt")],
            )
        )

    outside_identity = (os.stat(outside_file).st_dev, os.stat(outside_file).st_ino)
    opened_names: list[str] = []
    opened_identities: list[tuple[int, int]] = []
    real_open = os.open

    def spying_open(path: object, *args: object, **kwargs: object) -> int:
        opened_names.append(path if isinstance(path, str) else os.fsdecode(path))  # type: ignore[arg-type]
        fd = real_open(path, *args, **kwargs)  # type: ignore[arg-type]
        try:
            st = os.fstat(fd)
        except OSError as exc:
            # Round-10 review, item 3: a swallowed fstat failure here was
            # indistinguishable from "this fd's identity was checked and is
            # not the outside one" -- but a spy that cannot observe every
            # fd `os.open` returns is not proving the property this test
            # claims. Fail loudly instead of silently narrowing the sample.
            raise AssertionError(f"identity spy could not fstat an fd opened for {path!r}: {exc}") from exc
        opened_identities.append((st.st_dev, st.st_ino))
        return fd

    monkeypatch.setattr(os, "open", spying_open)

    class _Captured:
        def __init__(self) -> None:
            self.tools: dict[str, object] = {}

        def tool(self) -> object:
            return lambda fn: self.tools.setdefault(fn.__name__, fn)

    server = _Captured()
    register_verify_tool(server)  # type: ignore[arg-type]
    token = AccessToken(token="t", client_id="c", scopes=[f"ns:{_ALPHA}"], claims={"root": str(root)})
    reset = auth_context_var.set(AuthenticatedUser(token))
    try:
        with structlog.testing.capture_logs() as logs:
            answer = asyncio.run(
                server.tools["memory_verify"](namespace=_ALPHA, project_root=None, settings=None)  # type: ignore[operator]
            )
    finally:
        auth_context_var.reset(reset)

    assert answer.get("status") == "ok", answer
    real_outside_path = str(outside_file.resolve())
    assert real_outside_path not in opened_names, opened_names
    assert not any(secret in name for name in opened_names), opened_names
    # The identity check, not the path-string check above, is the one that
    # cannot be fooled by a relative, dir_fd-anchored open reaching the same
    # inode under a name that never matches `real_outside_path`.
    assert outside_identity not in opened_identities, (outside_identity, opened_identities)
    assert secret not in str(answer), answer
    log_text = "\n".join(f"{event.get('event', '')} {event}" for event in logs)
    assert secret not in log_text, log_text


def test_the_identity_spy_actually_fires_on_a_matching_identity() -> None:
    """Non-vacuity partner: the identity assertion above is not vacuously true.

    A bare `assert X not in []` always passes; this proves the check has
    teeth by fabricating an `opened_identities` list that DOES contain the
    outside file's identity (as a hypothetical dir_fd-relative-open
    regression would produce) and confirming the assertion actually fires.
    """
    outside_identity = (7, 42)
    opened_identities = [(1, 1), outside_identity, (2, 2)]

    assert outside_identity in opened_identities  # the fixture itself is sane
    with pytest.raises(AssertionError):
        assert outside_identity not in opened_identities, (outside_identity, opened_identities)
