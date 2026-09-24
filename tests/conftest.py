"""Shared test fixtures for the trw-memory test suite.

Test Tiering Philosophy
-----------------------
Tests are classified by their resource usage:

- **unit**: Pure logic — in-memory backends or no I/O at all, no ``tmp_path``.
  Target: <30s for the full unit tier.
- **integration**: Tests that write files, use SQLite on disk, or exercise
  the full ``MemoryClient`` stack with real storage.
- **slow**: Tests loading sentence-transformer models or running full
  consolidation cycles (individual runtime >5s).

To classify a test file:
  1. Uses ``tmp_path`` or real disk backends → integration (default).
  2. Only patches/mocks or uses ``:memory:`` SQLite → unit.
  3. Loads sentence-transformers or runs 100+ dedup cycles → slow.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# MUST be the very first runtime import: ``trw_memory.__init__`` runs the
# pysqlite3 shim which swaps ``sys.modules["sqlite3"]`` to pysqlite3. If
# any test module's ``import sqlite3`` resolves BEFORE the swap, the
# stdlib's exception classes (``sqlite3.DatabaseError``) won't match the
# pysqlite3 classes that production code raises, and ``except`` clauses
# silently miss them. Pulling trw_memory in here at the top of conftest
# guarantees the swap is in place before any test module is loaded.
import trw_memory as _trw_memory_shim_trigger  # noqa: F401
from tests._timing import apply_timing_policy, pytest_runtest_logreport, pytest_sessionfinish  # noqa: F401
from tests._trw_home import isolated_trw_home  # noqa: F401  (canonical copy; see that module's docstring)
from trw_memory.client import MemoryClient
from trw_memory.embeddings import reset_provider_cache
from trw_memory.graph import wait_for_graph_updates
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security.keys import clear_key_cache
from trw_memory.storage._resilient_fetch import reset_bytes_fallback_failures, reset_schema_row_quarantines
from trw_memory.storage.sqlite_backend import SQLiteBackend

# --------------------------------------------------------------------------
# xdist fan-out cap (2026-09-05 OOM incident)
# --------------------------------------------------------------------------
# A 2026-09-05 kernel OOM (191 pytest workers, ~109GB RSS) traced to delegated
# agents running `pytest -n auto` directly in several packages at once,
# bypassing the Makefile's `PYTEST_WORKERS ?= 4` default (which only guards
# `make test-parallel`/`make test-release`, not a raw `pytest` invocation).
# This guard is duplicated verbatim in every package conftest of the
# monorepo it is developed in — no shared test-support module exists across
# these independently-distributed packages (trw-mcp and trw-memory ship to
# PyPI; a new cross-package test dependency is not worth it for 15 lines).
_MAX_XDIST_WORKERS = 4
_ALLOW_WIDE_XDIST_ENV = "TRW_PYTEST_ALLOW_WIDE_XDIST"


def _xdist_fanout_violation(numprocesses: object, allow_wide: bool) -> str | None:
    """Return a violation reason if ``numprocesses`` exceeds the workstation
    cap, else ``None``.

    ``numprocesses`` is ``config.option.numprocesses`` as pytest-xdist sets
    it: ``None`` when ``-n`` was not passed, the literal string ``"auto"`` or
    ``"logical"`` when the caller asked xdist to size itself off the CPU core
    count, or an ``int``/int-like value from an explicit ``-n N``.
    """
    if allow_wide or numprocesses is None:
        return None
    if isinstance(numprocesses, str):
        return f"xdist fan-out -n {numprocesses!r} is uncapped"
    if not isinstance(numprocesses, int):
        return None
    worker_count = numprocesses
    if worker_count > _MAX_XDIST_WORKERS:
        return f"xdist fan-out -n {worker_count} exceeds the cap of {_MAX_XDIST_WORKERS}"
    return None


def pytest_configure(config: pytest.Config) -> None:
    """Refuse a wide xdist fan-out before it OOMs the workstation again."""
    allow_wide = os.environ.get(_ALLOW_WIDE_XDIST_ENV) == "1"
    violation = _xdist_fanout_violation(getattr(config.option, "numprocesses", None), allow_wide)
    if violation is not None:
        pytest.exit(
            f"{violation}. xdist fan-out capped at 4 workers on this "
            "workstation (2026-09-05 OOM); use -n 4 or set "
            "TRW_PYTEST_ALLOW_WIDE_XDIST=1",
            returncode=3,
        )


@pytest.fixture(autouse=True)
def reset_bytes_fallback_counter() -> Iterator[None]:
    """Zero the process-global row-recovery counters before each test.

    ``_resilient_fetch._fallback_metrics.bytes_fallback_failures`` is a
    process-wide counter (it must survive across calls so monitoring can poll
    it). Tests that trigger a hard bytes-mode fallback failure or a cleanly
    decoded schema quarantine would otherwise leak a non-zero value into later
    tests, making counter assertions order-dependent. Resetting *before* each
    test guarantees a known baseline regardless of collection order.
    """
    reset_bytes_fallback_failures()
    reset_schema_row_quarantines()
    yield


@pytest.fixture(autouse=True)
def drain_background_graph_updates() -> Iterator[None]:
    """Finish graph worker threads before pytest closes per-test capture streams."""
    yield
    try:
        wait_for_graph_updates(timeout=1.0)
    except TimeoutError:  # trw-fail-silent-allow: best-effort drain; a fault-injected worker must not hang teardown
        # Graph enrichment is best-effort; tests should not hang if a worker is
        # already blocked on an intentionally fault-injected backend.
        pass


@pytest.fixture(autouse=True)
def reset_embedding_provider_cache() -> Iterator[None]:
    """Empty the process-level embedding-provider cache around each test.

    PRD-CORE-279 gave ``get_local_embedder`` a per-process cache so a long-lived
    server loads the model once. That cache is process-global state, so without
    this a test that patched ``LocalEmbeddingProvider`` would be answered by the
    provider an earlier test built -- which is exactly how
    ``test_returns_none_on_unexpected_exception`` started passing a MagicMock.
    """
    reset_provider_cache()
    yield
    reset_provider_cache()


@pytest.fixture(autouse=True)
def restore_daemon_store_pins() -> Iterator[None]:
    """Undo the store pins a daemon start writes into ``os.environ``.

    ``serve_loopback`` pins MEMORY_STORAGE_PATH and MEMORY_SINGLE_STORE_PATH so
    every ``MemoryConfig()`` in the DAEMON's process resolves the one user-space
    store (PRD-CORE-253 FR01). That is right for a process whose whole life is
    that daemon, and wrong for a test process: the pins outlive the test, and
    every later test in the same xdist worker then resolves its store to a
    deleted temporary directory. Measured: running
    ``tests/test_daemon_lifecycle_signals.py`` before
    ``tests/test_tools_recall_core.py`` made the latter see another namespace's
    rows. Snapshot and restore rather than delete, so a deliberate outer pin
    survives.
    """
    pinned = {name: os.environ.get(name) for name in ("MEMORY_STORAGE_PATH", "MEMORY_SINGLE_STORE_PATH")}
    try:
        yield
    finally:
        for name, value in pinned.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(autouse=True)
def reset_process_global_runtime_caches() -> Iterator[None]:
    """Clear the process-lifetime caches the served tool paths populate.

    Three module-level caches survive a test by design, because in production
    they are per-PROCESS and the process is a server: the tier-manager cache
    (each entry owns an open warm-tier SQLite connection), the canary state
    (whose ``failed`` flag halts recall), and the offload worker pool. In a test
    process the same lifetime is cross-test leakage, and the tier cache is keyed
    on the configured storage path, so two tests using different temporary
    stores at the same configured path resolve one store's entry ids against
    the other's backend.
    """
    _clear_runtime_caches()
    yield
    _clear_runtime_caches()


def _clear_runtime_caches() -> None:
    from trw_memory.daemon._offload import shutdown_offload_pool
    from trw_memory.lifecycle.tiers._runtime import reset_tier_manager_cache
    from trw_memory.security._runtime_canary import CANARY_STATE

    reset_tier_manager_cache()
    CANARY_STATE.clear()
    shutdown_offload_pool()


@pytest.fixture(autouse=True)
def clear_master_key_cache_fixture() -> Iterator[None]:
    clear_key_cache()
    yield
    clear_key_cache()


@pytest.fixture(autouse=True)
def restore_logging_state() -> Iterator[None]:
    """Save+restore global structlog and stdlib-logging config around each test.

    ``trw_memory._logging.configure_logging`` mutates *process-global* state:
    it calls ``logging.basicConfig(force=True)`` (which tears out every root
    handler — including pytest's ``LogCaptureHandler`` — and pins the root
    level) and ``structlog.configure(...)``. Any test that invokes it (e.g.
    the CLI tests) leaves that configuration in place, so a later test relying
    on ``caplog`` (e.g. ``test_id_gen_collision_log``) can silently miss DEBUG
    records: the raised root level filters propagated records before pytest's
    re-attached handler ever sees them. That makes ``pytest -x`` order-dependent.

    Snapshotting structlog's config plus the root logger's handler list and
    level, then restoring both on teardown, makes the suite deterministic
    regardless of collection order without weakening any assertion.
    """
    import logging

    import structlog

    saved_structlog = structlog.get_config()
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        structlog.configure(**saved_structlog)
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


@pytest.fixture(autouse=True)
def _hermetic_keyring_backend() -> Iterator[None]:
    """Install an in-memory keyring backend for the duration of each test.

    The public-mirror CI installs the ``keyring`` package but the headless
    GitHub runner has NO OS keyring backend available (no SecretStorage /
    kwallet / macOS Keychain), so any code path that stores or reads a master
    key via ``keyring.set_password`` raises
    ``keyring.errors.NoKeyringError: No recommended backend was available``.
    In the monorepo dev box a real backend (or none) is present and masks this.

    Pinning a process-local in-memory backend makes the keyring path hermetic
    and deterministic: tests that expect a DIFFERENT raise downstream (e.g.
    ``test_local_mode_raises_when_sqlite_encryption_requested`` asserting the
    SQLCipher-driver ``EncryptionUnavailableError``) reach their intended
    assertion instead of being pre-empted by the keyring-store failure.

    This is a test-only fixture — it does NOT add ``keyrings.alt`` (or any
    other backend) to the runtime dependency set.
    """
    try:  # keyring is a test/CI dep, not a hard runtime dep — fail open if absent.
        import keyring
        import keyring.backend
        from keyring.errors import PasswordDeleteError
    except (
        Exception
    ):  # pragma: no cover  # trw-fail-silent-allow: keyring is an optional test dep; fixture is a no-op without it
        yield
        return

    class _InMemoryKeyring(keyring.backend.KeyringBackend):  # type: ignore[misc]
        """A minimal RAM-only keyring backend for hermetic tests."""

        priority = 1.0  # type: ignore[assignment]

        def __init__(self) -> None:
            super().__init__()
            self._store: dict[tuple[str, str], str] = {}

        def get_password(self, service: str, username: str) -> str | None:
            return self._store.get((service, username))

        def set_password(self, service: str, username: str, password: str) -> None:
            self._store[(service, username)] = password

        def delete_password(self, service: str, username: str) -> None:
            try:
                del self._store[(service, username)]
            except KeyError as exc:
                raise PasswordDeleteError("not found") from exc

    previous = keyring.get_keyring()
    keyring.set_keyring(_InMemoryKeyring())
    try:
        yield
    finally:
        keyring.set_keyring(previous)


@pytest.fixture(autouse=True)
def _sec001_hermetic_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC-001 anchor must not depend on an ambient ``.trw`` found by the cwd-walk
    (present in the monorepo, ABSENT in the standalone published package / public-mirror
    CI). Pin ``TRW_DIR`` to a hermetic per-test anchor so MemoryClient/security startup
    resolves via ``_discover_anchor`` step 2 instead of the cwd-walk. Anchor-resolution
    tests that exercise the cwd-walk / step-3 / no-anchor branches override or clear this
    with their own ``patch.dict(os.environ, {"TRW_DIR": ...})``.

    The anchor lives in a dedicated ``_sec001_anchor`` subdir of ``tmp_path`` (NOT
    ``tmp_path / ".trw"``) so it never collides with tests that create their own
    ``tmp_path / ".trw"`` (e.g. the cwd-walk / path-resolution anchor tests that share
    the same function-scoped ``tmp_path``).
    """
    anchor = tmp_path / "_sec001_anchor" / ".trw"
    anchor.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRW_DIR", str(anchor))


# ---------------------------------------------------------------------------
# SQLiteBackend fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    """Return an initialized SQLiteBackend using a temp-dir database.

    Use this in integration tests that need a real disk-backed store.
    For unit tests that need a backend, use ``sqlite_memory_backend`` instead.
    """
    db = SQLiteBackend(tmp_path / "test.db")
    yield db
    db.close()


@pytest.fixture()
def sqlite_memory_backend() -> Iterator[SQLiteBackend]:
    """Return an initialized in-memory SQLiteBackend.

    Use this in unit tests — no filesystem I/O, safe for the unit tier.
    """
    db = SQLiteBackend(Path(":memory:"))
    yield db
    db.close()


# ---------------------------------------------------------------------------
# MemoryClient fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Return a MemoryClient backed by a SQLite store in ``tmp_path``.

    Sets ``MEMORY_STORAGE_PATH`` and ``MEMORY_STORAGE_BACKEND`` env vars so
    the client does not accidentally write to the real user store.
    """
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
def yaml_memory_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Return a MemoryClient backed by a YAML store in ``tmp_path``."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "yaml_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
def yaml_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "yaml_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
    return MemoryClient(namespace="default", mode="local")


# ---------------------------------------------------------------------------
# MemoryEntry factory helpers
# ---------------------------------------------------------------------------


def make_entry(
    *,
    entry_id: str = "M-001",
    content: str = "test content",
    detail: str = "",
    tags: list[str] | None = None,
    importance: float = 0.5,
    q_value: float = 0.5,
    q_observations: int = 0,
    recurrence: int = 1,
    access_count: int = 0,
    source: str = "agent",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    created_at: datetime | None = None,
    last_accessed_at: datetime | None = None,
    namespace: str = "default",
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    """Create a ``MemoryEntry`` with sensible defaults.

    This is the canonical factory for unit tests — it avoids repetitive
    per-file ``_make_entry`` / ``_entry`` helpers that were duplicated
    across 10+ test files.

    Example::

        entry = make_entry(content="use absolute paths", tags=["gotcha"])
        assert entry.importance == 0.5
    """
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        tags=tags or [],
        importance=importance,
        q_value=q_value,
        q_observations=q_observations,
        recurrence=recurrence,
        access_count=access_count,
        source=source,  # type: ignore[arg-type]  # validator coerces str to Literal
        status=status,
        created_at=created_at or now,
        last_accessed_at=last_accessed_at or now,
        namespace=namespace,
        metadata=metadata or {},
    )


def make_entry_dict(
    *,
    entry_id: str = "M-001",
    content: str = "test content",
    detail: str = "",
    tags: list[str] | None = None,
    importance: float = 0.5,
    q_value: float = 0.5,
    q_observations: int = 0,
    recurrence: int = 1,
    access_count: int = 0,
    source: str = "agent",
    status: str = "active",
    created_at: datetime | None = None,
    last_accessed_at: datetime | None = None,
) -> dict[str, Any]:
    """Create a minimal entry dict matching the ``MemoryEntry`` serialised shape.

    Use this when the code under test expects a plain dict rather than a
    ``MemoryEntry`` model (e.g., scoring functions, lifecycle utilities).
    """
    now = datetime.now(timezone.utc)
    return {
        "id": entry_id,
        "content": content,
        "detail": detail,
        "tags": tags or [],
        "importance": importance,
        "q_value": q_value,
        "q_observations": q_observations,
        "recurrence": recurrence,
        "access_count": access_count,
        "source": source,
        "status": status,
        "created_at": (created_at or now).isoformat(),
        "last_accessed_at": (last_accessed_at or now).isoformat(),
    }


# ---------------------------------------------------------------------------
# MemoryConfig fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory_config(tmp_path: Path) -> MemoryConfig:
    """Return a MemoryConfig pointing to ``tmp_path`` for storage.

    Avoids hardcoding paths in tests that need a config object but don't
    care about the storage backend specifics.
    """
    return MemoryConfig(storage_path=str(tmp_path / "mem"))


# Git exports GIT_DIR / GIT_INDEX_FILE / GIT_WORK_TREE into hooks and some worktree
# contexts. A test that runs `git init --bare` from a directory while GIT_DIR is
# inherited re-initialises the REAL repository as bare: on 2026-09-18 that set
# core.bare=true on the shared checkout and broke every git command until a peer
# session restored it. No test may address the developer's repository implicitly.
for _git_var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_PREFIX"):
    os.environ.pop(_git_var, None)


@pytest.fixture(scope="session")
def provisioned_embedding_cache() -> str:
    """Resolve the model cache before function-scoped HOME isolation.

    Daemon subprocesses must see the same provisioned models as the parent,
    while their HOME and all writable TRW state remain in temporary directories.
    """
    from trw_memory.embeddings._hf_cache import _resolve_cache_dir

    return _resolve_cache_dir() or str(Path.home() / ".cache" / "huggingface" / "hub")


#: Every env var that can switch on or point the Jev decision backend.
_JEV_ENV = ("TRW_JEV_ENABLED", "TRW_JEV_BASE_URL", "TRW_JEV_MODEL", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def _no_live_decision_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """No test may reach the live Jev provider: cost, flakiness, and an API key leaving the machine.

    Clearing the env is sufficient to keep a bare ``judge_from_env(env={})`` off in THIS fixture's
    scope: with no ``project_root``, only the process env and the operator's own
    ``~/.trw/config.yaml`` machine switch are consulted (``resolve_backend_enablement``), and this
    fixture does not touch HOME -- tests that need a hermetic machine-switch layer pin HOME
    themselves (e.g. the decisions module's own ``_isolated_home`` fixture). The transport guard is
    the backstop for a test that builds a live judge directly. It records rather than raises,
    because the judge turns every transport exception into a ``provider_error`` and a raise alone
    would be swallowed; the test fails at teardown instead. Yields the record so the test that
    proves the guard can inspect it.
    """
    import httpx

    for name in _JEV_ENV:
        monkeypatch.delenv(name, raising=False)
    attempted: list[str] = []

    def _guard(real: Any) -> Any:
        def handle(self: Any, request: httpx.Request) -> Any:
            host = request.url.host or ""
            if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
                attempted.append(f"{request.method} {request.url}")
                raise httpx.ConnectError("blocked in tests: live Jev provider", request=request)
            return real(self, request)

        return handle

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _guard(httpx.HTTPTransport.handle_request))
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", _guard(httpx.AsyncHTTPTransport.handle_async_request)
    )
    yield attempted
    assert not attempted, f"test tried to reach the live Jev provider: {attempted}"


def _require_this_checkout(module_name: str, src: Path) -> None:
    """Stop collection when ``module_name`` comes from a DIFFERENT source checkout.

    The shared ``.venv`` holds editable installs of the main checkout, so tests run from a git
    worktree silently exercise main's code unless the worktree's ``src`` is on the path; then
    passes and failures both mislead (three agents hit it on 2026-09-22). An installed wheel
    (no ``src/`` under a ``pyproject.toml``) is not a checkout mix-up and passes.
    """
    import importlib

    origin = Path(importlib.import_module(module_name).__file__ or "").resolve()
    if origin.is_relative_to(src.resolve()):
        return
    other = next((p for p in origin.parents if p.name == "src" and (p.parent / "pyproject.toml").is_file()), None)
    if other is not None:
        pytest.exit(
            f"{module_name} is imported from {other}, not from this checkout's {src}. Run with "
            f"PYTHONPATH={src} (plus any sibling package src you changed), or install this checkout editable.",
            returncode=4,
        )


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_require_this_checkout("trw_memory", _PACKAGE_ROOT / "src")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Host-resource budgets are skipped on CI runners; the timing job measures them (PRD-QUAL-141)."""
    apply_timing_policy(items)


@pytest.fixture(autouse=True)
def _isolated_user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No test sees the machine's real user directory, so a daemon the operator runs never changes an outcome.

    ``trw-memory import`` and ``reembed`` refuse while a daemon record exists
    (PRD-CORE-298 FR02); without this, a developer's live daemon would fail them.
    """
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "trw-user"))
