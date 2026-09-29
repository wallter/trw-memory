"""A DEFAULT store lives beside the config that governs it -- never in an arbitrary cwd.

``MemoryConfig().storage_path`` is the relative ``.memory``. Every ``config or MemoryConfig()``
fallback used to root a store (and its derived security state) wherever the process ran. Now a
config records the ``.trw`` it loaded ``config.yaml`` from (``source_trw_dir``: the nearest one at
or above the cwd, skipping HOME's machine tier and the temp roots); a default store is
``<that project>/.memory``, its derived security files ``<that .trw>/security``, and the contact
switch is read from that same ``.trw``. So RBAC, sync and the kill switch always govern the store
they sit beside, from the project root or any subdirectory. ``TRW_DIR``/``TRW_PROJECT_ROOT`` do
not steer a default store; with no project a default write is refused before touching disk. An
explicit path is unchanged. One direct test per arm (Q7(25)).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import AuthorizationError, StorageRootUnresolvableError
from trw_memory.exceptions import MemoryError as TrwMemoryError
from trw_memory.integrations._backend import create_backend
from trw_memory.lifecycle.tiers._runtime import get_tier_manager
from trw_memory.models.config import MemoryConfig
from trw_memory.security.runtime import append_audit_event, enforce_write_rate_limit
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.store import memory_store_impl

_LOCKED = 'rbac_enabled: true\nnamespace_roles:\n  "project:locked": reader\n'
_SYNC = "sync_enabled: true\nplatform_url: https://platform.example.com\nplatform_api_key: k\n"


def _listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.fixture(autouse=True)
def _no_env_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TRW_PROJECT_ROOT", "MEMORY_STORAGE_PATH", "MEMORY_SINGLE_STORE_PATH", "TRW_PLATFORM_CONTACT_ENABLED"):
        monkeypatch.delenv(name, raising=False)


def _project(root: Path, config: str = "") -> Path:
    (root / ".trw").mkdir(parents=True)
    (root / "src").mkdir()
    if config:
        (root / ".trw" / "config.yaml").write_text(config, encoding="utf-8")
    return root


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    return _project(tmp_path / "proj")


@pytest.fixture()
def unanchored_cwd(tmp_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No ``.trw`` at or above the cwd (``project`` is a sibling, watched for stray writes)."""
    cwd = tmp_path / "bare"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def _client_default() -> None:
    MemoryClient(namespace="default", mode="local")


def _store_default(outside: Path) -> Callable[[], object]:
    def run() -> object:
        with SQLiteBackend(outside / "memory.db") as backend:
            return memory_store_impl("a learning", "default", backend=backend, session_id="s1")

    return run


_DEFAULT_WRITERS: dict[str, Callable[[Path], Callable[[], object]]] = {
    "MemoryClient": lambda _o: _client_default,
    "memory_store_impl": _store_default,
    "tier_manager": lambda _o: lambda: get_tier_manager(MemoryConfig(), "default"),
    "adapter_create_backend": lambda _o: lambda: create_backend("default"),
    "rate_limit_state": lambda _o: (
        lambda: enforce_write_rate_limit(
            MemoryConfig(), session_id="s1", actor="a", namespace="default", entry_id="M-1"
        )
    ),
    "audit_log": lambda _o: lambda: append_audit_event(MemoryConfig(), "store_rejected", namespace="default"),
}


@pytest.mark.parametrize("writer", sorted(_DEFAULT_WRITERS))
def test_arm_refuse_every_default_writer_without_a_project(
    writer: str, unanchored_cwd: Path, project: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "explicit-backend"
    outside.mkdir()
    watched = {root: _listing(root) for root in (unanchored_cwd, project)}

    refusal: TrwMemoryError | None = None
    try:
        _DEFAULT_WRITERS[writer](outside)()
    except TrwMemoryError as exc:
        refusal = exc

    assert {root: _listing(root) for root in watched} == watched
    assert refusal is not None, f"{writer} wrote with no project anchor instead of refusing"
    assert "anchor" in str(refusal)


def test_arm_refuse_resolve_storage_root_directly(unanchored_cwd: Path) -> None:
    from trw_memory._project_anchor import resolve_storage_root

    with pytest.raises(StorageRootUnresolvableError, match="project anchor"):
        resolve_storage_root(MemoryConfig())


async def test_arm_cwd_trw_places_the_default_store_and_security_state_in_the_project(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(project)
    client = MemoryClient(namespace="default", mode="local")
    try:
        await client.store("anchored at the project root")
    finally:
        await client.close()

    assert (project / ".memory" / "default" / "memory.db").is_file()
    assert (project / ".trw" / "security" / "audit.jsonl").is_file()


def test_arm_cwd_trw_anchors_every_derived_security_file_under_the_source_trw(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory._project_anchor import resolve_state_path

    monkeypatch.chdir(project)
    config = MemoryConfig()
    security = project.resolve() / ".trw" / "security"

    assert {
        f: resolve_state_path(config, f) for f in ("audit_log_path", "rate_limit_state_path", "quarantine_path")
    } == {
        "audit_log_path": security / "audit.jsonl",
        "rate_limit_state_path": security / "rate_limits.yaml",
        "quarantine_path": security / "quarantine",
    }


def test_arm_subdir_run_uses_the_ancestor_projects_store_and_security_state(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory._project_anchor import resolve_state_path, resolve_storage_root

    monkeypatch.chdir(project / "src")
    config = MemoryConfig()

    assert config.source_trw_dir == project.resolve() / ".trw"
    assert resolve_storage_root(config) == project.resolve() / ".memory"
    assert resolve_state_path(config, "audit_log_path") == project.resolve() / ".trw" / "security" / "audit.jsonl"


def test_arm_nested_project_uses_the_nearest_trw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory._project_anchor import resolve_storage_root

    outer = _project(tmp_path / "outer", _LOCKED)
    inner = _project(outer / "inner")
    monkeypatch.chdir(inner / "src")
    config = MemoryConfig()

    assert resolve_storage_root(config) == inner.resolve() / ".memory"
    assert config.rbac_enabled is False  # the outer project's policy does not leak into the inner one


@pytest.mark.parametrize("machine_dir", ["HOME", "TEMP"])
def test_arm_home_and_temp_root_trw_are_not_projects(
    machine_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HOME's ``.trw`` is the machine tier; a temp root's is a stray. Neither anchors a store."""
    import tempfile

    from trw_memory._project_anchor import resolve_storage_root

    root = tmp_path / machine_dir.lower()
    (root / ".trw").mkdir(parents=True)
    (root / "work").mkdir()
    if machine_dir == "HOME":
        monkeypatch.setenv("HOME", str(root))
    else:
        monkeypatch.setattr(tempfile, "tempdir", str(root))
    monkeypatch.chdir(root / "work")

    with pytest.raises(StorageRootUnresolvableError):
        resolve_storage_root(MemoryConfig())
    assert not (root / ".memory").exists()


@pytest.mark.parametrize("anchor_var", ["TRW_DIR", "TRW_PROJECT_ROOT"])
def test_arm_env_anchors_do_not_steer_the_default_store(
    anchor_var: str, unanchored_cwd: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory._project_anchor import resolve_storage_root

    named = _project(tmp_path / "named")
    monkeypatch.setenv(anchor_var, str(named / ".trw" if anchor_var == "TRW_DIR" else named))

    with pytest.raises(StorageRootUnresolvableError):
        resolve_storage_root(MemoryConfig())
    assert not (named / ".memory").exists()


@pytest.mark.parametrize("explicit", ["relative/store", ".memory", "ABSOLUTE"])
def test_arm_explicit_constructor_path_is_used_unchanged(explicit: str, unanchored_cwd: Path, tmp_path: Path) -> None:
    from trw_memory._project_anchor import resolve_storage_root

    path = str(tmp_path / "abs-store") if explicit == "ABSOLUTE" else explicit

    assert resolve_storage_root(MemoryConfig(storage_path=path)) == Path(path)


def test_arm_explicit_env_storage_path_is_used_unchanged(unanchored_cwd: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory._project_anchor import resolve_storage_root

    monkeypatch.setenv("MEMORY_STORAGE_PATH", "from-env")

    assert resolve_storage_root(MemoryConfig()) == Path("from-env")


def test_the_anchor_is_the_config_objects_own_source_by_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory._project_anchor import resolve_storage_root

    a, b = _project(tmp_path / "a", _LOCKED), _project(tmp_path / "b")
    monkeypatch.chdir(a)
    config = MemoryConfig()
    monkeypatch.chdir(b)

    assert config.rbac_enabled is True  # a's policy ...
    assert resolve_storage_root(config) == a.resolve() / ".memory"  # ... governs a's store
    assert resolve_storage_root(config.model_copy(update={"embedding_dim": 8})) == a.resolve() / ".memory"


def test_revalidating_a_config_keeps_its_recorded_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review r3 P1: ``model_validate(config)`` re-derived the source, so a's policy governed b's store."""
    from trw_memory._project_anchor import resolve_storage_root

    a, b = _project(tmp_path / "a", _LOCKED), _project(tmp_path / "b")
    monkeypatch.chdir(a)
    config = MemoryConfig()
    monkeypatch.chdir(b)

    again = MemoryConfig.model_validate(config)

    assert again.rbac_enabled is True
    assert resolve_storage_root(again) == a.resolve() / ".memory"


# --- the contact switch is read from the governing config's .trw -------------------------------


@pytest.fixture()
def post_calls() -> Iterator[MagicMock]:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 201
    response.json.return_value = {"id": "R-1"}
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = response
    with patch("trw_memory.sync._remote_publish.httpx.Client", return_value=client):
        yield client.post


def _publish_with_a_fresh_config() -> None:
    from trw_memory.sync._remote_publish import _publish_payload_result

    _publish_payload_result({"source_learning_id": "M-1"}, MemoryConfig(), entry_id="M-1")


def test_r2_unrelated_cwd_with_trw_dir_makes_zero_posts(
    unanchored_cwd: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, post_calls: MagicMock
) -> None:
    named = _project(tmp_path / "named", _SYNC + "platform_contact_enabled: false\n")
    monkeypatch.setenv("TRW_DIR", str(named / ".trw"))

    _publish_with_a_fresh_config()

    assert post_calls.call_count == 0


@pytest.mark.parametrize(("contact", "posts"), [("false", 0), ("true", 1)])
def test_contact_switch_is_read_from_the_governing_trw_not_trw_project_root(
    contact: str, posts: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, post_calls: MagicMock
) -> None:
    governing = _project(tmp_path / "governing", _SYNC + f"platform_contact_enabled: {contact}\n")
    elsewhere = _project(
        tmp_path / "elsewhere", f"platform_contact_enabled: {'false' if contact == 'true' else 'true'}\n"
    )
    monkeypatch.setenv("TRW_PROJECT_ROOT", str(elsewhere))
    monkeypatch.chdir(governing)

    _publish_with_a_fresh_config()

    assert post_calls.call_count == posts


@pytest.mark.parametrize(("contact", "posts"), [("false", 0), ("true", 1)])
def test_contact_switch_from_a_subdir_is_read_from_the_ancestor_project(
    contact: str, posts: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, post_calls: MagicMock
) -> None:
    governing = _project(tmp_path / "governing", _SYNC + f"platform_contact_enabled: {contact}\n")
    monkeypatch.chdir(governing / "src")

    _publish_with_a_fresh_config()

    assert post_calls.call_count == posts


# --- RBAC through the registered memory_store tool ---------------------------------------------


async def _call_registered_memory_store(namespace: str) -> object:
    from fastmcp import FastMCP

    from trw_memory.tools.store import register_store_tool

    mcp = FastMCP("anchor-rbac")
    register_store_tool(mcp)
    tool = await mcp.get_tool("memory_store")
    return await tool.fn(content="a locked-namespace write", namespace=namespace)  # type: ignore[attr-defined]


async def test_registered_memory_store_from_the_root_enforces_rbac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path / "locked", _LOCKED)
    monkeypatch.chdir(root)

    with pytest.raises(AuthorizationError, match="project:locked"):
        await _call_registered_memory_store("project:locked")


@pytest.mark.parametrize("env", ["none", "TRW_PROJECT_ROOT"])
async def test_registered_memory_store_from_src_enforces_the_root_projects_rbac(
    env: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The r1 scenario: from ``src/`` the root config's reader-only namespace is enforced, not bypassed."""
    root = _project(tmp_path / "locked", _LOCKED)
    monkeypatch.chdir(root / "src")
    if env == "TRW_PROJECT_ROOT":
        monkeypatch.setenv("TRW_PROJECT_ROOT", str(root))

    with pytest.raises(AuthorizationError, match="project:locked"):
        await _call_registered_memory_store("project:locked")

    assert not (root / "src" / ".memory").exists()


def test_config_readers_never_refuse_without_a_project(unanchored_cwd: Path) -> None:
    """Refusal lives at the write waist: reading non-path fields needs no anchor."""
    config = MemoryConfig()

    assert config.source_trw_dir is None
    assert config.embedding_model
    assert _listing(unanchored_cwd) == []
