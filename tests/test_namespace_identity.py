"""PRD-CORE-253 FR01 — project identity and the user-space path resolver.

Every test here drives the real resolver against real directories (real `git
init`, real `git worktree add`, real symlinks); nothing about the identity
function is mocked, because the whole point of the requirement is what it
computes from a filesystem it did not create.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.identity import (
    SLUG_MAX_CHARS,
    canonical_project_root,
    project_slug,
    resolve_project_namespace,
)
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.user_paths import resolve_user_memory_dir


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "T", cwd=root)
    (root / "README.md").write_text("x", encoding="utf-8")
    _git("add", "README.md", cwd=root)
    _git("commit", "-qm", "init", cwd=root)
    return root


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_project_identity_is_path_digest_not_basename(tmp_path: Path) -> None:
    """Two checkouts whose basenames differ only by suffix resolve apart.

    The prior identity was ``resolve_project_root().name``, so ``a/repo`` and
    ``b/repo`` collided outright; and a remote-derived key would have merged
    these two, whose remote sets are identical (both empty).
    """
    left = _init_repo(tmp_path / "a" / "trw-framework")
    right = _init_repo(tmp_path / "b" / "trw-framework")

    assert resolve_project_namespace(left) != resolve_project_namespace(right)
    assert project_slug(left) == project_slug(right) == "trw-framework"


def test_a_worktree_resolves_to_the_main_checkouts_namespace(tmp_path: Path) -> None:
    """FR01: the digest input is --git-common-dir, so worktrees share one namespace.

    Fails against a ``--show-toplevel`` derivation, which is what makes this
    test the discriminating one: a toplevel digest splits one repository's
    memory across every branch an agent checks out in parallel.
    """
    main = _init_repo(tmp_path / "trw-framework")
    linked = tmp_path / "trw-framework-frontend-ux"
    _git("worktree", "add", "-q", "-b", "feature", str(linked), cwd=main)

    assert resolve_project_namespace(linked) == resolve_project_namespace(main)
    assert canonical_project_root(linked) == main.resolve()


def test_a_symlinked_checkout_resolves_to_the_same_namespace(tmp_path: Path) -> None:
    """realpath: reaching a checkout through a link is not a different project."""
    real = _init_repo(tmp_path / "real-project")
    link = tmp_path / "linked-project"
    link.symlink_to(real, target_is_directory=True)

    assert resolve_project_namespace(link) == resolve_project_namespace(real)


def test_a_subdirectory_resolves_to_the_repository_namespace(tmp_path: Path) -> None:
    """Entering from a subdirectory must not change which slice you see."""
    root = _init_repo(tmp_path / "proj")
    nested = root / "src" / "deep"
    nested.mkdir(parents=True)

    assert resolve_project_namespace(nested) == resolve_project_namespace(root)


def test_a_second_clone_is_a_distinct_namespace(tmp_path: Path) -> None:
    """Decided in FR01: two clones are two working states, kept apart by default."""
    origin = _init_repo(tmp_path / "origin")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True)

    assert resolve_project_namespace(clone) != resolve_project_namespace(origin)


def test_a_non_git_directory_resolves_without_error(tmp_path: Path) -> None:
    """The fallback is the resolved project root, not a failure."""
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    namespace = resolve_project_namespace(plain)

    assert namespace == f"project:not-a-repo-{namespace.rsplit('-', 1)[1]}"
    assert canonical_project_root(plain) == plain.resolve()


def test_resolution_defaults_to_trw_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no argument the resolver honours TRW_PROJECT_ROOT, then the cwd."""
    root = _init_repo(tmp_path / "envproj")
    monkeypatch.setenv("TRW_PROJECT_ROOT", str(root))

    assert resolve_project_namespace() == resolve_project_namespace(root)


@pytest.mark.parametrize(
    ("directory", "expected_slug"),
    [
        ("My Project!", "my-project"),
        ("....", "root"),
        ("UPPER_snake.case", "upper-snake-case"),  # "_" is outside [a-z0-9-] too
        ("a" * 80, "a" * SLUG_MAX_CHARS),
    ],
)
def test_slug_sanitises_to_the_namespace_grammar(tmp_path: Path, directory: str, expected_slug: str) -> None:
    """Every derived namespace passes validate_namespace unchanged."""
    target = tmp_path / directory
    target.mkdir()

    namespace = resolve_project_namespace(target)

    assert project_slug(target) == expected_slug
    assert namespace.startswith(f"project:{expected_slug}-")
    assert validate_namespace(namespace) == namespace
    assert len(namespace) <= 128


def test_identity_is_stable_across_repeated_resolution(tmp_path: Path) -> None:
    """The digest is a pure function of the canonical root, not of the clock."""
    root = _init_repo(tmp_path / "stable")

    assert resolve_project_namespace(root) == resolve_project_namespace(root)


# ---------------------------------------------------------------------------
# The user-space resolver, promoted into trw-memory (FR01)
# ---------------------------------------------------------------------------


def test_user_memory_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TRW_USER_DIR beats XDG_DATA_HOME beats the home fallback."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "explicit"))
    assert resolve_user_memory_dir() == (tmp_path / "explicit" / "memory").resolve()

    monkeypatch.delenv("TRW_USER_DIR")
    assert resolve_user_memory_dir() == (tmp_path / "xdg" / "trw" / "memory").resolve()

    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert resolve_user_memory_dir() == (tmp_path / "home" / ".trw" / "memory").resolve()


def test_resolver_does_not_create_when_asked_not_to(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A presence probe must not be the thing that creates the directory."""
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "probe"))

    resolved = resolve_user_memory_dir(create=False)

    assert not resolved.exists()


def test_trw_mcp_delegates_to_the_promoted_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR01: 'no second resolver is introduced'.

    trw-mcp keeps its import site but must not carry its own implementation,
    or the daemon and the ceremony server could disagree about where the store
    lives.
    """
    pytest.importorskip("trw_mcp")
    from trw_mcp.state import _user_paths

    assert _user_paths.resolve_user_memory_dir.__module__ == "trw_memory.user_paths"


# ---------------------------------------------------------------------------
# Security-path derivation (FR01)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "storage_path",
    [
        "/tmp/xdgtest/trw/memory",
        "/home/someone/.trw/memory",  # trw-leak-allow: machine_path synthetic fixture string
        "/var/lib/trw-user/memory",
    ],
)
def test_security_paths_are_the_stores_sibling_under_every_base(storage_path: str) -> None:
    """No nested ``.trw`` under an XDG base, and one rule for all three branches.

    Measured before this change: an XDG base derived
    ``/tmp/xdgtest/trw/.trw/security/quarantine.db``, a ``.trw`` inside an XDG
    data directory, detached from the store it describes.
    """
    config = MemoryConfig(storage_path=storage_path)
    expected = Path(storage_path).parent / "security"

    for field in ("quarantine_db_path", "audit_log_path", "rate_limit_state_path", "provenance_signing_key_path"):
        derived = Path(getattr(config, field))
        assert derived.parent == expected, f"{field} derived {derived}"
        assert ".trw" not in derived.parts[len(expected.parts) - 1 :]


def test_xdg_base_no_longer_nests_a_trw_directory() -> None:
    """The exact measured regression, as a named guard."""
    config = MemoryConfig(storage_path="/tmp/xdgtest/trw/memory")

    assert config.quarantine_db_path == "/tmp/xdgtest/trw/security/quarantine.db"


def test_daemon_config_fields_are_typed_and_bounded() -> None:
    """FR03's three tunables exist with the documented bounds; no bind-host field."""
    config = MemoryConfig()

    assert config.memory_daemon_port == 0
    assert config.memory_daemon_idle_shutdown_seconds == 1800
    assert config.memory_daemon_startup_timeout_seconds == 10.0
    assert not any("bind_host" in name or "daemon_host" in name for name in type(config).model_fields)

    for field, bad in (
        ("memory_daemon_port", 70000),
        ("memory_daemon_idle_shutdown_seconds", 5),
        ("memory_daemon_startup_timeout_seconds", 0.0),
    ):
        with pytest.raises(ValueError):
            MemoryConfig(**{field: bad})


def test_git_timeout_does_not_leak_the_environments_project_root(tmp_path: Path) -> None:
    """canonical_project_root of a file (not a directory) degrades, never raises."""
    target = tmp_path / "afile"
    target.write_text("x", encoding="utf-8")

    assert canonical_project_root(target) == target.resolve()
    assert os.path.isabs(str(canonical_project_root(target)))


# ---------------------------------------------------------------------------
# PRD-CORE-253 FR03/FR09 — the encryption + single-store refusal, and the
# daemon env aliases
# ---------------------------------------------------------------------------


def test_encryption_and_a_single_store_are_refused_together() -> None:
    """A per-NAMESPACE SQLCipher key cannot open a shared FILE.

    SQLCipher keys the whole file; ``derive_namespace_key`` derives a different
    key per namespace. Combined, the first namespace to open the shared store
    sets ``PRAGMA key`` to its own key and every other namespace then cannot
    decrypt the file it is supposed to share — silent at config time, fatal at
    the second namespace. Refused until the per-file key redesign (FR09).
    """
    from trw_memory.exceptions import ConfigError

    with pytest.raises(ConfigError) as refusal:
        MemoryConfig(encryption_enabled=True, memory_single_store_path="/tmp/x/memory.db")

    message = str(refusal.value)
    assert "encryption_enabled" in message and "memory_single_store_path" in message
    assert "FR09" in message, "the error must name the follow-up that unblocks it"

    # Each alone stays valid — the refusal is the COMBINATION, not either field.
    assert MemoryConfig(encryption_enabled=True).encryption_enabled
    assert MemoryConfig(memory_single_store_path="/tmp/x/memory.db").memory_single_store_path


def test_discovery_never_opens_an_encrypted_single_store_keyless(tmp_path: Path) -> None:
    """The single-store discovery branch passes ``sqlcipher_key_hex=None``.

    That is only correct because the combination is refused. If the guard were
    removed, discovery would hand SQLCipher no key for an encrypted file — which
    does not read plaintext, it fails to open, reported far from its cause. This
    pins the guard at the point of use, not just at config construction.
    """
    from trw_memory.exceptions import ConfigError
    from trw_memory.integrations._backend import (
        create_backend_from_config,
        discover_namespace_backends,
    )

    store = tmp_path / "memory.db"
    config = MemoryConfig(storage_path=str(tmp_path), memory_single_store_path=str(store))
    # Mutate past the model validator, which is exactly the shape a caller that
    # assigns to a validated model would produce.
    object.__setattr__(config, "encryption_enabled", True)

    with pytest.raises(ConfigError, match="mutually exclusive"):
        with discover_namespace_backends(config) as stores:
            list(stores)

    with pytest.raises(ConfigError, match="mutually exclusive"):
        create_backend_from_config(config, "project:enc-aaaaaaaa")


@pytest.mark.parametrize(
    ("env_var", "value", "field", "expected"),
    [
        ("MEMORY_DAEMON_PORT", "41234", "memory_daemon_port", 41234),
        ("MEMORY_DAEMON_IDLE_SHUTDOWN_SECONDS", "900", "memory_daemon_idle_shutdown_seconds", 900),
        ("MEMORY_DAEMON_STARTUP_TIMEOUT_SECONDS", "2.5", "memory_daemon_startup_timeout_seconds", 2.5),
        ("MEMORY_SINGLE_STORE_PATH", "/tmp/one/memory.db", "memory_single_store_path", "/tmp/one/memory.db"),
    ],
)
def test_daemon_env_aliases_are_read_without_the_env_prefix(
    monkeypatch: pytest.MonkeyPatch, env_var: str, value: str, field: str, expected: object
) -> None:
    """These fields carry a ``validation_alias``, so ``env_prefix`` does NOT apply.

    The trap, and the reason this test exists: the ``MEMORY_`` prefix rule would
    predict ``MEMORY_MEMORY_SINGLE_STORE_PATH``, and pydantic-settings reads the
    alias VERBATIM instead. The daemon shipped that wrong name once; the setting
    was silently inert and only an end-to-end assertion caught it. Pinning the
    names here means the next such field cannot repeat it quietly.
    """
    monkeypatch.setenv(env_var, value)

    assert getattr(MemoryConfig(), field) == expected


def test_the_prefixed_daemon_env_names_do_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The negative half: the name the prefix rule predicts is NOT read.

    Without this, the test above would still pass if pydantic-settings accepted
    both spellings, and the pin would prove nothing about which one is real.
    """
    monkeypatch.setenv("MEMORY_MEMORY_SINGLE_STORE_PATH", "/tmp/wrong/memory.db")
    monkeypatch.setenv("MEMORY_MEMORY_DAEMON_PORT", "5555")

    config = MemoryConfig()

    assert config.memory_single_store_path == ""
    assert config.memory_daemon_port == 0
