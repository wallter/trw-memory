"""PRD-SEC-016: store-path symlink TOCTOU hardening in ``trw_memory._dir_trust``.

Each test targets one acceptance criterion from the PRD:

* a directory swapped for a symlink between check and use is refused
* a symlinked store/sidecar file is refused without touching its target
* a world-writable parent owned by someone else is refused
* a world-writable parent WE own is auto-hardened, not refused
* the returned dir_fd anchors a genuinely TOCTOU-safe dir_fd-relative open
"""

from __future__ import annotations

import importlib
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from trw_memory._dir_trust import (
    DIR_FD_SUPPORTED,
    make_private_dirs,
    open_anchored_walk,
    open_component_fd,
    open_or_create_at,
    open_verified_dir_fd,
    verify_ancestor_chain_trusted,
    verify_and_harden_dir_fd,
)
from trw_memory.exceptions import UntrustedDirectoryError

_POSIX_ONLY = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode/owner bits")


# ---------------------------------------------------------------------------
# open_verified_dir_fd: symlink refusal
# ---------------------------------------------------------------------------


@_POSIX_ONLY
def test_open_verified_dir_fd_refuses_symlinked_directory(tmp_path: Path) -> None:
    target = tmp_path / "real-target"
    target.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)

    with pytest.raises((OSError, UntrustedDirectoryError)):
        open_verified_dir_fd(linked, create=False)


@_POSIX_ONLY
def test_open_verified_dir_fd_swapped_between_check_and_use_is_refused(tmp_path: Path) -> None:
    """The check-then-use race itself: caller checks existence, attacker swaps
    in a symlink, caller's open still refuses -- because the open (not a
    prior stat) is what raises."""
    real = tmp_path / "trusted"
    real.mkdir(mode=0o700)
    assert real.exists()  # the "check" a naive caller might perform

    # The "swap": attacker replaces the directory with a symlink to a
    # location they control before the "use" (the open) happens.
    evil = tmp_path / "evil"
    evil.mkdir(mode=0o777)
    real.rmdir()
    real.symlink_to(evil, target_is_directory=True)

    with pytest.raises((OSError, UntrustedDirectoryError)):
        open_verified_dir_fd(real, create=False)


def test_open_verified_dir_fd_missing_and_create_false_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_verified_dir_fd(tmp_path / "absent", create=False)


def test_open_verified_dir_fd_create_true_creates_and_opens(tmp_path: Path) -> None:
    target = tmp_path / "new-dir"
    fd = open_verified_dir_fd(target, create=True)
    try:
        assert target.is_dir()
        assert not target.is_symlink()
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# verify_and_harden_dir_fd: ownership / world-writable
# ---------------------------------------------------------------------------


@_POSIX_ONLY
def test_verify_and_harden_dir_fd_auto_hardens_own_permissive_dir(tmp_path: Path) -> None:
    target = tmp_path / "permissive-own"
    target.mkdir(mode=0o775)
    os.chmod(target, 0o775)

    fd = open_verified_dir_fd(target, create=False)
    try:
        verify_and_harden_dir_fd(fd, target)
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o700
        assert stat.S_IMODE(target.stat().st_mode) == 0o700
    finally:
        os.close(fd)


@_POSIX_ONLY
def test_verify_and_harden_dir_fd_refuses_world_writable_owned_by_other_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trw_memory._dir_trust as dir_trust

    target = tmp_path / "foreign"
    target.mkdir(mode=0o777)
    os.chmod(target, 0o777)
    real_uid = target.stat().st_uid
    monkeypatch.setattr(dir_trust.os, "geteuid", lambda: real_uid + 1)

    fd = open_verified_dir_fd(target, create=False)
    try:
        with pytest.raises(UntrustedDirectoryError, match="group/world-writable"):
            verify_and_harden_dir_fd(fd, target)
    finally:
        os.close(fd)
    # Refused, not silently hardened -- the mode on disk is untouched.
    assert stat.S_IMODE(target.stat().st_mode) == 0o777


@_POSIX_ONLY
def test_verify_and_harden_dir_fd_sticky_bit_is_treated_as_safe(tmp_path: Path) -> None:
    target = tmp_path / "sticky"
    target.mkdir()
    os.chmod(target, 0o1777)

    fd = open_verified_dir_fd(target, create=False)
    try:
        st = verify_and_harden_dir_fd(fd, target)
        assert stat.S_IMODE(st.st_mode) == 0o1777  # untouched: sticky bit makes world-writable safe
    finally:
        os.close(fd)


def test_verify_and_harden_dir_fd_private_dir_is_a_noop(tmp_path: Path) -> None:
    target = tmp_path / "already-private"
    target.mkdir(mode=0o700)
    os.chmod(target, 0o700)

    fd = open_verified_dir_fd(target, create=False)
    try:
        st = verify_and_harden_dir_fd(fd, target)
        assert stat.S_IMODE(st.st_mode) == 0o700
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# verify_ancestor_chain_trusted (PRD-SEC-016 FR04)
# ---------------------------------------------------------------------------


@_POSIX_ONLY
def test_verify_ancestor_chain_trusted_refuses_a_foreign_owned_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trw_memory._dir_trust as dir_trust

    store_dir = tmp_path / "home" / "user" / "memory"
    store_dir.mkdir(parents=True, mode=0o700)
    real_uid = store_dir.stat().st_uid
    if real_uid == 0:
        # As root (the Linux release container) every directory the test makes is
        # root-owned, and a root-owned ancestor is trusted by design: pretending the
        # daemon is another uid proves nothing. Give the leaf a real foreign owner.
        os.chown(store_dir, 4242, -1)
    else:
        monkeypatch.setattr(dir_trust.os, "geteuid", lambda: real_uid + 1)

    with pytest.raises(UntrustedDirectoryError, match="not the daemon's uid"):
        verify_ancestor_chain_trusted(store_dir)


@_POSIX_ONLY
def test_verify_ancestor_chain_trusted_serves_the_default_private_layout(tmp_path: Path) -> None:
    store_dir = tmp_path / "home" / "user" / "memory"
    store_dir.mkdir(parents=True, mode=0o700)

    verify_ancestor_chain_trusted(store_dir)  # must not raise


@pytest.fixture
def umask_0002() -> Iterator[None]:
    """The login umask stock Ubuntu gives its users (user-private groups)."""
    previous = os.umask(0o002)
    try:
        yield
    finally:
        os.umask(previous)


@_POSIX_ONLY
@pytest.mark.usefixtures("umask_0002")
def test_make_private_dirs_creates_every_component_private(tmp_path: Path) -> None:
    leaf = tmp_path / "a" / "b" / "c"
    make_private_dirs(leaf)
    for component in (tmp_path / "a", tmp_path / "a" / "b", leaf):
        assert stat.S_IMODE(component.stat().st_mode) == 0o700, component


@_POSIX_ONLY
@pytest.mark.usefixtures("umask_0002")
def test_the_default_user_dir_is_trusted_under_a_0002_umask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stock Ubuntu: ``~/.trw`` created by this package itself must not be refused as group-writable."""
    from trw_memory.user_paths import resolve_user_memory_dir

    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o750)  # what useradd -m gives a stock Ubuntu user
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TRW_USER_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    memory_dir = resolve_user_memory_dir(create=True)

    assert stat.S_IMODE((home / ".trw").stat().st_mode) == 0o700
    verify_ancestor_chain_trusted(memory_dir)  # must not raise


def _user_env(monkeypatch: pytest.MonkeyPatch, home: Path, *, xdg: Path | None = None) -> None:
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TRW_USER_DIR", raising=False)
    if xdg is None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg))


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@_POSIX_ONLY
@pytest.mark.parametrize("create", [True, False])
def test_a_3x_era_group_writable_trw_dir_is_hardened_before_the_daemon_checks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, create: bool
) -> None:
    """Release-verify P1-A: 3.1.0 left ``~/.trw`` and ``~/.trw/memory`` 0775 under umask 0002."""
    from trw_memory.user_paths import resolve_user_memory_dir

    home = tmp_path / "home"
    home.mkdir(mode=0o750)
    for legacy in (home / ".trw", home / ".trw" / "memory"):
        legacy.mkdir()
        legacy.chmod(0o775)
    _user_env(monkeypatch, home)

    memory_dir = resolve_user_memory_dir(create=create)

    assert _mode(home / ".trw") == 0o700
    assert _mode(memory_dir) == 0o700
    verify_ancestor_chain_trusted(memory_dir)  # the daemon's startup gate must pass


@_POSIX_ONLY
def test_the_xdg_trw_dir_is_hardened_but_xdg_data_home_itself_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.user_paths import resolve_user_memory_dir

    home = tmp_path / "home"
    home.mkdir(mode=0o750)
    xdg = home / "share"
    (xdg / "trw").mkdir(parents=True)
    xdg.chmod(0o775)
    (xdg / "trw").chmod(0o775)
    _user_env(monkeypatch, home, xdg=xdg)

    memory_dir = resolve_user_memory_dir(create=True)

    assert _mode(xdg / "trw") == 0o700
    assert _mode(xdg) == 0o775, "a directory trw-memory does not own is never chmodded"
    with pytest.raises(UntrustedDirectoryError, match="group/world-writable"):
        verify_ancestor_chain_trusted(memory_dir)


@_POSIX_ONLY
def test_a_foreign_owned_group_writable_trw_dir_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trw_memory._dir_trust as dir_trust
    from trw_memory.user_paths import resolve_user_memory_dir

    home = tmp_path / "home"
    home.mkdir(mode=0o750)
    trw = home / ".trw"
    trw.mkdir()
    trw.chmod(0o775)
    monkeypatch.setattr(dir_trust.os, "geteuid", lambda: trw.stat().st_uid + 1)
    _user_env(monkeypatch, home)

    with pytest.raises(UntrustedDirectoryError, match="owned by uid"):
        resolve_user_memory_dir(create=True)
    assert _mode(trw) == 0o775


@_POSIX_ONLY
def test_verify_ancestor_chain_trusted_sticky_ancestor_serves(tmp_path: Path) -> None:
    sticky = tmp_path / "sticky"
    sticky.mkdir(mode=0o1777)
    os.chmod(sticky, 0o1777)
    store_dir = sticky / "user" / "memory"
    store_dir.mkdir(parents=True, mode=0o700)

    verify_ancestor_chain_trusted(store_dir)  # must not raise


def test_verify_ancestor_chain_trusted_ignores_an_unreadable_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ancestor that cannot even be ``.stat()``'d is skipped -- there is
    nothing to trust or refuse about a directory that raises on inspection."""
    store_dir = tmp_path / "home" / "user" / "memory"
    store_dir.mkdir(parents=True, mode=0o700)

    real_stat = Path.stat

    def _flaky_stat(self: Path, *a: object, **k: object) -> object:
        if self.name == "home":
            raise OSError("simulated: cannot stat this ancestor")
        return real_stat(self, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", _flaky_stat)

    verify_ancestor_chain_trusted(store_dir)  # must not raise -- unreadable ancestor is skipped


# ---------------------------------------------------------------------------
# dir_fd-relative open is genuinely TOCTOU-safe end to end
# ---------------------------------------------------------------------------


@_POSIX_ONLY
@pytest.mark.skipif(not DIR_FD_SUPPORTED, reason="platform lacks openat(2) dir_fd support")
def test_dir_fd_relative_open_is_immune_to_a_post_verification_swap(tmp_path: Path) -> None:
    """Once a dir_fd is verified, swapping the PATH it came from must not
    redirect an open anchored to that fd -- the fd is bound to the original
    inode, not the name."""
    real = tmp_path / "verified"
    real.mkdir(mode=0o700)
    fd = open_verified_dir_fd(real, create=False)
    try:
        verify_and_harden_dir_fd(fd, real)

        # Swap the NAME "verified" out from under the already-open fd.
        real.rename(tmp_path / "renamed-away")
        evil = tmp_path / "verified"
        evil.mkdir(mode=0o777)
        (evil / "secret.txt").write_text("attacker-planted")

        # A dir_fd-relative open still reaches the ORIGINAL directory, not
        # whatever now occupies the swapped-in name.
        write_fd = os.open("secret.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=fd)
        os.close(write_fd)
        # Landed in the ORIGINAL (renamed) directory, not the attacker's swapped-in one --
        # the attacker's own file at the same name is left untouched.
        assert (tmp_path / "renamed-away" / "secret.txt").exists()
        assert (evil / "secret.txt").read_text() == "attacker-planted"
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# PRD-SEC-016 round-2 finding 5: a missing O_NOFOLLOW must refuse, never
# silently degrade to flag 0 (which follows a symlink without complaint).
# ---------------------------------------------------------------------------


@_POSIX_ONLY
def test_open_component_fd_refuses_when_o_nofollow_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patching ``NOFOLLOW_SUPPORTED`` off must refuse the open, not silently follow a symlink."""
    import trw_memory._dir_trust as dir_trust

    real_target = tmp_path / "real"
    real_target.mkdir()
    (real_target / "file.txt").write_text("content")
    dir_fd = open_verified_dir_fd(tmp_path, create=False)
    try:
        monkeypatch.setattr(dir_trust, "NOFOLLOW_SUPPORTED", False)

        with pytest.raises(UntrustedDirectoryError, match="O_NOFOLLOW"):
            dir_trust.open_component_fd(dir_fd, "real", directory=True)
    finally:
        os.close(dir_fd)


@_POSIX_ONLY
def test_open_anchored_walk_refuses_when_o_nofollow_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trw_memory._dir_trust as dir_trust

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(dir_trust, "NOFOLLOW_SUPPORTED", False)

    with pytest.raises(UntrustedDirectoryError, match="O_NOFOLLOW"):
        dir_trust.open_anchored_walk(checkout)


@_POSIX_ONLY
def test_create_private_file_fd_refuses_when_o_nofollow_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trw_memory._dir_trust as dir_trust

    dir_fd = open_verified_dir_fd(tmp_path, create=False)
    try:
        monkeypatch.setattr(dir_trust, "NOFOLLOW_SUPPORTED", False)

        with pytest.raises(UntrustedDirectoryError, match="O_NOFOLLOW"):
            dir_trust.create_private_file_fd(dir_fd, "new-file.db")
    finally:
        os.close(dir_fd)


@pytest.fixture
def dir_trust_without_o_nofollow() -> Iterator[ModuleType]:
    """``_dir_trust`` re-imported on a platform with no ``os.O_NOFOLLOW``; always reloaded back afterwards.

    A reload runs the module's own flag computation rather than patching its
    result. It updates the module dict in place, so every importer's function
    references see the restored values after the teardown reload.
    """
    import trw_memory._dir_trust as dir_trust

    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.delattr(os, "O_NOFOLLOW", raising=False)
            yield importlib.reload(dir_trust)
    finally:
        importlib.reload(dir_trust)
        assert dir_trust.NOFOLLOW_SUPPORTED is True


@_POSIX_ONLY
def test_a_missing_o_nofollow_constant_computes_as_unsupported(
    tmp_path: Path, dir_trust_without_o_nofollow: ModuleType
) -> None:
    """With no O_NOFOLLOW the flag is False and the no-follow open refuses instead of degrading to flag 0."""
    dir_trust = dir_trust_without_o_nofollow
    assert dir_trust._NOFOLLOW == 0
    assert dir_trust.NOFOLLOW_SUPPORTED is False

    (tmp_path / "plain.txt").write_text("x")
    dir_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(UntrustedDirectoryError, match="lacks O_NOFOLLOW;"):
            dir_trust.open_component_fd(dir_fd, "plain.txt", directory=False)
    finally:
        os.close(dir_fd)


@_POSIX_ONLY
def test_open_component_fd_still_refuses_a_real_symlink_when_nofollow_is_supported(tmp_path: Path) -> None:
    """Sanity control: with real capabilities (no patching), the ordinary symlink refusal still holds."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "link.txt").symlink_to(outside)
    dir_fd = open_verified_dir_fd(checkout, create=False)
    try:
        with pytest.raises(UntrustedDirectoryError):
            open_component_fd(dir_fd, "link.txt", directory=False)
    finally:
        os.close(dir_fd)


# ---------------------------------------------------------------------------
# PRD-SEC-016 round-2 finding 1: open_anchored_walk must refuse a symlinked
# ANCESTOR of the walked path itself, not just accept whatever a caller's
# own prior Path.resolve() already followed.
# ---------------------------------------------------------------------------


@_POSIX_ONLY
def test_open_anchored_walk_refuses_a_symlinked_ancestor_of_the_root(tmp_path: Path) -> None:
    """An ancestor of *root* swapped for a symlink AFTER the checkout was created must be refused.

    This is the nested-tenant scenario the docstring names: checkout B's root
    sits under checkout A's; A's grant lets A swap an ancestor of B's root.
    """
    import shutil

    outer = tmp_path / "outer"
    checkout = outer / "inner-checkout"
    checkout.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "inner-checkout").mkdir(parents=True)
    (elsewhere / "inner-checkout" / "planted.txt").write_text("attacker content")

    # The swap: "outer" (an ancestor of the checkout, but not the checkout
    # itself) is replaced by a symlink pointing somewhere else entirely.
    shutil.rmtree(outer)
    outer.symlink_to(elsewhere)

    with pytest.raises(UntrustedDirectoryError):
        open_anchored_walk(outer / "inner-checkout")


@_POSIX_ONLY
def test_open_anchored_walk_succeeds_on_a_pre_resolved_path_through_a_legitimate_symlinked_home(
    tmp_path: Path,
) -> None:
    """A once-resolved path (what ``mint_grant`` stores) walks cleanly: no symlink remains IN it.

    Simulates the macOS ``/var`` -> ``/private/var`` case: the caller's raw
    path goes through a symlinked "home," but the grant records the ALREADY
    RESOLVED destination, so the walk downstream never encounters a symlink
    component at all.
    """
    real_home = tmp_path / "real-home"
    (real_home / "project").mkdir(parents=True)
    symlinked_home = tmp_path / "home"
    symlinked_home.symlink_to(real_home)

    raw_caller_path = symlinked_home / "project"
    resolved_at_mint_time = raw_caller_path.resolve()  # what mint_grant would store, once
    assert resolved_at_mint_time == real_home / "project"

    fd = open_anchored_walk(resolved_at_mint_time)  # walking the resolved form: no symlink in it
    os.close(fd)


# ---------------------------------------------------------------------------
# open_or_create_at: concurrent first creates of one name (7.0.0 daemon P0)
# ---------------------------------------------------------------------------

_CREATE_FLAGS = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


@_POSIX_ONLY
@pytest.mark.skipif(not DIR_FD_SUPPORTED, reason="platform lacks openat(2) dir_fd support")
def test_open_or_create_at_survives_concurrent_first_creates(tmp_path: Path) -> None:
    """Three openers creating one new name at once all get a descriptor to the same file.

    A bare ``openat(dir_fd, name, O_CREAT)`` hands the loser ENOENT on macOS
    (about 7 in 10 contended pairs): the daemon's first stores then failed with
    ``Secure SQLite open failed: [Errno 2]``. Two hundred rounds make a return
    of that open near-certain to fail here on macOS; elsewhere this is a smoke.
    """
    import threading

    failures: list[str] = []
    for round_no in range(200):
        directory = tmp_path / f"d{round_no}"
        directory.mkdir(mode=0o700)
        barrier = threading.Barrier(3)
        inodes: list[int] = []

        def create(
            directory: Path = directory,
            barrier: threading.Barrier = barrier,
            inodes: list[int] = inodes,
            round_no: int = round_no,
        ) -> None:
            barrier.wait()
            dir_fd = open_verified_dir_fd(directory, create=False)
            try:
                fd = open_or_create_at(dir_fd, "memory.db", _CREATE_FLAGS, 0o600)
                inodes.append(os.fstat(fd).st_ino)
                os.close(fd)
            except OSError as exc:  # recorded, then asserted empty below
                failures.append(f"round {round_no}: {exc!r}")
            finally:
                os.close(dir_fd)

        threads = [threading.Thread(target=create) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if len(set(inodes)) > 1:
            failures.append(f"round {round_no}: openers got different files {inodes}")

    assert failures == []


@_POSIX_ONLY
@pytest.mark.skipif(not DIR_FD_SUPPORTED, reason="platform lacks openat(2) dir_fd support")
def test_open_or_create_at_goes_round_when_the_file_vanishes_between_its_two_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced interleaving: someone else creates the name, then removes it; the next round creates it."""
    import errno

    import trw_memory._dir_trust as dir_trust

    real_open = os.open
    script = [
        OSError(errno.EEXIST, "created by another opener"),
        OSError(errno.ENOENT, "removed by another opener"),
    ]
    seen: list[bool] = []

    def scripted_open(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is not None and path == "memory.db":
            seen.append(bool(flags & os.O_EXCL))
            if script:
                exc = script.pop(0)
                raise FileExistsError(*exc.args) if exc.errno == errno.EEXIST else FileNotFoundError(*exc.args)
            return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]
        if dir_fd is not None:
            return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]
        return real_open(path, flags, mode)  # type: ignore[arg-type]

    monkeypatch.setattr(dir_trust.os, "open", scripted_open)
    dir_fd = open_verified_dir_fd(tmp_path, create=False)
    try:
        fd = open_or_create_at(dir_fd, "memory.db", _CREATE_FLAGS, 0o600)
    finally:
        os.close(dir_fd)
    os.close(fd)

    assert seen == [True, False, True], "exclusive create, plain open, then a second exclusive create"
    assert stat.S_IMODE((tmp_path / "memory.db").stat().st_mode) == 0o600


@_POSIX_ONLY
@pytest.mark.skipif(not DIR_FD_SUPPORTED, reason="platform lacks openat(2) dir_fd support")
def test_open_or_create_at_still_refuses_a_symlink(tmp_path: Path) -> None:
    """The split create keeps O_NOFOLLOW: a planted symlink fails the create and then the open."""
    target = tmp_path / "target.db"
    target.write_text("do not touch")
    target.chmod(0o644)
    (tmp_path / "memory.db").symlink_to(target)

    dir_fd = open_verified_dir_fd(tmp_path, create=False)
    try:
        with pytest.raises(OSError) as refused:
            open_or_create_at(dir_fd, "memory.db", _CREATE_FLAGS, 0o600)
    finally:
        os.close(dir_fd)

    assert not isinstance(refused.value, (FileExistsError, FileNotFoundError))
    assert target.read_text() == "do not touch"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
