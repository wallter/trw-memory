"""Minting a grant drops the grants whose checkout is gone (TEST-ISOLATION-DAEMON-GRANTS).

An operator's real `daemon-grants.json` held 1,379 grants, ~1,350 of them rooted in temporary directories that no longer
exist, each granting `user:local`. Nothing ever removed a grant. `mint_grant` already holds the grants lock and the map,
so it now drops the grants whose root has been deleted, and never one that merely cannot be seen: a rootless grant, a
root that still exists, or a root on a volume that is not mounted (its parent directory is missing too).
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.daemon._grants import mint_grant, read_grant
from trw_memory.daemon._paths import DaemonPaths


def _paths(tmp_path: Path) -> DaemonPaths:
    user_dir = tmp_path / "userdir"
    user_dir.mkdir(mode=0o700)
    return DaemonPaths(user_memory_dir=user_dir)


def test_a_grant_for_a_deleted_checkout_is_dropped_by_the_next_mint(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    live = tmp_path / "live"
    live.mkdir()
    gone_parent = tmp_path / "scratch"
    gone_parent.mkdir()
    gone = gone_parent / "deleted-checkout"
    gone.mkdir()

    t_live = mint_grant(paths, ["project:a-11111111", "user:local"], root=live)
    t_gone = mint_grant(paths, ["project:b-22222222", "user:local"], root=gone)
    gone.rmdir()  # the checkout is deleted; its parent directory remains
    t_new = mint_grant(paths, ["project:c-33333333", "user:local"], root=live)

    assert read_grant(paths, t_live) is not None
    assert read_grant(paths, t_new) is not None
    assert read_grant(paths, t_gone) is None  # the dead checkout's grant to user:local is gone


def test_a_rootless_grant_and_a_live_root_are_never_pruned(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    live = tmp_path / "live"
    live.mkdir()
    t_rootless = mint_grant(paths, ["project:a-11111111", "user:local"])
    t_live = mint_grant(paths, ["project:b-22222222", "user:local"], root=live)

    mint_grant(paths, ["project:c-33333333", "user:local"], root=live)

    assert read_grant(paths, t_rootless) is not None and read_grant(paths, t_live) is not None


def test_a_root_on_an_unmounted_volume_is_kept(tmp_path: Path) -> None:
    """Both the root and its parent are missing (an unplugged drive): that is absence of evidence, not a deleted checkout."""
    paths = _paths(tmp_path)
    volume = tmp_path / "Volumes" / "External"
    volume.mkdir(parents=True)
    root = volume / "repo"
    root.mkdir()
    token = mint_grant(paths, ["project:a-11111111", "user:local"], root=root)
    root.rmdir()
    volume.rmdir()  # "unplug" the drive: root and parent both gone

    mint_grant(paths, ["project:c-33333333", "user:local"])

    assert read_grant(paths, token) is not None


def test_pruning_logs_a_count_and_never_the_roots(tmp_path: Path) -> None:
    import structlog.testing

    paths = _paths(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    root = scratch / "x"
    root.mkdir()
    mint_grant(paths, ["project:a-11111111"], root=root)
    root.rmdir()

    with structlog.testing.capture_logs() as logs:
        mint_grant(paths, ["project:b-22222222"])

    pruned = [entry for entry in logs if entry.get("event") == "daemon_grants_pruned"]
    assert pruned and pruned[0]["count"] == 1 and str(root) not in str(pruned[0])
