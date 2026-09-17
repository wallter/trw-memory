"""PRD-CORE-279 FR05/FR06/NFR03: the daemon cleans up after itself.

The defect these pin is subtle: ``serve_loopback`` always called
``release_single_instance`` in a ``finally``, and the record still survived
SIGTERM. uvicorn restores the handlers it found and re-raises the captured
signal when ``serve()`` returns, so the default disposition killed the process
from inside ``serve()``. The subprocess test therefore asserts the OUTCOME (no
record, killed by the signal), not the mechanism.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from trw_memory.daemon._discovery import DaemonInfo, write_discovery
from trw_memory.daemon._instance import claim_single_instance, release_single_instance
from trw_memory.daemon._paths import DaemonPaths

_START_TIMEOUT = 90.0
_EXIT_TIMEOUT = 30.0

_LAUNCHER = """
import sys
from trw_memory.server import main
main(["serve", "http", "--port", "0", "--idle-shutdown-seconds", "120"])
"""


def _paths(tmp_path: Path) -> DaemonPaths:
    return DaemonPaths(user_memory_dir=tmp_path)


def _wait_for_discovery(discovery: Path, proc: subprocess.Popen[bytes]) -> dict[str, object]:
    deadline = time.monotonic() + _START_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(f"daemon exited early ({proc.returncode}): {err.decode(errors='replace')[-2000:]}")
        if discovery.exists():
            try:
                return json.loads(discovery.read_text())
            except ValueError:
                pass
        time.sleep(0.1)
    proc.kill()
    pytest.fail("daemon never published a discovery record")


@pytest.mark.slow
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_sigterm_removes_the_discovery_record(tmp_path, signum):
    """FR05: the record is gone once the signalled daemon is gone."""
    user_dir = tmp_path / "user"
    env = dict(os.environ)
    env["TRW_USER_DIR"] = str(user_dir)
    env.pop("TRW_META_TUNE_ENABLED", None)
    discovery = user_dir / "memory" / "daemon.json"

    proc = subprocess.Popen(
        [sys.executable, "-c", _LAUNCHER],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        record = _wait_for_discovery(discovery, proc)
        assert record["pid"] == proc.pid

        proc.send_signal(signum)
        try:
            returncode = proc.wait(timeout=_EXIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail(f"daemon did not exit after {signum!r}")

        assert not discovery.exists(), "the daemon left a record naming its dead pid"
        # The process must still die FROM the signal, not quietly exit 0: an
        # operator's `kill` and a service manager's stop both read the code.
        assert returncode == -signum, f"expected death by {signum!r}, got returncode {returncode}"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


#: Runs the real ``serve_loopback`` with the app build sabotaged. This runs in a
#: SUBPROCESS rather than in the test process on purpose: ``serve_loopback`` is
#: a process-lifetime routine -- it pins MEMORY_STORAGE_PATH and
#: MEMORY_SINGLE_STORE_PATH into ``os.environ``, installs signal handlers and
#: touches the module-level FastMCP singleton -- and a test process has many
#: logical lifetimes in one real one. Running it in-process made an unrelated
#: real-daemon test elsewhere in the suite time out under ``-n 4``.
_STARTUP_FAILURE_LAUNCHER = """
import asyncio
import sys
from pathlib import Path

import trw_memory.daemon._serve as serve_mod
from trw_memory.daemon._paths import DaemonPaths


def _explode(_token):
    raise RuntimeError("app build failed")


serve_mod._build_app = _explode
paths = DaemonPaths(user_memory_dir=Path(sys.argv[1]))
options = serve_mod.DaemonServeOptions(port=0, idle_shutdown_seconds=5.0)
try:
    asyncio.run(serve_mod.serve_loopback(options, paths=paths))
except RuntimeError as exc:
    print(f"RAISED:{exc}")
    sys.exit(3)
sys.exit(0)
"""


def test_startup_failure_after_claim_releases_it(tmp_path):
    """FR06: a failure between the claim and serving must not leak the record."""
    user_dir = tmp_path / "user"
    user_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TRW_USER_DIR"] = str(tmp_path / "unused")
    env.pop("TRW_META_TUNE_ENABLED", None)

    completed = subprocess.run(
        [sys.executable, "-c", _STARTUP_FAILURE_LAUNCHER, str(user_dir)],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=_START_TIMEOUT,
        check=False,
    )

    assert completed.returncode == 3, f"expected the build failure to propagate: {completed.stderr[-2000:]}"
    assert "RAISED:app build failed" in completed.stdout
    assert not (user_dir / "daemon.json").exists(), "a startup failure left the instance claimed"


def test_release_leaves_a_successor_record_alone(tmp_path):
    """FR06 negative: a record written by another claim is never deleted."""
    paths = _paths(tmp_path)
    claim = claim_single_instance(paths, port=0, token="t0ken", version="test")
    try:
        first = claim.info
        # A second claim from the SAME pid is permitted by design, and rewrites
        # the record. A late release from the first claim must not delete it.
        time.sleep(0.01)
        successor = write_discovery(paths, url="http://127.0.0.1:1/mcp", token="t0ken", version="test")
        assert successor.started_at != first.started_at

        release_single_instance(paths, claimed=first)
        assert paths.discovery.exists(), "the stale claim deleted its successor's record"

        release_single_instance(paths, claimed=successor)
        assert not paths.discovery.exists()
    finally:
        claim.sock.close()


def test_release_leaves_another_pids_record_alone(tmp_path):
    """FR06 negative: another live pid's record is not ours to remove."""
    paths = _paths(tmp_path)
    paths.user_memory_dir.mkdir(parents=True, exist_ok=True)
    other = DaemonInfo(
        pid=os.getpid() + 1,
        url="http://127.0.0.1:2/mcp",
        token="t0ken",
        started_at="2026-09-17T00:00:00+00:00",
        version="test",
    )
    from trw_memory.daemon._paths import write_secret_file

    write_secret_file(paths.discovery, other.model_dump_json())

    release_single_instance(paths)
    assert paths.discovery.exists()


def test_release_without_a_claim_still_removes_our_own_record(tmp_path):
    """Backward compatibility: the one-argument call keeps its old behaviour."""
    paths = _paths(tmp_path)
    claim = claim_single_instance(paths, port=0, token="t0ken", version="test")
    try:
        release_single_instance(paths)
        assert not paths.discovery.exists()
    finally:
        claim.sock.close()


def test_trust_boundary_is_documented():
    """NFR03: the single-principal property is stated where the daemon runs."""
    import trw_memory.daemon._serve as serve_mod

    doc = serve_mod.serve_loopback.__doc__ or ""
    assert "one principal" in doc.lower()
    assert "every namespace" in doc.lower()

    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text()
    assert "Loopback daemon" in text
    assert "one principal" in text.lower()
