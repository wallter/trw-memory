"""PRD-CORE-298 FR04 -- the daemon boots in under 30 s and torch stays off its import path.

A client that auto-starts the daemon waits for the discovery file. That wait
is bounded by ``memory_daemon_startup_timeout_seconds`` and by the MCP
client's own connect timeout, and an import of torch alone costs 2.5-5.6 s
(``test_lazy_imports.py``). These tests pin both halves: the three modules on
the boot path import torch-free in a clean interpreter, and a measured cold
auto-start publishes discovery inside 30 s. The measurement goes to the test
report, so drift shows before the assertion trips.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from trw_memory.daemon import DaemonClient, DaemonPaths
from trw_memory.daemon._discovery import read_live_discovery
from trw_memory.daemon._grants import mint_grant
from trw_memory.models.config import MemoryConfig

pytest.importorskip("fastmcp")

_BOOT_CEILING_S = 30.0
_NAMESPACE = "project:boot-aaaaaaaa"

_BOOT_PATH_MODULES = ("trw_memory.daemon._serve", "trw_memory.daemon.client", "trw_mcp.server._cli")


#: The monorepo checkout that ships trw-mcp beside trw-memory. The public
#: trw-memory mirror has no trw-mcp, so there the trw-mcp guard is skipped.
_MONOREPO_TRW_MCP = Path(__file__).resolve().parents[2] / "trw-mcp"


@pytest.mark.parametrize("module", _BOOT_PATH_MODULES)
def test_a_boot_path_module_imports_without_torch(module: str) -> None:
    package = module.split(".")[0]
    if importlib.util.find_spec(package) is None:
        if package == "trw_mcp" and _MONOREPO_TRW_MCP.is_dir():
            pytest.fail("trw-mcp is in this checkout but not importable; install it so its boot guard runs")
        pytest.skip(f"{package} is not installed alongside trw-memory")
    code = (
        f"import {module}, sys\n"
        "leaked = {'sentence_transformers', 'torch'} & set(sys.modules)\n"
        "print(json.dumps(sorted(leaked)))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", f"import json\n{code}"], capture_output=True, text=True, timeout=120, check=False
    )

    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[DaemonPaths]:
    """An isolated daemon home; any daemon auto-started under it is stopped afterwards."""
    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    resolved = DaemonPaths.resolve()
    yield resolved
    info = read_live_discovery(resolved)
    pid = getattr(info, "pid", None)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:  # trw-fail-silent-allow: the daemon already exited, nothing to stop
            pass


async def test_a_cold_auto_start_publishes_discovery_in_under_30_seconds(
    paths: DaemonPaths, record_property: Callable[[str, object], None]
) -> None:
    assert not paths.discovery.exists()
    client = DaemonClient(
        mint_grant(paths, [_NAMESPACE]),
        config=MemoryConfig(memory_daemon_startup_timeout_seconds=_BOOT_CEILING_S),
        paths=paths,
    )

    started = time.time()
    status = await client.status(_NAMESPACE)
    first_call_s = time.time() - started
    boot_s = paths.discovery.stat().st_mtime - started  # written once, atomically, at publication
    report = {"discovery_published_s": round(boot_s, 3), "first_call_s": round(first_call_s, 3)}
    record_property("core298_fr04_boot", json.dumps(report))
    print(f"CORE-298 FR04 cold auto-start: {json.dumps(report)}")
    assert isinstance(status, dict) and "error" not in status, status
    assert 0 < boot_s < _BOOT_CEILING_S, report
