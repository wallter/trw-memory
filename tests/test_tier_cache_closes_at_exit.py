"""The process-lifetime TierManager cache closes its warm.db connections at interpreter exit.

Nothing else closes them: before the atexit hook, every process that wrote the
warm tier left warm.db-wal behind, uncheckpointed (W37).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_WRITER = """
import asyncio, sys
from trw_memory.client import MemoryClient

async def main():
    client = MemoryClient("project:exit-check", mode="local")
    await client.store("the warm tier mirrors this row", tags=["exit"])
    await client.close()

asyncio.run(main())
"""


def test_a_process_that_wrote_the_warm_tier_leaves_no_warm_wal(tmp_path: Path) -> None:
    env = {**os.environ, "MEMORY_STORAGE_PATH": str(tmp_path), "HF_HUB_OFFLINE": "1"}
    subprocess.run([sys.executable, "-c", _WRITER], env=env, check=True, timeout=180)

    warm = list(tmp_path.rglob("warm.db"))
    assert warm, "the writer never reached the warm tier, so this test proves nothing"
    leftovers = [p for p in tmp_path.rglob("*-wal") if p.stat().st_size > 0]
    assert leftovers == [], f"an uncheckpointed WAL survived the process: {leftovers}"
