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

# The writer swaps in a fixed-dimension embedder before building the client. warm.db
# (the SQLiteBackend) is only created for an entry that carries an embedding; with no
# cached model (empty HOME, offline) the real embedder yields none and the warm tier
# would write only its JSONL sidecar, so the test could not depend on the machine's
# model cache. The stub needs no network and no download; everything after the
# embedding (warm_add, the tier cache, the atexit close) is the real production path.
#
# ``atexit.register(os._exit, 0)`` is registered FIRST, so (atexit is LIFO) it runs LAST:
# the process ends right after every other atexit handler, skipping interpreter
# finalization. Finalization alone garbage-collects a lone sqlite3 connection and
# checkpoints the WAL, which would let this test pass with the tier-cache close
# removed. Measured: with no such hard exit and no torch loaded, the WAL is gone even
# with the hook disabled; with torch loaded (a real embedder) 1.8 MB of WAL survives.
# Only an atexit handler can close the connection here.
_WRITER = """
import asyncio, atexit, os, sys
atexit.register(os._exit, 0)
import trw_memory.embeddings as embeddings
from trw_memory.client import MemoryClient


class _FixedEmbedder:
    def embed(self, text):
        return [0.25, 0.5, 0.75, 1.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def available(self):
        return True

    def dim(self):
        return 4


embeddings.get_local_embedder = lambda **_kw: _FixedEmbedder()

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
    sidecars = sorted(p.name for p in tmp_path.rglob("warm.jsonl"))
    assert warm, (
        "warm.db was not created, so this test proves nothing: the warm tier only opens warm.db for an "
        f"entry that carries an embedding (sidecar-only files found: {sidecars}); the stub embedder was not used"
    )
    leftovers = [p for p in tmp_path.rglob("*-wal") if p.stat().st_size > 0]
    assert leftovers == [], f"an uncheckpointed WAL survived the process: {leftovers}"
