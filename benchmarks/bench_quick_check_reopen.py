"""One diagnostic run: 1,200 back-to-back single-row MemoryClient.store() calls, drained per row.

Prints one JSON line: which trw_memory it imported, per-bucket mean (store+drain) ms,
and per-bucket mean store / drain ms. Run under PYTHONPATH pointing at the tree to measure.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time

ROWS = int(os.environ.get("BENCH_ROWS", "1200"))
BUCKET = 300
DIM = 384


class HashEmbedder:
    """Deterministic pseudo-random unit vectors, so no two rows dedup or link by accident."""

    def _vec(self, text: str) -> list[float]:
        rng = random.Random(hashlib.sha256(text.encode()).digest())  # noqa: S311 - deterministic vectors
        raw = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]

    def embed(self, text: str) -> list[float] | None:
        return self._vec(text)

    def embed_query(self, text: str) -> list[float] | None:
        return self._vec(text)

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self._vec(t) for t in texts]

    def available(self) -> bool:
        return True

    def dim(self) -> int:
        return DIM

    def embedding_space(self) -> object:
        from trw_memory.embeddings.provenance import EmbeddingSpace

        return EmbeddingSpace("c" * 64, "test-encoder:fix143", DIM)


async def main(seed: int) -> None:
    root = tempfile.mkdtemp(prefix="bench-fix143-")
    os.environ["MEMORY_STORAGE_PATH"] = os.path.join(root, "storage")
    os.environ["MEMORY_STORAGE_BACKEND"] = "sqlite"
    import trw_memory
    from trw_memory.client import MemoryClient
    from trw_memory.graph import wait_for_graph_updates

    embedder = HashEmbedder()
    client = MemoryClient(namespace="default", mode="local")
    client._get_embedder = lambda: embedder  # type: ignore[method-assign]
    await client.__aenter__()
    store_ms: list[float] = []
    drain_ms: list[float] = []
    rng = random.Random(seed)  # noqa: S311 - deterministic benchmark input
    words = ["cache", "sqlite", "graph", "worker", "queue", "index", "recall", "vector", "anomaly", "audit"]
    try:
        for row in range(ROWS):
            text = f"row {seed}-{row}: " + " ".join(rng.choice(words) for _ in range(12))
            t0 = time.perf_counter()
            await client.store(text, tags=["bench"])
            t1 = time.perf_counter()
            wait_for_graph_updates(timeout=5.0)
            t2 = time.perf_counter()
            store_ms.append((t1 - t0) * 1000)
            drain_ms.append((t2 - t1) * 1000)
    finally:
        await client.close()
    buckets = []
    for start in range(0, ROWS, BUCKET):
        s = store_ms[start : start + BUCKET]
        d = drain_ms[start : start + BUCKET]
        buckets.append(
            {
                "total": statistics.fmean(a + b for a, b in zip(s, d, strict=True)),
                "store": statistics.fmean(s),
                "drain": statistics.fmean(d),
            }
        )
    print(json.dumps({"module": trw_memory.__file__, "has_pool": _has_pool(), "buckets": buckets}))


def _has_pool() -> bool:
    try:
        import trw_memory._graph_worker_pool  # noqa: F401
    except ImportError:  # trw-fail-silent-allow: probing whether the tree under test has the pool
        return False
    return True


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1])))
