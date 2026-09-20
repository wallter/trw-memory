"""A retried mem0 write joins the first extraction instead of paying for a second one (offline)."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("BENCH_BACKEND", "mem0")
    monkeypatch.setenv("BENCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BENCH_MEM0_SESSION_DATE", "0")
    sys.path.insert(0, str(Path(__file__).parent))
    sys.modules.pop("server", None)
    import server

    calls: list[int] = []

    class SlowMemory:
        def add(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
            calls.append(1)
            time.sleep(0.2)
            if messages[0]["content"] == "fail":
                raise RuntimeError("provider 500")
            if messages[0]["content"] == "timeout":
                raise TimeoutError("read timed out")
            if messages[0]["content"] == "wrapped":
                try:
                    raise TimeoutError("read timed out")
                except TimeoutError as inner:
                    raise RuntimeError("LLMError: generation failed") from inner  # how mem0 wraps it
            return {"results": [{"memory": "m"}]}

    b = server.Mem0Backend.__new__(server.Mem0Backend)  # skip model/qdrant construction
    b.memory, b._write_lock, b._pool, b._writes = SlowMemory(), threading.Lock(), ThreadPoolExecutor(4), {}
    b.calls = calls
    yield b
    sys.modules.pop("server", None)
    os.environ.pop("BENCH_BACKEND", None)


def test_concurrent_and_later_retries_share_one_extraction(backend: Any) -> None:
    msgs = [{"role": "user", "content": "hi"}]

    async def go() -> list[Any]:
        first = asyncio.create_task(backend.add(msgs, "u", 1))
        await asyncio.sleep(0.05)  # the client times out and retries while the first write runs
        return [*(await asyncio.gather(first, backend.add(msgs, "u", 1))), await backend.add(msgs, "u", 1)]

    results = asyncio.run(go())
    assert len(backend.calls) == 1 and all(r == {"results": [{"memory": "m"}]} for r in results)
    asyncio.run(backend.add(msgs, "u", 2))  # a different session date is a different write
    assert len(backend.calls) == 2


@pytest.mark.parametrize("content", ["timeout", "wrapped"])
def test_a_timed_out_write_is_never_re_run(backend: Any, content: str) -> None:
    msgs = [{"role": "user", "content": content}]

    async def go() -> None:
        for _ in range(3):  # the runner retries; the provider may already have billed the first call
            with pytest.raises((TimeoutError, RuntimeError)):
                await backend.add(msgs, "u", 1)

    asyncio.run(go())
    assert len(backend.calls) == 1


def test_a_failed_write_can_be_retried(backend: Any) -> None:
    msgs = [{"role": "user", "content": "fail"}]
    for _ in range(2):
        with pytest.raises(RuntimeError):
            asyncio.run(backend.add(msgs, "u", 1))
    assert len(backend.calls) == 2
