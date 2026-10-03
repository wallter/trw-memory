"""DAEMON-MODEL-WARMUP: the daemon loads its embedding and rerank models right after it binds.

The first recall after every daemon start used to pay the torch and sentence-transformers import plus the two
model loads (about 6 s, against about 1 s warm). A background thread now does that work while the daemon is
already serving, so the operator's first recall of a session meets a warm model.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig


class _Calls:
    def __init__(self) -> None:
        self.embedder: list[str] = []
        self.reranker: list[str] = []
        self.threads: list[str] = []


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    from trw_memory.daemon import _warmup

    seen = _Calls()

    def embedder(config: MemoryConfig, *, surface: str) -> object:
        seen.embedder.append(surface)
        seen.threads.append(threading.current_thread().name)
        return object()

    def reranker(model_name: str) -> object:
        seen.reranker.append(model_name)
        seen.threads.append(threading.current_thread().name)
        return object()

    monkeypatch.setattr(_warmup, "resolve_embedder", embedder)
    monkeypatch.setattr(_warmup, "load_reranker", reranker)
    return seen


def test_both_models_load_on_a_background_thread(calls: _Calls) -> None:
    from trw_memory.daemon._warmup import start_model_warmup

    thread = start_model_warmup(MemoryConfig(embeddings_enabled=True))

    assert thread is not None
    thread.join(timeout=10)
    assert calls.embedder == ["warmup"]
    assert calls.reranker == [MemoryConfig().recall_rerank_model]
    assert set(calls.threads) == {"trw-model-warmup"}


def test_embeddings_off_loads_nothing(calls: _Calls) -> None:
    from trw_memory.daemon._warmup import start_model_warmup

    assert start_model_warmup(MemoryConfig(embeddings_enabled=False)) is None
    assert calls.embedder == [] and calls.reranker == []


def test_a_failing_load_is_logged_and_never_stops_the_other_model(
    calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.daemon import _warmup

    def boom(config: MemoryConfig, *, surface: str) -> object:
        raise OSError("model cache unreadable")

    monkeypatch.setattr(_warmup, "resolve_embedder", boom)

    thread = _warmup.start_model_warmup(MemoryConfig(embeddings_enabled=True))

    assert thread is not None
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert calls.reranker == [MemoryConfig().recall_rerank_model]


_SERVE_CHILD = """
import asyncio, sys
from pathlib import Path

from trw_memory.daemon import DaemonPaths, _serve

marker = Path(sys.argv[1])
_serve.start_model_warmup = lambda *_a, **_k: marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
paths = DaemonPaths.resolve()
asyncio.run(_serve.serve_loopback(_serve.DaemonServeOptions(port=0, idle_shutdown_seconds=1.0), paths=paths))
"""


def test_the_serving_loop_starts_the_warmup_once_after_it_binds(tmp_path: Path) -> None:
    """In a child process: a real in-process ``serve_loopback`` leaves state that breaks the tests after it."""
    import os
    import subprocess
    import sys

    marker = tmp_path / "started"
    env = {**os.environ, "TRW_USER_DIR": str(tmp_path / "userhome")}

    done = subprocess.run(
        [sys.executable, "-c", _SERVE_CHILD, str(marker)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert done.returncode == 0, done.stderr[-800:]
    assert marker.read_text() == "x"
