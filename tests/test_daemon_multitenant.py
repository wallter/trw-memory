"""PRD-CORE-298 FR03 -- four tenants on one daemon share one embedder and see only their own rows.

The properties come from PRD-CORE-279 (``embeddings/_provider_cache.py`` loads
the model once per process, ``daemon/_offload.py`` serves tool bodies off the
event loop) and PRD-CORE-298 FR02 (a token reaches only its granted
namespaces). Each was tested alone, and this test runs them together. Four
grants for four project namespaces store concurrently, which races the first
model load, then 4 x 25 recalls run concurrently against the one daemon, and
then each grant asks for its neighbour's namespace and must be refused. Rows
coming back only from the caller's namespace is not enough on its own, since
that also holds when a grant is ignored and each caller names only its own.

The daemon runs in its own process, so the construction count is taken there:
a launcher replaces ``LocalEmbeddingProvider`` with a subclass that appends a
line to a file per construction and per vector produced, before handing over to
the real entry point.
No source module changes for the count.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import textwrap
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import pytest
from fastmcp.exceptions import ToolError

from trw_memory.daemon import DaemonClient, DaemonPaths
from trw_memory.daemon._grants import mint_grant

from .test_daemon_server import _await_discovery

pytest.importorskip("fastmcp")
pytest.importorskip("sentence_transformers")

_T = TypeVar("_T")

_TENANTS = tuple(f"project:tenant{index}-{index}{index}{index}{index}aaaa" for index in range(4))
_RECALLS_PER_TENANT = 25
_ROWS_PER_TENANT = 3
#: Each concurrent phase must finish inside this. A regression that reloads the
#: model per call would otherwise hang the test rather than fail it.
_PHASE_DEADLINE_S = 180.0

_LAUNCHER = textwrap.dedent(
    """
    import os, sys, threading
    import trw_memory.embeddings as embeddings

    _log = os.environ["CORE298_CONSTRUCTIONS"]
    _real = embeddings.LocalEmbeddingProvider

    def _note(event):
        with open(_log, "a", encoding="utf-8") as handle:
            handle.write(f"{event} {os.getpid()} {threading.get_ident()}\\n")

    class _Counted(_real):
        def __init__(self, *args, **kwargs):
            _note("built")
            super().__init__(*args, **kwargs)

        def embed(self, text):
            vector = super().embed(text)
            if vector:
                _note("embedded")
            return vector

        def embed_query(self, text):
            vector = super().embed_query(text)
            if vector:
                _note("embedded")
            return vector

        def embed_batch(self, texts):
            vectors = super().embed_batch(texts)
            if any(vectors):
                _note("embedded")
            return vectors

    embeddings.LocalEmbeddingProvider = _Counted
    from trw_memory.server import main
    main(sys.argv[1:])
    """
)


def _spawn_counted_daemon(user_dir: Path, constructions: Path, hf_cache: str) -> subprocess.Popen[str]:
    env = {
        **os.environ,
        "TRW_USER_DIR": str(user_dir),
        "HF_HUB_CACHE": hf_cache,
        "CORE298_CONSTRUCTIONS": str(constructions),
    }
    # A file, never an undrained pipe: the daemon logs each refused call's traceback from its
    # event loop, and once a pipe's buffer fills that write blocks every request (C2 Linux, 7.0).
    with (constructions.parent / "daemon.log").open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            [sys.executable, "-c", _LAUNCHER, "serve", "http", "--idle-shutdown-seconds", "120"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _events(constructions: Path, kind: str) -> list[str]:
    lines = constructions.read_text(encoding="utf-8").splitlines() if constructions.exists() else []
    return [line for line in lines if line.split()[0] == kind]


def _built(constructions: Path) -> list[str]:
    return _events(constructions, "built")


async def _within_deadline(phase: str, constructions: Path, calls: list[Awaitable[_T]]) -> list[_T]:
    try:
        return await asyncio.wait_for(asyncio.gather(*calls), _PHASE_DEADLINE_S)
    except asyncio.TimeoutError:  # a distinct class before Python 3.11
        pytest.fail(
            f"{phase} missed its {_PHASE_DEADLINE_S:.0f}s deadline; {len(_built(constructions))} embedder(s) built"
        )


async def _refused(client: DaemonClient, namespace: str) -> bool:
    try:
        await client.recall("orchard ledger fact", namespace)
    except ToolError as exc:
        return namespace in str(exc)
    return False


def _namespaces_of(result: object) -> set[str]:
    assert isinstance(result, dict), result
    return {str(row["namespace"]) for row in result["memories"]}


async def _timed(call: Awaitable[object]) -> tuple[float, object]:
    started = time.perf_counter()  # a coroutine starts running only when awaited
    result = await call
    return time.perf_counter() - started, result


async def test_four_tenants_share_one_embedder_and_see_only_their_own_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provisioned_embedding_cache: str,
    record_property: Callable[[str, object], None],
) -> None:
    user_dir = tmp_path / "userhome"
    monkeypatch.setenv("TRW_USER_DIR", str(user_dir))
    paths = DaemonPaths.resolve()
    clients = {namespace: DaemonClient(mint_grant(paths, [namespace]), paths=paths) for namespace in _TENANTS}
    constructions = tmp_path / "constructions.log"
    proc = _spawn_counted_daemon(user_dir, constructions, provisioned_embedding_cache)
    try:
        _await_discovery(paths, proc)
        # Every tenant stores at once, so the first model load is itself contended.
        stored = await _within_deadline(
            "concurrent store",
            constructions,
            [
                clients[namespace].store(f"orchard ledger {namespace} fact {index}", namespace)
                for namespace in _TENANTS
                for index in range(_ROWS_PER_TENANT)
            ],
        )
        assert all(isinstance(row, dict) and row["status"] == "stored" for row in stored), stored

        timed = await _within_deadline(
            "concurrent recall",
            constructions,
            [
                _timed(clients[namespace].recall("orchard ledger fact", namespace))
                for namespace in _TENANTS
                for _ in range(_RECALLS_PER_TENANT)
            ],
        )
        # A grant is the boundary: every tenant asking for its neighbour's rows is refused.
        refusals = await _within_deadline(
            "cross-tenant recall",
            constructions,
            [
                _refused(clients[namespace], _TENANTS[(index + 1) % len(_TENANTS)])
                for index, namespace in enumerate(_TENANTS)
            ],
        )
    finally:
        proc.kill()
        proc.wait(timeout=30)

    assert all(refusals), "a grant read a namespace it was not granted"

    callers = [namespace for namespace in _TENANTS for _ in range(_RECALLS_PER_TENANT)]
    leaked = [(caller, _namespaces_of(result) - {caller}) for caller, (_, result) in zip(callers, timed, strict=True)]
    assert [entry for entry in leaked if entry[1]] == [], "a recall returned another tenant's rows"
    assert all(_namespaces_of(result) == {caller} for caller, (_, result) in zip(callers, timed, strict=True)), (
        "every recall finds its own tenant's rows"
    )

    built = _built(constructions)
    assert len(built) == 1, f"expected one embedder construction in the daemon, got {len(built)}: {built}"
    assert _events(constructions, "embedded"), "the shared embedder never produced a vector"

    latencies = sorted(seconds for seconds, _ in timed)
    p50 = statistics.median(latencies)
    p95 = latencies[max(0, round(0.95 * len(latencies)) - 1)]
    report = {"recalls": len(latencies), "tenants": len(_TENANTS), "p50_s": round(p50, 3), "p95_s": round(p95, 3)}
    record_property("core298_fr03_recall_latency", json.dumps(report))
    print(f"CORE-298 FR03 concurrent recall: {json.dumps(report)}")
