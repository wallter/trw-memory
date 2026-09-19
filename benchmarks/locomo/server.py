"""Mem0-OSS-compatible REST shim so mem0's own benchmark runner can drive trw-memory.

mem0's ``memory-benchmarks`` runner (https://github.com/mem0ai/memory-benchmarks)
talks to a self-hosted mem0 server over three endpoints::

    POST   /memories   {"messages": [...], "user_id": ..., "timestamp": <epoch>}
    POST   /search     {"query": ..., "user_id": ..., "limit": N}
    DELETE /memories?user_id=...

This module serves that exact interface from one process, backed by either

* ``BENCH_BACKEND=mem0`` -- the in-process ``mem0.Memory`` SDK (no Docker/Qdrant
  service; local on-disk Qdrant), or
* ``BENCH_BACKEND=trw``  -- ``trw_memory.MemoryClient``, one namespace per user.

Running the *unmodified* upstream runner against both backends through the same
shim is what makes the comparison apples-to-apples: identical dataset parsing,
chunking, answerer prompt, judge prompt, cutoffs and metrics.

Both backends are pinned to the same embedding model (``BENCH_EMBED_MODEL``,
default ``all-MiniLM-L6-v2``) so the comparison isolates the memory-system
design (extraction, storage, retrieval) from the embedder.

Env knobs::

    BENCH_BACKEND      mem0 | trw                 (required)
    BENCH_DATA_DIR     where backend state lives   (default: ./bench-state/<backend>)
    BENCH_EMBED_MODEL  sentence-transformers name  (default: all-MiniLM-L6-v2)
    BENCH_LLM_MODEL    model for mem0 fact extraction (default: llama3.1:latest)
    BENCH_EXTRACT_BASE_URL / BENCH_EXTRACT_API_KEY
                       OpenAI-compatible endpoint for mem0 extraction (default: local Ollama)
    OLLAMA_BASE_URL    default http://localhost:11434
    BENCH_TRW_CONTEXT  preceding turns carried as context per stored turn (default 1)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger("bench-shim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BACKEND = os.getenv("BENCH_BACKEND", "").strip().lower()
if BACKEND not in {"mem0", "trw"}:
    raise SystemExit("BENCH_BACKEND must be 'mem0' or 'trw'")
DATA_DIR = Path(os.getenv("BENCH_DATA_DIR", f"./bench-state/{BACKEND}")).resolve()
EMBED_MODEL = os.getenv("BENCH_EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("BENCH_LLM_MODEL", "llama3.1:latest")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _mem0_llm_config() -> dict[str, Any]:
    """mem0's extraction LLM: local Ollama by default, or any OpenAI-compatible
    endpoint (OpenRouter, OpenAI, vLLM) when BENCH_EXTRACT_BASE_URL is set."""
    base_url = os.getenv("BENCH_EXTRACT_BASE_URL", "").strip()
    if not base_url:
        return {"provider": "ollama", "config": {"model": LLM_MODEL, "ollama_base_url": OLLAMA_URL, "temperature": 0.1}}
    api_key = os.getenv("BENCH_EXTRACT_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("BENCH_EXTRACT_API_KEY is required with BENCH_EXTRACT_BASE_URL")
    return {
        "provider": "openai",
        "config": {"model": LLM_MODEL, "openai_base_url": base_url, "api_key": api_key, "temperature": 0.1},
    }


def _epoch_to_iso(ts: int | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Backend: mem0 (in-process SDK)
# ---------------------------------------------------------------------------


class Mem0Backend:
    """Drives ``mem0.Memory`` in-process; calls are serialised through a pool."""

    def __init__(self) -> None:
        from mem0 import Memory

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cfg: dict[str, Any] = {
            "version": "v1.1",
            "llm": _mem0_llm_config(),
            "embedder": {"provider": "huggingface", "config": {"model": EMBED_MODEL}},
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "memories",
                    "path": str(DATA_DIR / "qdrant"),
                    "on_disk": True,
                },
            },
            "history_db_path": str(DATA_DIR / "history.db"),
        }
        # Probe embedding dims so qdrant creates the right-sized collection.
        from sentence_transformers import SentenceTransformer

        dims = SentenceTransformer(EMBED_MODEL).get_sentence_embedding_dimension()
        cfg["embedder"]["config"]["embedding_dims"] = dims
        cfg["vector_store"]["config"]["embedding_model_dims"] = dims
        self.memory = Memory.from_config(cfg)
        # mem0's add() is CPU+LLM bound and not thread-safe around the local
        # qdrant client, so serialise writes; searches can overlap.
        self._write_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=8)
        log.info("mem0 backend ready: llm=%s embedder=%s dims=%d dir=%s", LLM_MODEL, EMBED_MODEL, dims, DATA_DIR)

    async def add(self, messages: list[dict[str, Any]], user_id: str, timestamp: int | None) -> dict[str, Any]:
        def _do() -> Any:
            with self._write_lock:
                kwargs: dict[str, Any] = {"user_id": user_id}
                if timestamp is not None:
                    # The OSS SDK rejects ``timestamp=``; upstream's docker shim
                    # silently drops it, leaving every memory dated at ingest
                    # time. mem0 honours a caller-supplied ``created_at`` in
                    # metadata, so carry the session date through that route:
                    # the answerer prompt then sees real conversation dates,
                    # exactly as it would against Mem0 Cloud.
                    kwargs["metadata"] = {"created_at": _epoch_to_iso(timestamp)}
                return self.memory.add(messages, **kwargs)

        return await asyncio.get_running_loop().run_in_executor(self._pool, _do)

    async def search(self, query: str, user_id: str, limit: int) -> dict[str, Any]:
        def _do() -> Any:
            return self.memory.search(query, filters={"user_id": user_id}, top_k=limit)

        return await asyncio.get_running_loop().run_in_executor(self._pool, _do)

    async def delete_user(self, user_id: str) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, lambda: self.memory.delete_all(user_id=user_id))


# ---------------------------------------------------------------------------
# Backend: trw-memory
# ---------------------------------------------------------------------------


class TrwBackend:
    """One ``MemoryClient`` namespace per benchmark user; ``store_conversation`` ingestion.

    Every turn is stored verbatim (speaker-prefixed) with the session date in
    metadata and the preceding ``BENCH_TRW_CONTEXT`` turns of the same session
    carried as context. No LLM is involved at ingest time, so every gain over
    the raw baseline comes from the memory system itself. The upstream runner
    posts one turn per request, so the shim remembers each user's last turns
    and hands them to ``store_conversation`` as ``preceding``; a new session
    (different timestamp) resets that window, as it would in a live agent.
    """

    def __init__(self) -> None:
        os.environ.setdefault("MEMORY_STORAGE_PATH", str(DATA_DIR / "store"))
        os.environ.setdefault("MEMORY_EMBEDDING_MODEL", EMBED_MODEL)
        from trw_memory.client import MemoryClient

        self._client_cls = MemoryClient
        self._clients: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._context_turns = int(os.getenv("BENCH_TRW_CONTEXT", "1"))
        self._recent: dict[str, tuple[str, list[str]]] = {}  # ns -> (session key, last turns)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log.info("trw backend ready: embedder=%s context=%d dir=%s", EMBED_MODEL, self._context_turns, DATA_DIR)

    @staticmethod
    def _ns(user_id: str) -> str:
        return "project:" + "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in user_id)

    async def _client(self, user_id: str) -> Any:
        ns = self._ns(user_id)
        async with self._lock:
            client = self._clients.get(ns)
            if client is None:
                client = self._client_cls(ns, mode="local")
                self._clients[ns] = client
        return client

    async def add(self, messages: list[dict[str, Any]], user_id: str, timestamp: int | None) -> dict[str, Any]:
        client = await self._client(user_id)
        ns = self._ns(user_id)
        observed = _epoch_to_iso(timestamp)
        session_key, preceding = self._recent.get(ns, ("", []))
        if session_key != observed:
            preceding = []  # new session: context does not cross the session boundary
        turns = [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages]
        summary = await client.store_conversation(
            turns, context_turns=self._context_turns, preceding=preceding, observed_at=observed or None
        )
        texts = [t["content"].strip() for t in turns if t["content"].strip()]
        self._recent[ns] = (observed, (preceding + texts)[-max(self._context_turns, 1) :])
        results = [
            {"memory": text, "event": "ADD", "id": item.memory_id}
            for text, item in zip(texts, summary.items, strict=False)
        ]
        return {"results": results}

    async def search(self, query: str, user_id: str, limit: int) -> dict[str, Any]:
        client = await self._client(user_id)
        # Each benchmark user is an unrelated namespace; org-sibling discovery
        # would open every other user's store per query for nothing.
        rows = await client.recall(query, limit=limit, include_org_memories=False)
        out = []
        for r in rows:
            meta = r.get("metadata") or {}
            out.append(
                {
                    "id": r.get("memory_id", ""),
                    "memory": r.get("content", ""),
                    "score": float(r.get("score", 0.0)),
                    "created_at": meta.get("observed_at") or r.get("created_at", ""),
                }
            )
        return {"results": out}

    async def delete_user(self, user_id: str) -> None:
        ns = self._ns(user_id)
        async with self._lock:
            client = self._clients.pop(ns, None)
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                res = close()
                if asyncio.iscoroutine(res):
                    await res
        ns_dir = DATA_DIR / "store" / ns.replace(":", "_")
        if ns_dir.exists():
            shutil.rmtree(ns_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# FastAPI surface (mirrors memory-benchmarks/docker/mem0/main.py)
# ---------------------------------------------------------------------------


class AddRequest(BaseModel):
    messages: list[dict[str, Any]]
    user_id: str
    timestamp: int | None = None
    metadata: dict[str, Any] | None = None
    custom_instructions: str | None = None


class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = Field(default=100, ge=1, le=1000)
    rerank: bool = False


app = FastAPI(title=f"bench shim ({BACKEND})")
backend: Any = Mem0Backend() if BACKEND == "mem0" else TrwBackend()
_stats = {"add": 0, "search": 0, "add_ms": 0.0, "search_ms": 0.0}


@app.post("/memories")
async def add_memories(req: AddRequest) -> Any:
    t = time.monotonic()
    try:
        out = await backend.add(req.messages, req.user_id, req.timestamp)
    except Exception as exc:  # surface as 500 like upstream
        log.exception("add failed")
        raise HTTPException(500, str(exc)) from exc
    _stats["add"] += 1
    _stats["add_ms"] += (time.monotonic() - t) * 1000
    return out


@app.post("/search")
async def search_memories(req: SearchRequest) -> Any:
    t = time.monotonic()
    try:
        out = await backend.search(req.query, req.user_id, req.limit)
    except Exception as exc:
        log.exception("search failed")
        raise HTTPException(500, str(exc)) from exc
    _stats["search"] += 1
    _stats["search_ms"] += (time.monotonic() - t) * 1000
    return out


@app.delete("/memories")
async def delete_all(user_id: str = Query(...)) -> Any:
    await backend.delete_user(user_id)
    return {"message": "All memories deleted"}


@app.get("/health")
def health() -> Any:
    return {"status": "ok", "backend": BACKEND, "stats": _stats}


def main() -> None:
    import uvicorn

    port = int(os.getenv("BENCH_PORT", "8888"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
