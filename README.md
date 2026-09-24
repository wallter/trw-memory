# trw-memory

**Local-first memory for AI agents: a SQLite file on your machine, keyword and vector recall.**

trw-memory gives your AI agents long-term memory that runs locally. Memories live in a SQLite file on your machine, recall combines keyword ranking, local vector search and a re-ranker, and storing a conversation calls no LLM by default. On all 1,540 LOCOMO questions, trw-memory 2.0.0 answered 88.9% correctly against 84.1% for mem0 OSS 2.0.20, a statistically significant lead ([benchmarks](#benchmarks)). Use it from Python, the command line, or any MCP client. It is the memory engine behind [TRW Framework](https://trwframework.com)'s MCP server, trw-mcp, and works on its own.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](https://trwframework.com/license)
[![Docs](https://img.shields.io/badge/docs-trwframework.com-blue)](https://trwframework.com/docs)

> **Status:** alpha, source-available under BSL 1.1. The public API may change in future releases; test an upgrade before you roll it out.

**[What's new](#whats-new-in-3x)** · **[Quick start](#install-and-quick-start)** · **[Benchmarks](#benchmarks)** · **[How it works](#how-it-works)** · **[MCP server](#mcp-memory-server)** · **[Network and security](#telemetry-and-network-behavior)** · **[Upgrading](#upgrading)** · **[FAQ](#faq)**

## Why trw-memory

- **More right answers.** On the LOCOMO benchmark, trw-memory 2.0.0 answered 88.9% of 1,540 questions correctly against 84.1% for mem0 OSS, ahead in all ten conversations. On LongMemEval, the evidence for 93.8% of questions lands in its top 10 results.
- **No LLM calls to store.** `store_conversation()` keeps each chat turn as written, with its date and the turn it replied to, so saving memory costs no API calls (by default). The reader does the inference at recall time.
- **Runs on your machine.** Memories live in a local SQLite file. There is no hosted service, no account and no usage tracking. Once the models are cached, `TRW_OFFLINE=1` keeps it off the network; remote sync and the decision judge are opt-in and off by default.
- **Hybrid recall.** Keyword ranking (BM25) and local vector search are fused, then a local cross-encoder re-ranks the results.
- **Stale knowledge stays out.** Every memory has a lifecycle status, and superseded or retired memories are excluded from recall by default. Near-duplicates are merged and old memories move to colder storage tiers.
- **Three ways in.** An async Python SDK (`MemoryClient`), a CLI (`trw-memory`), and an MCP server (`trw-memory-server`) that any MCP client can launch.

## Install and quick start

```bash
# Core: SQLite store, full-text keyword search, sqlite-vec vector storage, MCP server
pip install trw-memory

# Recommended: adds the local embedding model runtime and BM25, which turn on
# vector search and the re-ranker
pip install "trw-memory[all]"
```

Without the `[embeddings]` extra (included in `[all]`) nothing produces vectors, so recall is keyword-only. The embedding model (`BAAI/bge-small-en-v1.5`, about 130 MB) and the re-ranker download from Hugging Face on first use unless they are already cached; see [Telemetry and network behavior](#telemetry-and-network-behavior) to pre-fetch them or block the download.

**Supported platforms:** macOS arm64/x86_64, manylinux x86_64/aarch64 and Windows x86_64, the platforms `sqlite-vec` publishes wheels for. CPython 3.10 through 3.14.

### Python SDK

```python
import asyncio

from trw_memory.client import MemoryClient


async def main() -> None:
    async with MemoryClient(namespace="project:my-app") as client:
        await client.store(
            "Pydantic v2 requires use_enum_values=True for YAML round-trip",
            tags=["pydantic", "gotcha"],
            importance=0.8,
        )

        results = await client.recall("pydantic serialization", limit=10)
        high_impact = await client.search(min_importance=0.7, tags=["gotcha"])
        print(results, high_impact)


asyncio.run(main())
```

The SDK stores memories in `.memory/` under the current directory by default (set `MEMORY_STORAGE_PATH` to move it). `MemoryClient` also has `store_many()` and `bulk_store()` for batch writes, `search_fts()` for keyword-only lookup, `forget()`, `reembed()`, and `audit_learning()` / `review_quarantined()` for lifecycle and security workflows.

### Conversation memory

```python
# inside `async with MemoryClient(...) as client:`
turns = [
    {"role": "user", "speaker": "Caroline", "content": "I went to a LGBTQ support group yesterday."},
    {"role": "assistant", "speaker": "Melanie", "content": "That's great! What did it look like?"},
]
summary = await client.store_conversation(turns, observed_at="2023-05-08T13:56:00+00:00", session_id="s1")
rows = await client.recall("what did the support group look like", limit=5)
```

Each turn's text is stored as written, apart from trimmed whitespace and a `Speaker: ` prefix when a speaker is given, along with the preceding `context_turns` (default 1), so a reply like "What did it look like?" can be found by what it replied to. Blank turns are skipped, and rows the write gate rejects (PII, poisoning, schema) are counted in the returned summary's `rejected` and `rejected_reasons`. Feeding a conversation in chunks? Pass the last turns you already stored as `preceding=`.

### Agent framework integration

```python
from trw_memory.client import MemoryClient

client = MemoryClient(namespace="project:my-app")

# Register tools with any agent that has a register_tool() or tool() API
client.register_tools(agent)

# Or inject recalled memories into a handler
@client.auto_recall(query_from="prompt")
async def handle_prompt(prompt: str, recalled_memories: list | None = None) -> str:
    recalled_memories = recalled_memories or []
    return f"Found {len(recalled_memories)} relevant memories"
```

### CLI

```bash
trw-memory store --summary "Always use connection pooling for PostgreSQL" --tags db --tags performance --importance 0.8
trw-memory recall "database optimization" --limit 5
trw-memory search --tags security --status active
trw-memory consolidate --namespace project:my-app --dry-run
trw-memory export --format json > memories.json
trw-memory import memories.json --namespace project:new-app
trw-memory forget M-abc12345 --namespace project:my-app
trw-memory status

# Store-file maintenance (stop the daemon first)
trw-memory reembed --namespace project:my-app    # re-encode vectors after a model change
trw-memory restore --from-cold                   # or --from-snapshot latest
trw-memory snapshot create --tier daily          # also: snapshot list, snapshot rotate

# Wiki lint and the explicit code index
trw-memory wiki-lint pages.json
trw-memory code-index ./src
trw-memory code-search ./src "hybrid_search" --language python --limit 5
trw-memory code-symbol ./src MemoryClient
```

`store`, `recall`, `search`, `forget`, `consolidate`, `export` and `status` run over the [loopback daemon](#loopback-daemon-serve-http). They start one if none is running and present this checkout's grant, which `trw-mcp memory token` mints; without a grant, or for a namespace outside it, the command prints the remedy and exits 1. `--namespace` defaults to the checkout's pinned `project_namespace` (pass `--namespace default` for the old default). `import`, `reembed`, `restore` and `snapshot create` open the store file directly, so they refuse while a daemon runs.

trw-memory itself has no command that mints a grant; `trw-mcp memory token` is the one that does. On a standalone install, use the Python SDK or `trw-memory-server` for everyday reads and writes.

`export` covers one namespace and holds it in memory; it is not a streaming backup and does not include stored vectors. Export from an unchanged store: pagination is not a snapshot across concurrent writes. Use the `snapshot` commands for database backups. `import` keeps export-format rows whole, re-screens every row through the write gate, writes rejected rows to `<file>.rejected.jsonl`, and exits 1 when any row is rejected.

### Low-level backend access

```python
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.models.memory import MemoryEntry

backend = SQLiteBackend(db_path=".trw/memory.db")
entry = MemoryEntry(id="M-abc12345", content="Use WAL mode for concurrent readers", namespace="default")
backend.store(entry)
results = backend.search("query", top_k=10, namespace="default")
```

## What's new in 3.x

<!-- whats-new: 3.1.0 -->

- **One store for all your projects.** The CLI and trw-mcp share one local daemon per user account; each checkout's grant limits it to its own namespaces.
- **Re-ranked recall everywhere.** With `[embeddings]` installed, the SDK, the daemon and trw-mcp's `trw_recall` all re-rank with the cross-encoder.
- **Older memories stay reachable.** Recall adds full-text matches from beyond the 1,000 most recent rows.
- **Vector storage built in.** `sqlite-vec` ships with every install; the `[vectors]` extra is gone.
- **Offline keeps working.** Without a cached embedding model, the daemon's recall and store fall back to keyword search instead of failing.
- **Sturdier daemon.** An I/O error no longer quarantines a healthy store (Python 3.11+), and a daemon that never starts is cleaned up.
- **With trw-mcp 6.x:** the installer adds embeddings by default, and `doctor` reports whether each retrieval component works.

3.0.0 is a breaking release for 2.x users: see [Upgrading](#upgrading-from-200). Full list: [CHANGELOG.md](https://github.com/wallter/trw-memory/blob/main/CHANGELOG.md).

## Benchmarks

Each result names the trw-memory version it was measured on. Harnesses: [`benchmarks/locomo/`](https://github.com/wallter/trw-memory/blob/main/benchmarks/locomo/README.md), [`benchmarks/longmemeval/`](https://github.com/wallter/trw-memory/blob/main/benchmarks/longmemeval/README.md), [`benchmarks/engmem/`](https://github.com/wallter/trw-memory/blob/main/benchmarks/engmem/README.md).

### More right answers than mem0 on LOCOMO

| All 1,540 LOCOMO questions, judged correct | mem0 OSS 2.0.20 | trw-memory 2.0.0 |
|---|:---:|:---:|
| Answer accuracy | 84.1% | **88.9%** |

trw-memory answered 4.8 points more questions correctly (paired 95% CI +2.9 to +6.8; McNemar p < 0.0001) and led in all ten conversations. It made no LLM calls to store the 5,882 conversation turns; mem0 makes an extraction call for every write.

<sub>Method: mem0's evaluation harness (commit `4b61c5d`) on all ten [LOCOMO](https://github.com/snap-research/locomo) conversations; each system answered from its top 10 memories with the same reader and judge (`gpt-4o-mini`) and the same prompts. trw-memory 2.0.0 with `bge-small-en-v1.5` vs self-hosted `mem0ai` 2.0.20 with `all-MiniLM-L6-v2`; the embedders differed. September 2026.</sub>

### Finds the evidence on LOCOMO and LongMemEval

With no LLM in the loop, how often the turn that holds the answer comes back in the top results (trw-memory 2.0.0 defaults):

| Benchmark | Questions | In top 10 | In top 50 |
|---|:---:|:---:|:---:|
| LongMemEval_S (cleaned) | 470 | **93.8%** | **97.4%** |
| LOCOMO, all ten conversations | 1,540 | **85.7%** | **92.7%** |

These are retrieval hit rates, a different measure from the judged answer accuracy above. Every retrieval change has to hold up on both benchmarks, question by question, before it ships.

### Keeps finding the right record as the store grows

On EngMem, a synthetic benchmark of engineering learnings where newer records supersede older ones, the 3.0.0 recall path found every required record for 16 of 16 queries at 1,000 rows and 14 of 16 at 5,000 and 20,000 rows. It never returned a retired record in its top 10, and median recall stayed near 300 ms at 20,000 rows.

### Costs that stay small as the store grows

These operations used to slow down sharply as the store grew, or as more projects shared it. Profiling in 1.0.0 and 2.0.0 cut them by 2x to over 200x (single runs, one Apple-silicon machine):

| Operation | Before | After |
|---|---|---|
| Ingest per row, 4th LOCOMO conversation into a shared store | 1,464 ms | 15 ms |
| Store one entry with 20 sibling project namespaces present | 708 ms | 15 ms |
| Access-time sidecar write per recall, 5,000 / 20,000 rows | 144 / 289 ms | 1.2 / 1.3 ms |
| Warm-row scan per recall, 5,000 / 20,000 rows | 11.3 / 60.5 ms | 0.3 / 3.2 ms |
| Store + background graph enrichment, rows 901-1,200 | 15.85 ms | 8.38 ms |
| Similarity-edge enrichment at 10,000 rows | 12.4 ms | 2.3 ms |

### Hybrid retrieval vs a single ranker

On 889 queries over real engineering learnings, fusing BM25 and vectors had the highest scores: Recall@10 0.938 against 0.914 for vectors alone and 0.869 for BM25 alone (point estimates; significance not assessed). On 175 near-duplicate "rediscoveries", hybrid recall would have surfaced the earlier record 94.3% of the time against 72.0% for keyword search alone, with non-overlapping 95% confidence intervals. Measured on trw-memory 0.9.12.

### Memory lets agents finish work they otherwise cannot

On a controlled benchmark where each task needs a fact from an earlier session, agents with memory solved 58 of 58 tasks and agents without it solved 0 of 50 (paired McNemar p = 3.6×10⁻¹⁵ over 49 matched pairs), and the result replicated on a second model family.

## How it works

### Recall pipeline

```
Query --> BM25 (keyword, rank-bm25) --+
                                       +--> RRF fusion --> cross-encoder re-rank --> results
Query --> Dense (cosine, sqlite-vec) --+
```

BM25 drops function words from the query and suffix-stems tokens on both sides ("researched" meets "research"). Dense search scores only vectors from the active embedding model's space. Reciprocal Rank Fusion merges the two lists (the constant `k` is `MemoryConfig.rrf_k`, env `MEMORY_RRF_K`; the default is tuned and may change between releases). A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) then re-ranks the top `recall_rerank_candidates` and drops rows it scores below -8, except the top `max(5, ceil(limit / 2))`, which are always kept (5 rows at the default `limit=10`, 25 at `limit=50`). So `recall(limit=N)` can return fewer than N rows. There is no switch to turn re-ranking off; it is skipped only when `sentence-transformers` or the cached model is unavailable, and recall then keeps fusion order.

The pipeline degrades instead of failing: without BM25 only dense search runs, and the reverse; with neither, recall falls back to the store's built-in keyword search.

Candidates come from two places: the most recently updated rows of the namespace, plus older active rows that full-text search matches (since 3.0.0). The recency pool holds `max(limit * 5, hybrid_search_candidate_pool_size)` rows for `MemoryClient.recall()` and `max(limit * 25, hybrid_search_candidate_pool_size)` for the daemon's `memory_recall` and trw-mcp's `trw_recall`, which rank five times deeper; the pool-size default is 1000. An older row that shares no keyword with the query can still be missed, so an empty result is not evidence of absence. Raising `MEMORY_HYBRID_SEARCH_CANDIDATE_POOL_SIZE` widens the recency pool at a real cost: on a 6,500-row namespace, warm recall took 139.6 ms at 1000 and 1045.8 ms at 10000.

### Scoring and lifecycle

- **Utility score:** author-assigned impact (0.0-1.0), multiplied by `0.95 ** recall_count` (never below `feedback_decay_min_factor`, default 0.5), with an Ebbinghaus forgetting curve applied at query time (a higher recurrence count slows it) and a boost for recent access. There is no reward or helpful/unhelpful feedback signal (removed in 3.0.0).
- **Status:** retired and superseded memories are excluded from recall by default.
- **Deduplication:** near-duplicates are merged at a cosine threshold (0.85 on the reference scale, calibrated per embedding model).
- **Consolidation:** episodic-to-semantic consolidation clusters related memories and summarises each cluster with a longest-content heuristic. No LLM is involved.
- **Knowledge graph:** tag co-occurrence and similarity edges, BFS traversal, importance boost and decay.
- **Tiers:** hot (in-memory LRU, design target under 1 ms), warm (SQLite plus a JSONL sidecar, under 50 ms) and cold (YAML archive partitioned by year and month, under 200 ms). The targets are for the tier lookup, not end-to-end recall. Cold hits are promoted back to warm within the same recall, and `TierManager.sweep()` applies the archive and purge policy.
- **Remote sync:** optional publish and fetch across installations, with vector-clock conflict resolution and SSE live updates. Off by default.

### Security features

| Feature | Implementation |
|---------|---------------|
| Encryption at rest | Optional SQLCipher whole-database encryption (AES-256-CBC), keyed by an HKDF-SHA256 per-namespace derivation from the master key. It works for per-namespace stores (the Python SDK, `trw-memory-server serve stdio`), not the loopback daemon, which refuses to start with `encryption_enabled` because it keeps every namespace in one file; so the daemon-backed CLI verbs and trw-mcp cannot use it today. There is no key-rotation path (removed in 3.0.0). |
| PII detection | Regex patterns (email, phone, SSN, credit card, API keys) plus Shannon entropy. The store path **blocks** writes that contain a recognised credential (API keys and provider access tokens) and **records** every other detection as metadata without rewriting your text. Masking happens at the publish boundary (`strip_pii`). |
| Poisoning defense | Z-score anomaly detection on frequency, size and content patterns. Observe mode by default; `enforce` is opt-in. |
| Access control | Role-based (admin/editor/viewer) per namespace |
| Audit trail | Append-only security event log |

### Package layout

| Path | Responsibility |
|------|---------------|
| `client.py` | `MemoryClient`, the recommended entry point |
| `cli.py` | The `trw-memory` command |
| `server.py`, `tools/`, `daemon/` | The MCP server, its tools, and the loopback daemon |
| `storage/` | SQLite backend (WAL, sqlite-vec vectors, snapshots, recovery) and the YAML backend |
| `retrieval/` | BM25, dense search, RRF fusion, re-ranking and `hybrid_search()` |
| `lifecycle/` | Utility scoring, dedup, consolidation, hot/warm/cold tiers |
| `graph.py`, `embeddings/`, `sync/`, `security/`, `code_index/`, `wiki/` | Knowledge graph, embedding providers, remote sync, security, code index, wiki lint |

For the full, current layout, browse `src/trw_memory/`.

## MCP memory server

The MCP server ships with the core install:

```bash
trw-memory-server              # stdio transport
trw-memory-server serve http   # the per-user loopback daemon
```

To wire it into an MCP client (Claude Code, Cursor, Claude Desktop and others use this shape):

```json
{
  "mcpServers": {
    "memory": { "command": "trw-memory-server" }
  }
}
```

The everyday tools:

| Tool | Purpose |
|------|---------|
| `memory_store` | Store an entry, and a vector when embeddings are available (offline without a cached model, the row is stored without one). A refused write comes back as a status (`invalid`, `blocked` or `rate_limited`), not an MCP error |
| `memory_recall` | Hybrid retrieval, with optional graph traversal; the same ranking as trw-mcp's `trw_recall` |
| `memory_search` | List and filter by status and tags, with pagination |
| `memory_get`, `memory_update`, `memory_forget` | Read, change or delete entries |
| `memory_consolidate` | Episodic-to-semantic consolidation |
| `memory_maintain` | Run decay, consolidation and WAL checkpointing for a namespace |
| `memory_status` | Store statistics and health; `security_settings_only=True` reports the daemon-wide security settings |
| `memory_audit`, `memory_review`, `memory_quarantine_list` | Provenance audit and quarantine review |
| `memory_code_index`, `memory_code_search`, `memory_code_symbol` | The explicit code index |
| `memory_wiki_lint` | Lint wiki pages for missing targets, backlinks and provenance gaps |

Graph, sync, namespace-administration and import tools are also registered; `REGISTERED_TOOL_NAMES` in `server.py` is the complete list.

### Loopback daemon (`serve http`)

`trw-memory-server serve http` runs one process per operating-system user, serving the same MCP tools over `streamable-http` on 127.0.0.1, authenticated by per-checkout namespace grants. The port is ephemeral and published in a 0600 `daemon.json` beside the store, so clients discover it instead of hardcoding it. The daemon keeps every namespace in one store, by default `~/.trw/memory/memory.db` (`$XDG_DATA_HOME/trw/memory/` or `$TRW_USER_DIR/memory/` when set). A memory client that finds no daemon starts one, and an idle daemon exits.

**Trust boundary: a token reaches only its grant.** `trw-mcp memory token` mints a token for the calling checkout's project namespace plus `user:local`. Without `--namespace` it grants the pinned `project_namespace` only when that pin matches the namespace derived from the checkout's location; for a moved checkout, name it with `--namespace`. The daemon keeps only the token's sha256 digest, in the 0600 `daemon-grants.json`; the raw token lives in that checkout's `.trw/runtime/memory-token`. Every namespaced call is checked against the grant before RBAC, so a request for any other namespace is refused even with RBAC off. A leftover Slice A `daemon-token` (one all-namespace bearer) makes the daemon refuse to start; `trw-mcp memory token --migrate` deletes it.

**Security settings are daemon-wide.** RBAC, the recall filter, canary, poisoning, trust-scoring and provenance settings come from the environment the daemon starts from.

**Concurrency: four workers.** Each `memory_recall`, `memory_store` and `memory_maintain` call runs its synchronous work in a bounded thread pool (`OFFLOAD_MAX_WORKERS = 4`), with its own SQLite connection. Four calls make progress at once; the fifth queues, and the queue is unbounded. A request cancelled after it starts still runs to completion; only the result is discarded.

**Shutdown.** On SIGTERM or SIGINT the daemon cancels queued calls that have not started, waits up to 5 seconds for running calls (`OFFLOAD_SHUTDOWN_GRACE_SECONDS`), then removes its discovery record whether or not they finished. It is a bounded wait, not a guaranteed drain. After a signal it re-delivers that signal with the default handler, which ends the process; after an idle shutdown there is no signal to re-deliver, so a call still running past the 5 seconds can keep the process alive until it returns. The record is removed only when it names this process and the start time this process wrote, so a slow exit cannot delete a successor's record.

**Maintenance.** A daemon has no session end, so decay, consolidation and WAL checkpointing never run on their own. `memory_maintain(namespace)` triggers them (trw-mcp calls it at delivery) and records `last_attempted_at` / `last_maintained_at` per namespace in `maintenance.json`. Consolidation is namespace-scoped; the decay pass and the WAL checkpoint act on the whole store.

## Using trw-memory with trw-mcp

[trw-mcp](https://github.com/wallter/trw-mcp) is the MCP server of [TRW Framework](https://trwframework.com), and trw-memory is its memory backend. trw-mcp 6.1.0 requires trw-memory 3.1.0 or later 3.x.

- Every checkout reaches memory through the loopback daemon: `trw_learn` writes through `memory_store`, and `trw_recall` and `memory_recall` rank with the same `retrieval.recall_policy`.
- One store per user account (under `$TRW_USER_DIR`, `$XDG_DATA_HOME/trw` or `~/.trw`) holds every checkout's namespace. Each checkout has its own `project_namespace` and a grant in `.trw/runtime/memory-token`; portable learnings go to `user:local`.
- trw-mcp no longer reads or writes a checkout's `.trw/memory/memory.db`. `trw-mcp memory migrate --to user --apply` moves an existing one into the daemon; trw-mcp 6.1.0's installer runs or offers that migration.
- trw-mcp refuses a daemon whose daemon-wide security settings differ from its own.
- Vector recall needs the `[embeddings]` extra in the environment the daemon runs from; without it the daemon recalls by keyword. trw-mcp 6.1.0's installer adds it by default, and `trw-mcp doctor` reports whether each retrieval component is active.

## Telemetry and network behavior

trw-memory is **local-first**: with the default configuration all data lives in a local SQLite store (and an optional YAML sidecar). With the default configuration it makes **no outbound network calls** except the model downloads below. Two opt-in features also reach the network: remote sync and the decision judge. There is no usage tracking or content phone-home.

### What can touch the network, when, and how to turn it off

| Surface | When | Default | Opt-out / control |
|---------|------|---------|-------------------|
| **Embedding model download** | Only with the `[embeddings]` extra installed, and only when the model (`BAAI/bge-small-en-v1.5` by default: 33M parameters, 384 dimensions, about 130 MB; set `MEMORY_EMBEDDING_MODEL` to change it) is **not** already complete in your local Hugging Face cache. The loader probes the cache first and forces `local_files_only=True` when the snapshot is complete, so a warm cache makes **zero** huggingface.co requests | enabled when the extra is present | `TRW_OFFLINE=1` / `HF_HUB_OFFLINE=1`, or `local_only: true` (alias `memory_local_only`), force `local_files_only` so no download is attempted; a disclosure log line precedes any network-capable load |
| **Re-ranker load** | Only with the `[embeddings]` extra installed, on the first recall in a process. Unlike the embedder, the re-ranker loader does not probe the cache first: without an offline switch it loads `cross-encoder/ms-marco-MiniLM-L-6-v2` with `local_files_only=False`, so Hugging Face may be contacted even when the files are cached, and they are downloaded when they are not. Under an offline switch it loads from the cache only; an uncached re-ranker is then skipped and recall keeps fusion order (no download, no error) | enabled when the extra is present | `TRW_OFFLINE=1`, `HF_HUB_OFFLINE=1` or `local_only: true` (loads with `local_files_only=True`); a disclosure log line precedes any network-capable load |
| **Remote sync / publish** | Only when `sync_enabled=true` AND `local_only=false` | **off** | leave sync disabled, or set `local_only: true` to hard-block it |
| **Decision judge** (`trw_memory.decisions`) | Only when enabled **and** an `OPENROUTER_API_KEY` is present. For the store path, enabled means `TRW_JEV_ENABLED` in the process environment or `assess_enabled` in the user's `~/.trw/config.yaml`; that path does not read project `.trw/config.yaml` or `.env`, and it takes the key from the process environment only. `python -m trw_memory.decisions.cli` also reads project settings via `--dotenv`. When enabled, the store path's poisoning screen sends each written entry's text, after credential and PII redaction, to the judge endpoint (default `https://openrouter.ai`, an allowlisted https host) as a shadow check that never changes the outcome; `python -m trw_memory.decisions.cli` calls it directly | **off** | leave it disabled: unset `TRW_JEV_ENABLED` / `assess_enabled` at every layer, or set `TRW_JEV_ENABLED=false` in the process environment, which overrides the others. `local_only` does **not** turn it off |

`learning_sharing_enabled` and `platform_telemetry_enabled` govern learning-content publishing and usage telemetry; **neither gates the model download**. Model egress is independent of the consent flags — it is governed by the local cache, the offline switches and `local_only`. Setting `local_only: true` is the hard block for model downloads and sync: a validator forces `sync_enabled=False`, clears `sync_namespace`, `platform_url` and `platform_api_key`, and pins `rbac_mode="local"`, so sync cannot be re-enabled while it is set. It does **not** disable the decision judge, whose on/off switch is separate; an offline deployment must also leave the judge disabled.

**Offline with an uncached embedding model**, behaviour depends on the entry point. Since 3.1.0 the daemon's `memory_recall`, `memory_store` and `memory_consolidate` (and so the CLI and trw-mcp) run keyword-only, store rows without vectors, and report `"dense": "unavailable: <reason>"` in the recall response. The in-process `MemoryClient` raises `LocalOnlyViolationError` with instructions to pre-download the model. `RemoteCodeNotPermittedError` raises in both.

### Environment-variable inventory

| Variable | Purpose | Default |
|----------|---------|---------|
| `TRW_OFFLINE` | Master offline switch: blocks the embedding-model and re-ranker downloads | unset |
| `HF_HUB_OFFLINE` | Upstream huggingface_hub offline switch, also honoured | unset |
| `MEMORY_EMBEDDING_MODEL` | Sentence-transformers model for dense vectors. Changing it leaves stored vectors in the old model's space until `trw-memory reembed` re-encodes them (see [Changing the embedding model](#changing-the-embedding-model)) | `BAAI/bge-small-en-v1.5` |
| `MEMORY_STORAGE_PATH` | Where the Python SDK keeps its stores | `.memory/` |
| `MEMORY_*` | Engine settings validated by `MemoryConfig` (for example `MEMORY_LOCAL_ONLY`, `MEMORY_EMBEDDING_TRUST_REMOTE_CODE`, retrieval and lifecycle tuning) | per field |

### Security defaults

| Capability | Default | Notes |
|-----------|---------|-------|
| Encryption at rest | **off** (`encryption_enabled=False`) | opt-in SQLCipher whole-database encryption (AES-256-CBC), HKDF-SHA256 per-namespace keys; needs the `[encryption]` extra. Per-namespace stores only (SDK, `serve stdio`): the loopback daemon refuses to start with it enabled |
| PII detection | **on** (`pii_enabled=True`) | scans `content`, `detail`, `tags`, `evidence[]` and `Assertion.last_evidence` on the store path. Recognised credentials (API keys and provider access tokens, detected as `PIIType.API_KEY`, the only type in `BLOCKING_PII_TYPES`) **block the write** (`PIIBlockError`); every other type is recorded in `pii_types` metadata and stored **verbatim**. `pii_action` (default `warn`) is only reported by `memory_status`; it does not change what the store path does. Emails, IPs, SSNs, phone numbers and card numbers are masked at the publish boundary. Set `pii_custom_patterns` to opt in to local masking with your own regexes |
| Poisoning / size-anomaly detection | **observe** (`poisoning_detection_mode="observe"`) | records anomaly stats and telemetry but does **not** quarantine; `enforce` is opt-in. A caller-supplied `metadata['source']` cannot skip enforce-mode quarantine |
| Trust scoring | **observe** (`trust_scoring_mode="observe"`) | logs intake trust decisions; `enforce` and `strict` are opt-in |
| Provenance signing | **required** (`provenance_required=True`) | persisted rows carry a signed provenance hash chain |
| Canary tamper response | **halt** (`canary_fail_mode="halt"`) | seeded canaries are probed on recall; tampering halts by default (`degrade` and `log-only` are opt-in) |
| Remote sync / publishing | **off** (`sync_enabled=False`) | `local_only=True` hard-blocks it |
| Decision judge (LLM calls to OpenRouter) | **off** (`TRW_JEV_ENABLED` unset) | needs both an enable switch and `OPENROUTER_API_KEY`; not covered by `local_only` |
| Model remote-code execution | **off** (`embedding_trust_remote_code=False`) | the only input to sentence-transformers' `trust_remote_code`. A model repository that ships its own Python modules is refused with `RemoteCodeNotPermittedError`; set it `true` only for a repository you trust. The default model needs no remote code |
| `memory.db` permissions | `0600` | the store file is `chmod 0600` on creation; a non-POSIX platform logs a `db_chmod_failed` warning |

### Enterprise hardening recipe

```bash
export TRW_OFFLINE=1   # block the huggingface.co model download (local_files_only)
```

```yaml
# MemoryConfig
local_only: true       # hard-block remote sync + model download
```

```bash
export TRW_JEV_ENABLED=false   # local_only does not cover the decision judge; keep it off explicitly
```

For vector recall offline, populate the model cache **before** enabling either switch, in the same environment:

```bash
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

If the **embedding** model is not cached, the daemon's tools run keyword-only and the in-process `MemoryClient` raises `LocalOnlyViolationError` on its first embedding load. If only the re-ranker is uncached, recall still uses vectors and simply skips re-ranking. To run keyword-only on purpose, omit the `[embeddings]` extra. Verify that the on-disk `memory.db` is mode `0600`, that `TRW_JEV_ENABLED` / `assess_enabled` are not set in any layer, and that no outbound connection is attempted on first use.

## Upgrading

### Upgrading from 2.0.0

If you use trw-memory through trw-mcp, upgrade both together (trw-mcp 6.1.0 requires trw-memory 3.1.0 or later 3.x) and follow the [trw-mcp README](https://github.com/wallter/trw-mcp). For a standalone install:

1. **Keep a way back.** 3.x opens a 2.0.0 store in place and never alters its schema, so 2.0.0 can still open that store afterwards. A store that 3.x *creates* has no legacy columns, and 2.0.0 fails on it (`no such column: q_value`). Before upgrading, export each namespace with 2.0.0 (`trw-memory export --namespace <ns> --format json > <ns>.json`); to roll back, re-import into a store 2.0.0 creates. Export does not include stored vectors, and 2.0.0's `import` keeps only content, detail, tags and importance, under new ids.
2. **Install:** `pip install -U "trw-memory[all]==3.1.0"` (or with the extras you use). Drop `[vectors]` from your requirements; pip only warns about it.
3. **Restart the daemon**, if one runs, so it serves 3.x code, and set the daemon-wide security settings in its environment: `MEMORY_RBAC_ENABLED`, `MEMORY_DEFAULT_ROLE`, `MEMORY_NAMESPACE_ROLES`, `MEMORY_ENABLE_RECALL_FILTER`, `MEMORY_RECALL_FILTER_MODE`, `MEMORY_CANARY_FAIL_MODE`, `MEMORY_POISONING_DETECTION_MODE`, `MEMORY_ENABLE_TRUST_SCORING`, `MEMORY_TRUST_SCORING_MODE` and `MEMORY_PROVENANCE_REQUIRED`.
4. **Update scripts that call the CLI.** `store`, `recall`, `search`, `forget`, `consolidate`, `export` and `status` need a checkout grant (`trw-mcp memory token`). Pass `--namespace default` where you relied on the old default. `search` lost `--min-importance` and `--since` and gained `--status`. Stop the daemon before `import`, `reembed`, `restore` or `snapshot create`. Treat exit 1 from `import` as "some rows were rejected" and read `<file>.rejected.jsonl`.
5. **Update code that uses the Python API:**
   - Stop reading or writing `q_value`, `q_observations`, `helpful_count` and `unhelpful_count` on `MemoryEntry`.
   - Drop the `learning_id=` keyword from `compute_anchor_validity()`.
   - Pass `namespace=` to `increment_recall_access`, `increment_session_counts` and `record_recall_access`.
   - Call `acquire_candidates(limit=..., config=...)` in place of `pool_size` / `fts_top_k`.
   - Pass `admit=store_gate(config, backend)` to `fetch_shared_memories` in place of `backend=`.
   - Replace `apply_source_policy()` with `SourcePolicy.resolve(...).apply(rows)`, and `count_with_assertions` with `entries_with_assertions`.
   - Branch on `memory_store`'s `status` (`invalid`, `blocked`, `rate_limited`) where you caught an MCP error.
   - Remove imports of the deleted APIs (among them `PoisoningDetector`, `ProvenanceChain`, `rotate_master_key`, `security.encryption.rotate_key`, `seed_canaries`, `verify_canaries`, the bandit selectors and `trw_memory.adapters`); the [CHANGELOG](https://github.com/wallter/trw-memory/blob/main/CHANGELOG.md) lists them all.
6. **Remove retired settings:** `lifecycle_use_fsrs`, `key_rotation_backup`, `concurrent_writer_warn_threshold` (`memory_concurrent_writer_warn_threshold`) and, in 3.1.0, `q_learning_rate`. Each logs `retired_setting_ignored` and is ignored. At-rest encryption (`encryption_enabled`) is unchanged; there is no key-rotation path.

Expect recall order to change on the same store: ranking no longer blends in reward feedback.

### Changing the embedding model

The default embedding model has been `BAAI/bge-small-en-v1.5` since 1.0.0 (384 dimensions, like the earlier `all-MiniLM-L6-v2`; queries carry the model's search instruction, stored documents do not). Vectors written by another model, or written before vectors recorded which model produced them, live in a different embedding space, so dense recall **ignores them** instead of scoring across spaces. Until they are re-encoded, BM25 still ranks those rows whenever they reach the recall candidate pool, and each recall logs one `dense_vectors_excluded_embedding_space` warning with the number of vectors held back.

Re-encode each namespace once (idempotent and resumable; rows already in the active space are skipped, and each batch commits on its own):

```bash
trw-memory reembed --namespace default            # --batch-size 64, --format json
```

```python
async with MemoryClient(namespace="default") as client:
    counts = await client.reembed()
```

`reembed` works on a store the SDK opens in local mode (under your `storage_path`), not on the daemon's store at `~/.trw/memory/memory.db`; 3.x has no command that re-embeds the daemon's store, and the CLI refuses to run while a daemon runs. It honours `TRW_OFFLINE` / `HF_HUB_OFFLINE` / `local_only` like every other load, so pre-download the model first. To keep the previous model, set `MEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2`; a process only ever scores vectors from the space its embedder reports, so switching back and forth never mixes spaces.

### Older releases

- **0.18.0 retired HyPE question generation and HyDE query expansion.** Delete imports of `QuestionGenerator` and `NoOpQuestionGenerator`; remove the `question_generator`, `query_expansion` and `collapse_hype` arguments and the `hype_*` settings. Neutral legacy values still only warn in 3.x; a later breaking release will remove them. Recall ignores the old derived question vectors (it requires canonical membership), and update and forget remove a parent's derived siblings. Nothing is purged at startup; the optional cleanup below is for a disposable snapshot.
- **0.18.0 also removed** the `[langchain]`, `[llamaindex]`, `[crewai]` and `[all-integrations]` extras and their adapter modules.
- **0.9.5** fixed concurrent-writer races; stores shared by concurrent agents need at least that version.

<details>
<summary>Optional: remove legacy HyPE question vectors from a disposable snapshot</summary>

Stop old-version writers first; they can regenerate retired vectors. Take a verified backup with SQLite's online backup API (the approach in `storage/_schema_backup.py`), **not** a copy of a live database without its WAL, and never overwrite canonical writes made since the snapshot to recover optional vectors.

This recipe is for an existing, disposable, unencrypted snapshot, not a live store. Choose its embedding dimension, namespace and parent IDs explicitly; opening `SQLiteBackend` can run normal schema initialization. Encrypted stores need their key-aware backup and open procedure instead. `apply = False` only lists the selected vectors; `True` removes those derived vectors atomically. It never deletes canonical records or other namespaces, and "no vectors installed" means unavailable, not a successful cleanup.

```python
from pathlib import Path
from trw_memory.storage.sqlite_backend import SQLiteBackend

snapshot = Path("/absolute/path/to/disposable-snapshot.db")
if not snapshot.is_file():
    raise FileNotFoundError(snapshot)
namespace = "default"  # explicitly selected, locally authorized namespace
parents = ["selected-parent-id"]
apply = False
backend = SQLiteBackend(snapshot, dim=384)  # use this snapshot's dimension
try:
    if not backend.supports_vectors():
        raise RuntimeError("legacy cleanup unavailable: sqlite-vec required")
    with backend.transaction():
        for parent_id in parents:
            siblings = backend.hype_sibling_ids(parent_id, namespace=namespace)
            print(parent_id, siblings)
            if apply:
                backend.delete_hype_siblings(parent_id, namespace=namespace)
finally:
    backend.close()
```

Cleanup is idempotent, and an interruption rolls the transaction back. Rolling the package back can reopen the same canonical store; restoring the previous optional ranking also needs its matching derived-index snapshot.

</details>

## Platform and interpreter notes

### Supported interpreters

`trw-memory` is tested on CPython 3.10 through 3.14. One property of the interpreter matters beyond the version: its bundled SQLite. WAL space is only reclaimed on SQLite >= 3.51.3 (or the 3.44.6 / 3.50.7 backports). Below that, `storage/_wal_checkpoint.py` coerces resetting checkpoints to `PASSIVE`, which is safe but lets the `-wal` file grow without shrinking. Check yours with `python -c "import sqlite3; print(sqlite3.sqlite_version)"`; on macOS, Homebrew's current Python ships a qualifying build, and `trw-mcp doctor` names the qualifying interpreters it finds. The 3.1.0 fix for I/O errors during the integrity check reads `sqlite_errorcode`, so it needs Python 3.11+.

The SQLite engine is selected at import by `storage/_dbapi.py`, which ranks the interpreter's SQLite against an installed `pysqlite3` on (carries the fix, version) and never replaces a newer engine with an older wheel. The optional `[sqlite-fix]` extra pulls `pysqlite3-binary` on x86_64 Linux only. No published wheel currently bundles a qualifying SQLite, so it is an engine override, not a fix.

### Platform notes

- **sqlite-vec is a base dependency** (since 3.1.0), which is why musl Linux (Alpine) and Windows ARM are unsupported. If its extension fails to load, the backend reports `supports_vectors()` as False and recall uses keyword search.
- **`pysqlite3-binary` is not a runtime dependency** on any platform. It used to be a hard Linux dependency, which broke aarch64 Linux installs while delivering SQLite 3.51.1, below the 3.51.3 fix it existed for. The runtime probe in `storage/_dbapi.py` decides and reports which engine is active.

## Development

```bash
pip install -e ".[dev]"

# Full suite (>=85% coverage required; see fail_under in pyproject.toml)
python -m pytest tests/ -v --cov=trw_memory --cov-report=term-missing

# Type checking
python -m mypy --strict src/trw_memory/

# Targeted runs
python -m pytest tests/test_client_*.py -v
python -m pytest tests/test_retrieval_*.py -v
```

### Optional dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[embeddings]` | sentence-transformers | Dense vectors (`BAAI/bge-small-en-v1.5` by default) and the cross-encoder re-ranker |
| `[bm25]` | rank-bm25 | BM25 keyword ranking |
| `[all]` | embeddings + bm25 | The full retrieval stack (sqlite-vec is in the core install) |
| `[encryption]` | sqlcipher3, keyring, cryptography | SQLCipher encryption at rest and key storage |
| `[sqlite-fix]` | pysqlite3-binary (x86_64 Linux only) | Optional SQLite engine override; see [Supported interpreters](#supported-interpreters) |
| `[dev]` | pytest, mypy, ruff, coverage, pip-audit, vulture, deptry | Testing and linting |

There is no `[llm]` extra and no LLM-backed consolidation.

### Entry points

| Command | Purpose |
|---------|---------|
| `trw-memory` | CLI: store, recall, search, forget, consolidate, export and import, plus restore, snapshot, reembed, wiki-lint and the code index |
| `trw-memory-server` | MCP server (stdio by default; `serve http` runs the loopback daemon) |

## FAQ

### What is trw-memory?

A local-first memory engine for AI agents. It stores memories in SQLite and recalls them with keyword search plus, with the `[embeddings]` extra, local vector search and a cross-encoder re-ranker. It ships a Python SDK (`MemoryClient`), a CLI (`trw-memory`) and an MCP server (`trw-memory-server`). You do not need TRW Framework to use it.

### Does it need an LLM to store memories?

Not by default. `store_conversation()` stores each turn's text as written (whitespace trimmed, speaker prefixed) and calls no generative model; the reader does the inference at recall time. If you enable the optional decision judge, each write also sends the entry's redacted text to it for a shadow poisoning check (see [Telemetry and network behavior](#telemetry-and-network-behavior)). Consolidation uses a heuristic, not an LLM. The embedding model and re-ranker are small local models from the `[embeddings]` extra.

### How does it compare to mem0?

On all 1,540 LOCOMO questions, run through mem0's own evaluation harness with the same reader and judge, trw-memory 2.0.0 answered 88.9% correctly against 84.1% for mem0 OSS 2.0.20: +4.8 points, statistically significant, and ahead in every conversation. trw-memory stores conversations without LLM calls, where mem0 runs an extraction call for every write. Method: [Benchmarks](#more-right-answers-than-mem0-on-locomo).

### Does it work offline?

Yes. With the default configuration all data is local, and remote sync and the decision judge are off. `TRW_OFFLINE=1`, `HF_HUB_OFFLINE=1` or `local_only: true` stop model downloads; `local_only` also blocks sync, but not the judge, so leave the judge disabled. Pre-download the models for full recall offline. Without the embedding model the daemon and CLI run keyword-only and the in-process `MemoryClient` raises `LocalOnlyViolationError`; without the re-ranker, recall just skips re-ranking. See [Telemetry and network behavior](#telemetry-and-network-behavior).

### Does recall search every stored memory?

Not always on a very large namespace. Recall ranks the most recently updated rows (at least 1000 by default, more for a large `limit`) plus older active rows that full-text search matches. An older row that shares no keyword with the query can be missed, so an empty result is not evidence of absence. `recall(limit=N)` can also return fewer than N rows, because the re-ranker drops low-confidence results beyond the top `max(5, ceil(N / 2))`.

### Where is my data stored?

The Python SDK uses `.memory/` under the current directory by default (`MEMORY_STORAGE_PATH` overrides it). The loopback daemon, which the CLI and trw-mcp use, keeps every namespace in one file, by default `~/.trw/memory/memory.db`. Nothing leaves the machine unless you enable remote sync or the decision judge (see [Telemetry and network behavior](#telemetry-and-network-behavior)).

### Does memory help agents get work done?

On tasks that need a fact from an earlier session, agents with memory solved 58 of 58 and agents without it solved 0 of 50. See [Memory lets agents finish work they otherwise cannot](#memory-lets-agents-finish-work-they-otherwise-cannot).

### What license is it under?

[Business Source License 1.1](https://trwframework.com/license): source-available, free for non-competing use, converting to Apache 2.0 on 2030-03-21. The package is alpha.

## License

[Business Source License 1.1](https://trwframework.com/license): source-available, free for non-competing use. Converts to Apache 2.0 on 2030-03-21.

---

Built by [Tyler Wall](http://tylerrwall.com) · [TRW Framework](https://trwframework.com) · [Documentation](https://trwframework.com/docs) · [License](https://trwframework.com/license)
