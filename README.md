# trw-memory

**Local-first memory for AI agents: a SQLite file on your machine, keyword and vector recall.**

trw-memory gives your AI agents long-term memory that runs locally. Memories live in a SQLite file on your machine, recall combines keyword ranking, local vector search and a re-ranker, and storing a conversation calls no LLM by default. On all 1,540 LOCOMO questions, trw-memory 2.0.0 answered 88.9% correctly against 84.1% for mem0 OSS 2.0.20, a statistically significant lead ([benchmarks](#benchmarks)). Use it from Python, the command line, or any MCP client. It is the memory engine behind [TRW Framework](https://trwframework.com)'s MCP server, trw-mcp, and works on its own.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](https://trwframework.com/license)
[![Docs](https://img.shields.io/badge/docs-trwframework.com-blue)](https://trwframework.com/docs)

> **Status:** alpha, source-available under BSL 1.1. The public API may change in future releases; test an upgrade before you roll it out.

**[What's new](#whats-new-in-4x)** · **[Quick start](#install-and-quick-start)** · **[Benchmarks](#benchmarks)** · **[How it works](#how-it-works)** · **[MCP server](#mcp-memory-server)** · **[Network and security](#telemetry-and-network-behavior)** · **[Upgrading](#upgrading)** · **[FAQ](#faq)**

## Why trw-memory

- **More right answers.** On the LOCOMO benchmark, trw-memory 2.0.0 answered 88.9% of 1,540 questions correctly against 84.1% for mem0 OSS, ahead in all ten conversations. Its search finds the evidence too: on LongMemEval, trw-memory 2.0.0 returned it in the top 10 search results for 93.8% of 470 questions.
- **No LLM calls to store.** `store_conversation()` keeps each chat turn as written, with its date and the turn it replied to, so saving memory costs no API calls (by default). The reader does the inference at recall time.
- **Runs on your machine.** Memories live in a local SQLite file. There is no hosted service, no account and no usage tracking. Models download once, when you fetch them, and runtime never touches the network; remote sync and the decision judge are opt-in and off by default.
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

Without the `[embeddings]` extra (included in `[all]`) nothing produces vectors, so recall is keyword-only. Fetch the embedding model (`BAAI/bge-small-en-v1.5`, about 130 MB) and the re-ranker once with `trw-mcp models fetch` or `trw_memory.embeddings.fetch_models()`; runtime loads never download (see [Telemetry and network behavior](#telemetry-and-network-behavior)).

**Supported platforms:** macOS arm64/x86_64 and Linux with glibc (x86_64, aarch64). CPython 3.11 through 3.14. 3.10 also works, but `memory_import_checkout` (the checkout-store import behind `trw-mcp memory migrate`) needs 3.11 or later. On Windows, use WSL2, which works as Linux. Native Windows is not supported in 4.0, because the store's file-safety checks need POSIX.

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

## What's new in 4.x

<!-- whats-new: 4.0.0 -->

- **No corrupted or lost writes.** A store shared by two processes keeps its SQLite locks, and a new store's first writes are kept. Both affected 2.x and 3.x.
- **Recall works on a default install.** rank-bm25 is a base dependency, so the entity-bridge hop no longer crashes and default installs get the BM25 lane.
- **A crashed daemon no longer strands clients.** A zombie daemon reads as dead, its starter reaps it, and models run on CPU on macOS, avoiding a Metal crash.
- **Text in, never vectors.** The daemon embeds and dedups itself, and no vector leaves your machine.
- **One download path.** `fetch_models()` gets the models, the embedding model at a pinned revision, and runtime loads are cache-only. `local_only` is retired.
- **Re-embed in place, and clear version errors.** `memory_reembed` re-encodes vectors outside the active space; a 4.x daemon refuses a 3.x client by name.
- **Faster daemon calls.** The daemon serves stateless JSON, and `DaemonClient(keep_session=True)` makes each call one HTTP request instead of six.

4.0.0 is a breaking release: read the [CHANGELOG](https://github.com/wallter/trw-memory/blob/main/CHANGELOG.md) before upgrading, and upgrade trw-mcp to 7.0.0 with it.

## Benchmarks

Each result names the trw-memory version it was measured on. Harnesses: [`benchmarks/locomo/`](https://github.com/wallter/trw-memory/blob/main/benchmarks/locomo/README.md), [`benchmarks/longmemeval/`](https://github.com/wallter/trw-memory/blob/main/benchmarks/longmemeval/README.md), [`benchmarks/engmem/`](https://github.com/wallter/trw-memory/blob/main/benchmarks/engmem/README.md).

### More right answers than mem0 on LOCOMO

| All 1,540 LOCOMO questions, judged correct | mem0 OSS 2.0.20 | trw-memory 2.0.0 |
|---|:---:|:---:|
| Answer accuracy | 84.1% | **88.9%** |

trw-memory answered 4.8 points more questions correctly (paired 95% CI +2.9 to +6.8; McNemar p < 0.0001) and led in all ten conversations. It made no LLM calls to store the 5,882 conversation turns; mem0 makes an extraction call for every write.

<sub>Method: mem0's evaluation harness (commit `4b61c5d`) on all ten [LOCOMO](https://github.com/snap-research/locomo) conversations; each system answered from its top 10 memories with the same reader and judge (`gpt-4o-mini`) and the same prompts. trw-memory 2.0.0 with `bge-small-en-v1.5` vs self-hosted `mem0ai` 2.0.20 with `all-MiniLM-L6-v2`; the embedders differed. September 2026.</sub>

### Finds the evidence on LOCOMO and LongMemEval

With no LLM in the loop, how often the turn that holds the answer comes back in the top results, on default settings:

| Benchmark | Questions | Measured on | In top 10 | In top 50 |
|---|:---:|:---:|:---:|:---:|
| LOCOMO, all ten conversations | 1,540 | 4.0.0 RC | **85.8%** (95% CI 83.9–87.4) | **92.7%** (91.3–93.9) |
| LongMemEval_S (cleaned) | 470 | 2.0.0 | **93.8%** (95% CI 91.3–95.7) | **97.4%** |

On the 4.0.0 release candidate, a 10% LongMemEval sample (47 questions) put the evidence in the top 10 for 95.7% of them (95% CI 85.8–98.8). The full 470-question run on the release candidate has not been completed; this sample's interval overlaps the full 2.0.0 result.

These are retrieval hit rates, a different measure from the judged answer accuracy above. Every retrieval change has to hold up on both benchmarks, question by question, before it ships.

<sub>4.0.0 RC: the trw-memory 4.0.0 release candidate (int `657629ecd`), September 2026. Intervals are Wilson 95% over questions. LOCOMO conversations were stored the way the product stores them (`store_conversation`) and searched with `MemoryClient.recall`.</sub>

### Keeps finding the right record as the store grows

On EngMem, a synthetic benchmark of engineering learnings where newer records supersede older ones, the MCP `memory_recall` tool found every required record for 16 of 16 queries at 1,000 rows, 14 of 16 at 5,000 and 20,000 rows, and 12 of 16 at 100,000 rows. It never returned a retired record in its top 10 at any size. Plain BM25 keyword search found every required record for none of the 16 queries at 5,000 rows and above, and grep returned a retired record for 8 of 16 at every size. Median `memory_recall` time was 272 ms at 20,000 rows and 467 ms at 100,000.

<sub>16 queries per size, so each rate is coarse: the Wilson 95% intervals are 80.6–100% at 16 of 16, 64.0–96.5% at 14, 50.5–89.8% at 12, 28.0–72.0% at 8 and 0–19.4% at 0. The framework's retrieval core and the library call `MemoryClient.recall` found every required record, and returned no retired one, for the same number of queries as the tool at every size; the latencies above are the tool's. At 100,000 rows a paired McNemar test against the retrieval core gives p = 0.0005 for BM25 and p = 0.002 for grep. Seed 7, one run per size on one Apple-silicon machine, trw-memory 4.0.0 release candidate (int `657629ecd`), September 2026.</sub>

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
| Encryption at rest | Not supported: trw-memory does not encrypt its store. It rejects `encryption_enabled=true` at config load, backend creation and `trw-memory-server` startup with `EncryptionAtRestUnsupportedError`. Protect the store with disk encryption (FileVault, LUKS, BitLocker) and the owner-only file permissions. |
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
| `graph.py`, `embeddings/`, `sync/`, `security/` | Knowledge graph, embedding providers, remote sync, security |

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
| `memory_store` | Store an entry, and a vector when embeddings are available (without a cached model, the row is stored without one). A refused write comes back as a status (`invalid`, `blocked` or `rate_limited`), not an MCP error |
| `memory_recall` | Hybrid retrieval, with optional graph traversal; the same ranking as trw-mcp's `trw_recall` |
| `memory_search` | List and filter by status and tags, with pagination |
| `memory_get`, `memory_update`, `memory_forget` | Read, change or delete entries |
| `memory_consolidate` | Episodic-to-semantic consolidation |
| `memory_maintain` | Run decay, consolidation and WAL checkpointing for a namespace |
| `memory_reembed` | Re-encode a namespace's vectors that are outside the active embedding space; safe to rerun. `memory_status` `coverage.outside_active_space` counts what is left |
| `memory_status` | Store statistics and health; `security_settings_only=True` reports the daemon-wide security settings |
| `memory_audit`, `memory_review`, `memory_quarantine_list` | Provenance audit and quarantine review |

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

trw-memory is **local-first**: all data lives in a local SQLite store (and an optional YAML sidecar), and at runtime it makes **no outbound network calls** by default. Models download once, when you fetch them; after that, recall and store never contact Hugging Face. Two opt-in features reach the network: remote sync and the decision judge. There is no usage tracking or content phone-home.

### What can touch the network, and when

| Surface | When | Default | Control |
|---------|------|---------|---------|
| **Model fetch** | Only when you run it: `trw-mcp models fetch`, the TRW installer, or `trw_memory.embeddings.fetch_models()`. It downloads the embedding model (`BAAI/bge-small-en-v1.5` by default, pinned to one Hub commit: 33M parameters, 384 dimensions, about 130 MB; `MEMORY_EMBEDDING_MODEL` changes it) and the re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) into your Hugging Face cache | runs only when invoked | don't run it; copy a populated Hugging Face cache instead |
| **Model loads at runtime** | Never. The embedder and re-ranker load from the local Hugging Face cache only (`local_files_only=True`, and a complete snapshot is opened from its directory), so a warm cache makes **zero** huggingface.co requests | cache-only, always | nothing to turn off |
| **Remote sync / publish** | Only when `sync_enabled=true` and a platform URL is set. Text only: no vector ever leaves the machine | **off** | leave sync disabled |
| **Decision judge** (`trw_memory.decisions`) | Only when enabled **and** an `OPENROUTER_API_KEY` is present. For the store path, enabled means `TRW_JEV_ENABLED` in the process environment or `assess_enabled` in the user's `~/.trw/config.yaml`; that path does not read project `.trw/config.yaml` or `.env`, and it takes the key from the process environment only. `python -m trw_memory.decisions.cli` also reads project settings via `--dotenv`. When enabled, the store path's poisoning screen sends each written entry's text, after credential and PII redaction, to the judge endpoint (default `https://openrouter.ai`, an allowlisted https host) as a shadow check that never changes the outcome; `python -m trw_memory.decisions.cli` calls it directly | **off** | leave it disabled: unset `TRW_JEV_ENABLED` / `assess_enabled` at every layer, or set `TRW_JEV_ENABLED=false` in the process environment, which overrides the others |

`learning_sharing_enabled` governs learning-content publishing. Model downloads are not a runtime behaviour at all, so no consent flag gates them.

**Without a cached embedding model**, behaviour depends on the entry point. The daemon's `memory_recall`, `memory_store` and `memory_consolidate` (and so the CLI and trw-mcp) run keyword-only, store rows without vectors, and report `"dense": "unavailable: <reason>"` with `trw-mcp models fetch` as the fix. The in-process `MemoryClient` does the same for store and recall (keyword-only, logged once); only `MemoryClient.reembed()`, which cannot degrade, raises `ModelNotCachedError` naming the same command. Without the re-ranker, recall keeps fusion order. `RemoteCodeNotPermittedError` raises in both.

### Environment-variable inventory

| Variable | Purpose | Default |
|----------|---------|---------|
| `MEMORY_EMBEDDING_MODEL` | Sentence-transformers model for dense vectors. Changing it leaves stored vectors in the old model's space until they are re-encoded (see [Changing the embedding model](#changing-the-embedding-model)) | `BAAI/bge-small-en-v1.5` |
| `MEMORY_STORAGE_PATH` | Where the Python SDK keeps its stores | `.memory/` |
| `MEMORY_*` | Engine settings validated by `MemoryConfig` (for example `MEMORY_SYNC_ENABLED`, `MEMORY_EMBEDDING_TRUST_REMOTE_CODE`, retrieval and lifecycle tuning) | per field |

`TRW_OFFLINE`, `HF_HUB_OFFLINE` and `local_only` are gone in 4.0.0: runtime is always offline for models, and sync is opt-in. A leftover `local_only`, in any source (`.trw/config.yaml`, `MEMORY_LOCAL_ONLY`, dotenv) and at any value, stops startup with a `ConfigError` naming the key, so remove it.

### Security defaults

| Capability | Default | Notes |
|-----------|---------|-------|
| Encryption at rest | **not supported** | `encryption_enabled=true` is rejected at config load, backend creation and `trw-memory-server` startup with `EncryptionAtRestUnsupportedError`. Use full-disk encryption (FileVault, LUKS, BitLocker) |
| PII detection | **on** (`pii_enabled=True`) | scans `content`, `detail`, `tags`, `evidence[]` and `Assertion.last_evidence` on the store path. Recognised credentials (API keys and provider access tokens, detected as `PIIType.API_KEY`, the only type in `BLOCKING_PII_TYPES`) **block the write** (`PIIBlockError`); every other type is recorded in `pii_types` metadata and stored **verbatim**. `pii_action` (default `warn`) is only reported by `memory_status`; it does not change what the store path does. Emails, IPs, SSNs, phone numbers and card numbers are masked at the publish boundary. Set `pii_custom_patterns` to opt in to local masking with your own regexes |
| Poisoning / size-anomaly detection | **observe** (`poisoning_detection_mode="observe"`) | records anomaly stats and telemetry but does **not** quarantine; `enforce` is opt-in. A caller-supplied `metadata['source']` cannot skip enforce-mode quarantine |
| Trust scoring | **observe** (`trust_scoring_mode="observe"`) | logs intake trust decisions; `enforce` and `strict` are opt-in |
| Provenance signing | **required** (`provenance_required=True`) | persisted rows carry a signed provenance hash chain |
| Canary tamper response | **halt** (`canary_fail_mode="halt"`) | seeded canaries are probed on recall; tampering halts by default (`degrade` and `log-only` are opt-in) |
| Remote sync / publishing | **off** (`sync_enabled=False`) | text only; vectors never leave the machine |
| Decision judge (LLM calls to OpenRouter) | **off** (`TRW_JEV_ENABLED` unset) | needs both an enable switch and `OPENROUTER_API_KEY` |
| Model downloads at runtime | **never** | models arrive only through an explicit fetch; loads are `local_files_only` |
| Model remote-code execution | **off** (`embedding_trust_remote_code=False`) | the only input to sentence-transformers' `trust_remote_code`. A model repository that ships its own Python modules is refused with `RemoteCodeNotPermittedError`; set it `true` only for a repository you trust. The default model needs no remote code |
| `memory.db` permissions | `0600` | the store file is `chmod 0600` on creation; a non-POSIX platform logs a `db_chmod_failed` warning |

### Enterprise hardening recipe

Fetch the models once, in the environment that will run trw-memory (or copy a populated Hugging Face cache into it):

```bash
trw-mcp models fetch
# standalone trw-memory:
python -c "from trw_memory.embeddings import fetch_models; fetch_models()"
```

Then keep the two opt-in network features off, which is the default:

```bash
export TRW_JEV_ENABLED=false   # the decision judge
# and leave sync_enabled / learning_sharing_enabled unset
```

From then on nothing leaves the machine: model loads are cache-only, and there is no runtime download to block. To run keyword-only on purpose, omit the `[embeddings]` extra. Verify that the on-disk `memory.db` is mode `0600`, that `TRW_JEV_ENABLED` / `assess_enabled` are not set in any layer, and that no outbound connection is attempted.

## Upgrading

### Upgrading from 3.x

If you use trw-memory through trw-mcp, upgrade both together (trw-mcp 7.0.0 needs trw-memory 4.x) and follow the [trw-mcp README](https://github.com/wallter/trw-mcp#upgrading). For a standalone install:

1. **Remove `local_only` first.** 4.0.0 refuses to start while it is set, in any source and at any value (a `ConfigError` naming the key). It used to force `sync_enabled: false` and `rbac_mode: local`, so where sharing must stay off, set `sync_enabled: false` explicitly and check `rbac_mode`.
2. **Install:** `pip install -U "trw-memory[embeddings]==4.0.0"`. rank-bm25 is now a base dependency: drop `bm25` from any extras list (`[all]` is now `[embeddings]`).
3. **Restart the daemon**, if one runs. A 4.x daemon refuses a 3.x client with `daemon_version_mismatch`, and a 4.x client refuses a 3.x daemon. The 4.0.0 daemon also refuses to start when its store directory, or any ancestor, is owned by a user other than you or root, or is group- or world-writable without the sticky bit.
4. **Fetch the models:** `trw_memory.embeddings.fetch_models()` (or `trw-mcp models fetch`). Runtime loads are cache-only: without the model the SDK and the daemon fall back to keyword search, and only `MemoryClient.reembed()` raises `ModelNotCachedError` (renamed from `LocalOnlyViolationError`).
5. **Check your stores.** Releases before 4.0.0 could corrupt a store shared by two processes, and 2.0.0, 3.0.0 and 3.1.0 could lose a new store's first writes. Run `sqlite3 <store> 'pragma integrity_check'` on the user store (`~/.trw/memory/memory.db`, or under `$TRW_USER_DIR` / `$XDG_DATA_HOME/trw` when set) and on any project store (`<project>/.trw/memory/memory.db`); output other than `ok` reports a problem, and `trw-memory restore --from-snapshot latest --db <path>` or `--from-cold` rebuilds it (stop every process using the store first). Look for `memory.db.corrupt.*.bak` files beside a store too: one from the store's first use may hold lost writes, so open it read-only before deleting it.
6. **Update code that calls the daemon or the Python API.** `memory_similar(namespace, text, skip_threshold, merge_threshold, top_k=10)` takes text, not a vector; `memory_vectors` drops `space`; `memory_maintain` takes the caller's consolidation policy. `trw_memory.bandit`, `trw_memory.code_index`, `trw_memory.migration` and `trw_memory.wiki` are deleted, with the `code-index`, `code-search`, `code-symbol` and `wiki-lint` subcommands. The [CHANGELOG](https://github.com/wallter/trw-memory/blob/main/CHANGELOG.md) lists every change.

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

`trw-memory reembed` works on a store the SDK opens in local mode (under your `storage_path`), and the CLI refuses to run while a daemon runs. For the daemon's store at `~/.trw/memory/memory.db`, use the daemon's `memory_reembed` tool (`trw-mcp memory reembed`). Both are idempotent and resumable, and neither downloads a model, so fetch the new one first. To keep the previous model, set `MEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2`; a process only ever scores vectors from the space its embedder reports, so switching back and forth never mixes spaces.

### Older releases

- **0.18.0 retired HyPE question generation and HyDE query expansion.** Delete imports of `QuestionGenerator` and `NoOpQuestionGenerator`; remove the `question_generator`, `query_expansion` and `collapse_hype` arguments and the `hype_*` settings. Neutral legacy values still only warn in 3.x; a later breaking release will remove them. Recall ignores the old derived question vectors (it requires canonical membership), and update and forget remove a parent's derived siblings. Nothing is purged at startup; the optional cleanup below is for a disposable snapshot.
- **0.18.0 also removed** the `[langchain]`, `[llamaindex]`, `[crewai]` and `[all-integrations]` extras and their adapter modules.
- **0.9.5** fixed concurrent-writer races; stores shared by concurrent agents need at least that version.

<details>
<summary>Optional: remove legacy HyPE question vectors from a disposable snapshot</summary>

Stop old-version writers first; they can regenerate retired vectors. Take a verified backup with SQLite's online backup API (the approach in `storage/_schema_backup.py`), **not** a copy of a live database without its WAL, and never overwrite canonical writes made since the snapshot to recover optional vectors.

This recipe is for an existing, disposable, unencrypted snapshot, not a live store. Choose its embedding dimension, namespace and parent IDs explicitly; opening `SQLiteBackend` can run normal schema initialization. `apply = False` only lists the selected vectors; `True` removes those derived vectors atomically. It never deletes canonical records or other namespaces, and "no vectors installed" means unavailable, not a successful cleanup.

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

- **sqlite-vec is a base dependency** (since 3.1.0), which is why musl Linux (Alpine) is unsupported. If its extension fails to load, the backend reports `supports_vectors()` as False and recall uses keyword search.
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
| `[all]` | embeddings | The full retrieval stack (sqlite-vec and rank-bm25 are in the core install) |
| `[sqlite-fix]` | pysqlite3-binary (x86_64 Linux only) | Optional SQLite engine override; see [Supported interpreters](#supported-interpreters) |
| `[dev]` | pytest, mypy, ruff, coverage, pip-audit, vulture, deptry | Testing and linting |

There is no `[llm]` extra and no LLM-backed consolidation.

### Entry points

| Command | Purpose |
|---------|---------|
| `trw-memory` | CLI: store, recall, search, forget, consolidate, export and import, plus restore, snapshot and reembed |
| `trw-memory-server` | MCP server (stdio by default; `serve http` runs the loopback daemon) |

## FAQ

### What is trw-memory?

A local-first memory engine for AI agents. It stores memories in SQLite and recalls them with keyword search plus, with the `[embeddings]` extra, local vector search and a cross-encoder re-ranker. It ships a Python SDK (`MemoryClient`), a CLI (`trw-memory`) and an MCP server (`trw-memory-server`). You do not need TRW Framework to use it.

### Does it need an LLM to store memories?

Not by default. `store_conversation()` stores each turn's text as written (whitespace trimmed, speaker prefixed) and calls no generative model; the reader does the inference at recall time. If you enable the optional decision judge, each write also sends the entry's redacted text to it for a shadow poisoning check (see [Telemetry and network behavior](#telemetry-and-network-behavior)). Consolidation uses a heuristic, not an LLM. The embedding model and re-ranker are small local models from the `[embeddings]` extra.

### How does it compare to mem0?

On all 1,540 LOCOMO questions, run through mem0's own evaluation harness with the same reader and judge, trw-memory 2.0.0 answered 88.9% correctly against 84.1% for mem0 OSS 2.0.20: +4.8 points, statistically significant, and ahead in every conversation. trw-memory stores conversations without LLM calls, where mem0 runs an extraction call for every write. Method: [Benchmarks](#more-right-answers-than-mem0-on-locomo).

### Does it work offline?

Yes, always. All data is local, remote sync and the decision judge are off by default, and runtime never downloads a model. Fetch the models once for full recall. Without the embedding model the daemon, the CLI and the in-process `MemoryClient` run keyword-only (only `MemoryClient.reembed()` raises `ModelNotCachedError`); without the re-ranker, recall just skips re-ranking. See [Telemetry and network behavior](#telemetry-and-network-behavior).

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
