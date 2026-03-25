# Changelog

All notable changes to the TRW Memory package.

## [0.3.0] — 2026-03-14

### Added

- **MemoryClient SDK** — high-level async Python client with store/recall/forget/search
- **Hybrid retrieval pipeline** — BM25 keyword + dense vector similarity via sqlite-vec, combined with Reciprocal Rank Fusion (RRF)
- **Knowledge graph** — tag co-occurrence and similarity edges, BFS traversal, importance boost/decay
- **LLM consolidation** — episodic-to-semantic clustering and summarization
- **Remote sync** — publish/fetch learnings with vector clock conflict resolution and SSE live updates
- **Security** — AES-256-GCM field encryption, PII detection/redaction, memory poisoning detection, RBAC, audit trail
- **Framework integrations** — LangChain memory, LlamaIndex reader/writer, CrewAI component, OpenAI-compatible adapter
- **CLI** — full command-line interface for store, recall, search, forget, consolidate, export/import
- **REST API** — FastAPI server with CRUD, search, namespace management, and background jobs
- **MCP tools** — 6 tools for store, recall, search, consolidate, forget, and status

### Changed

- **SQLite as primary backend** — migrated from YAML-first to SQLite-first with FTS5 and WAL mode
- **Scoring engine** — Q-learning with EMA updates, Ebbinghaus forgetting curve, Bayesian MACLA calibration
- **Tiered storage** — hot/warm/cold tiers with automatic promotion/demotion

## [0.2.0] — 2026-02-22

### Added

- Initial SQLite backend with FTS5 full-text search
- BM25 sparse retrieval
- Semantic dedup with cosine similarity
- YAML backend for backward compatibility

## [0.1.0] — 2026-02-07

### Added

- Initial release — YAML-based storage backend
- Basic keyword search
- Entry lifecycle (store, retrieve, forget)
