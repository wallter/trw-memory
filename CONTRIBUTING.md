# Contributing to trw-memory

Thank you for considering contributing to trw-memory. This guide covers the
development setup, conventions, and process.

## Prerequisites

- Python 3.10+
- git
- A virtual environment manager (venv, uv, etc.)

## Development Setup

```bash
git clone https://github.com/wallter/trw-memory.git
cd trw-memory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# With optional acceleration deps (dense vectors + BM25):
pip install -e ".[dev,vectors,bm25]"
```

## Running Tests

```bash
# Single test file (preferred during development)
python -m pytest tests/test_client.py -v

# Targeted suites
python -m pytest tests/test_retrieval_*.py -v
python -m pytest tests/test_storage_sqlite.py -v

# Full suite with coverage (>=85% required — see fail_under in pyproject.toml)
python -m pytest tests/ -v --cov=trw_memory --cov-report=term-missing

# Type checking (strict mode required)
mypy --strict src/trw_memory/

# Lint
ruff check src/
```

Optional dependencies degrade gracefully when absent — `sqlite-vec`
(`[vectors]`) and `sentence-transformers` (`[embeddings]`) tests skip with
`pytest.importorskip(...)` when the extra is not installed.

## Architecture

| Module | Purpose |
|--------|---------|
| `storage/` | SQLite + YAML dual backends |
| `retrieval/` | BM25 keyword + dense vector hybrid search |
| `lifecycle/` | Tier promotion/demotion, utility scoring, dedup, consolidation |
| `graph.py` | Knowledge graph with BFS traversal |
| `sync/` | Remote publish/fetch with vector clocks |
| `security/` | Field-level encryption, PII detection, RBAC |
| `client.py` | Public `MemoryClient` — primary consumer API |

## Error Handling

All `except Exception` blocks require a `# justified:` comment explaining why
the broad catch is necessary. This convention makes it easy to audit exception
handling and prevents silent swallowing.

## Logging

Use structlog throughout. The `event` keyword is reserved by structlog — use
alternative names such as `action=` or `operation=`:

```python
import structlog
logger = structlog.get_logger(__name__)

# Good
logger.info("store_complete", action="upsert", count=5)

# Bad — do NOT use event= as a keyword argument
logger.info("store_complete", event="bad")  # structlog reserves 'event'
```

## Pydantic v2 Conventions

- Use `use_enum_values=True` on models for YAML round-trip serialization
- Use `populate_by_name=True` when using `Field(alias=...)`

## Commit Format

```bash
git commit -m "feat(scope): short description" -m "WHY: rationale for the change"
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

## Pull Request Process

1. Branch from `main`
2. Write tests first (TDD), then implement
3. All tests must pass and coverage must stay at or above 85%
4. Type checking must pass: `mypy --strict src/trw_memory/`
5. Keep PRs focused — one concern per PR
6. Include a `WHY:` in your PR description

## Contribution licensing

By submitting a contribution to this project you agree to the following:

- **Developer Certificate of Origin (DCO).** All commits must be signed off
  to certify the [Developer Certificate of Origin](https://developercertificate.org/).
  Add a `Signed-off-by` line to each commit using your real name and email:

  ```bash
  git commit -s -m "fix(storage): correct WAL checkpoint gate"
  ```

  The `-s` flag appends `Signed-off-by: Your Name <you@example.com>`.

- **Inbound license.** Your contributions are provided under, and will be
  licensed as part of the project under, the project's license (Business
  Source License 1.1 — see [`LICENSE`](LICENSE)). You confirm you have the
  right to submit the work under that license.

## Questions?

Open an issue on [GitHub](https://github.com/wallter/trw-memory/issues) or
start a discussion.
