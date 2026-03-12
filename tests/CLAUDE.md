# trw-memory Test Suite Guide

This guide is for AI agents writing or modifying tests in the `trw-memory/tests/` directory.

## Quick Reference

```bash
# During implementation — run ONLY the file you changed:
cd trw-memory
.venv/bin/python -m pytest tests/test_specific_file.py -v

# Run a single test:
.venv/bin/python -m pytest tests/test_storage.py::test_store_entry -v

# Full suite with coverage (delivery only):
.venv/bin/python -m pytest tests/ --cov=trw_memory --cov-report=term-missing

# Type check:
.venv/bin/python -m mypy --strict src/trw_memory/
```

**NEVER run the full suite during implementation. Target specific files only.**

## Test Count & Performance

- **1,326 tests** across 38 files
- **Collection**: ~12s
- **Full suite**: ~5-8 minutes
- **Coverage threshold**: 80%

## Package Architecture

trw-memory is a standalone memory engine with these modules:

| Module | Tests In | Key Patterns |
|--------|----------|-------------|
| `storage/` | `test_storage*.py` | SQLite + YAML dual-write, `tmp_path` for DB files |
| `retrieval/` | `test_retrieval*.py`, `test_search*.py` | BM25 + dense vectors, in-memory index |
| `graph.py` | `test_graph*.py` | Knowledge graph, BFS traversal |
| `sync/` | `test_sync*.py` | Remote publish/fetch, vector clocks |
| `lifecycle/` | `test_lifecycle*.py` | Tier promotion/demotion, decay |
| `security/` | `test_security*.py` | Encryption for entry fields |
| `tools/` | `test_tools*.py` | MCP tool wrappers |

## No conftest.py

This package currently has **no `tests/conftest.py`**. Each test file manages its own fixtures. When adding tests:

- Use `tmp_path` for any test that creates SQLite databases or YAML files
- Create in-memory SQLite backends: `SQLiteBackend(":memory:")`
- Don't rely on shared fixtures — each file is self-contained

## Testing Patterns

### SQLite Backend Tests
```python
def test_store_and_recall(tmp_path):
    db_path = tmp_path / "test.db"
    backend = SQLiteBackend(str(db_path))
    backend.initialize()

    entry_id = backend.store({"summary": "test", "tags": ["unit"]})
    result = backend.recall(entry_id)
    assert result["summary"] == "test"
```

### YAML Backend Tests
```python
def test_yaml_roundtrip(tmp_path):
    yaml_dir = tmp_path / "entries"
    yaml_dir.mkdir()
    backend = YAMLBackend(str(yaml_dir))

    backend.store("test-id", {"summary": "test"})
    result = backend.recall("test-id")
    assert result["summary"] == "test"
```

### Hybrid Search Tests
```python
def test_hybrid_search():
    # In-memory index for speed
    index = HybridIndex()
    index.add("id1", "Python testing best practices")
    index.add("id2", "JavaScript deployment guide")

    results = index.search("Python testing")
    assert results[0].id == "id1"
```

## Known Gotchas

### sqlite-vec Optional Dependency
Dense vector search requires `sqlite-vec` which is an optional dependency (`[vectors]`). Tests that use dense vectors should check for availability:

```python
pytest.importorskip("sqlite_vec")
```

### Pydantic v2 Serialization
- `use_enum_values=True` required on models for YAML round-trip
- `dict[str, object]` values need `str()` cast for mypy --strict
- `Field(alias=...)` requires `populate_by_name=True`

### structlog Reserved Keywords
Never use `event=` as a kwarg in structlog calls — it's reserved. Use alternative names like `action=` or `operation=`.

## Test Classification (Planned)

Currently no markers are defined. Planned classification:

- **Unit**: Tests with in-memory backends, no `tmp_path`
- **Integration**: Tests using `tmp_path`, file-based backends, dual-write
- **Slow**: Tests loading sentence-transformer models, full consolidation cycles
