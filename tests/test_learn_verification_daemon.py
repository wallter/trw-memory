"""PRD-CORE-294 FR07: learning parity on the daemon path.

(a) ``memory_store``'s ``learning`` object and (c) ``memory_review``'s server-established
reviewer are driven through the REGISTERED tools on a real FastMCP server.

(b) The verification pass runs from the daemon.

Every test here drives the maintain passes (decay, consolidation, verification, WAL
checkpoint) against a real SQLite store via ``tests._maintain_sync.run_maintain_sync``,
then reads the row back. That helper calls the same production pass functions the
served ``memory_maintain`` tool's ``serve_maintain`` lane-job orchestration calls
(``tools.maintain._run_decay``/``_run_consolidation``/``_run_checkpoint``,
``tools._maintain_sweep.begin``/``verify_slice``/``finish``), sequenced synchronously
against an injected backend and config instead of the daemon's per-job lane and pool — see
its docstring. The pinned truths: a configured ``project_root`` persists real verdicts, no
root means nothing is verified (the verdict stays unknown, never "verified"), and the
daemon and trw-mcp's ``maintain-verify`` call the SAME trw-memory function.
"""

from __future__ import annotations

import asyncio
import getpass
import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from tests.conftest import make_entry
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.lifecycle import verification_pass
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import Assertion, MemoryEntry, MemoryStatus
from trw_memory.security.runtime import get_status_history, store_quarantined_entry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.review import register_review_tool
from trw_memory.tools.store import register_store_tool

from ._maintain_sync import run_maintain_sync

_NS = "project:default"


@pytest.fixture
def fr07b_store(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "store" / "memory.db")
    yield store
    store.close()


@pytest.fixture
def fr07b_project(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "claim.py").write_text("feature_flag = True\n")
    return root


def _fr07b_entry(entry_id: str, pattern: str, *, first_failed_at: datetime | None = None) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=f"claim.py carries {pattern}",
        namespace=_NS,
        status=MemoryStatus.ACTIVE,
        assertions=[
            Assertion(type="grep_present", pattern=pattern, target="claim.py", first_failed_at=first_failed_at)
        ],
    )


def _fr07b_config(project_root: Path | None) -> MemoryConfig:
    return MemoryConfig(project_root=str(project_root) if project_root else "")


def test_fr07b_configured_root_persists_verified_and_failed_verdicts(
    fr07b_store: SQLiteBackend, fr07b_project: Path
) -> None:
    fr07b_store.store(_fr07b_entry("L-holds", "feature_flag"))
    fr07b_store.store(_fr07b_entry("L-fails", "removed_symbol"))
    long_ago = datetime.now(timezone.utc) - timedelta(days=90)
    fr07b_store.store(_fr07b_entry("L-stale", "removed_symbol", first_failed_at=long_ago))

    result = run_maintain_sync(_NS, backend=fr07b_store, config=_fr07b_config(fr07b_project))

    verification = result["passes"]["verification"]  # type: ignore[index]
    assert verification["status"] == "ok", verification
    assert verification["entries_processed"] == 3
    assert verification["stale_transitions"] == 1
    assert verification["persist_failures"] == 0

    held = fr07b_store.get("L-holds", namespace=_NS)
    assert held is not None
    assert held.verification_status == "verified"
    assert held.verification_checked_at != ""
    assert held.assertions[0].last_result is True

    failed = fr07b_store.get("L-fails", namespace=_NS)
    assert failed is not None
    # One fresh failure is recorded on the assertion, but it is not a positive
    # verdict and not yet the persistent failure that makes an entry stale.
    assert failed.assertions[0].last_result is False
    assert failed.assertions[0].first_failed_at is not None
    assert failed.verification_status is None
    assert failed.verification_checked_at != ""

    stale = fr07b_store.get("L-stale", namespace=_NS)
    assert stale is not None
    assert stale.verification_status == "stale"


def test_fr07b_without_project_root_nothing_is_verified(fr07b_store: SQLiteBackend) -> None:
    fr07b_store.store(_fr07b_entry("L-holds", "feature_flag"))

    result = run_maintain_sync(_NS, backend=fr07b_store, config=_fr07b_config(None))

    verification = result["passes"]["verification"]  # type: ignore[index]
    assert (verification["status"], verification["reason"], verification["invalidated"]) == (
        "skipped",
        "no project_root",
        0,
    )
    assert result["status"] == "ok"
    row = fr07b_store.get("L-holds", namespace=_NS)
    assert row is not None
    assert row.verification_status is None
    assert row.verification_checked_at == ""
    assert row.assertions[0].last_result is None


def test_fr07b_a_verified_verdict_is_cleared_once_no_root_can_recheck_it(
    fr07b_store: SQLiteBackend, fr07b_project: Path
) -> None:
    fr07b_store.store(_fr07b_entry("L-holds", "feature_flag"))
    run_maintain_sync(_NS, backend=fr07b_store, config=_fr07b_config(fr07b_project))
    verified = fr07b_store.get("L-holds", namespace=_NS)
    assert verified is not None
    assert verified.verification_status == "verified"

    result = run_maintain_sync(_NS, backend=fr07b_store, config=_fr07b_config(None))

    assert result["passes"]["verification"]["invalidated"] == 1  # type: ignore[index]
    row = fr07b_store.get("L-holds", namespace=_NS)
    assert row is not None
    assert row.verification_status is None


def test_fr07b_a_failing_entry_fails_the_pass_and_holds_the_stamp(
    fr07b_store: SQLiteBackend, fr07b_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = verification_pass.run_verification_pass

    def malformed_on_bad(entry_id: str, *args: Any, **kwargs: Any) -> verification_pass.VerificationOutcome:
        if entry_id == "L-bad":
            raise ValueError("malformed stored assertion")
        return real(entry_id, *args, **kwargs)

    monkeypatch.setattr(verification_pass, "run_verification_pass", malformed_on_bad)
    fr07b_store.store(_fr07b_entry("L-bad", "feature_flag"))
    fr07b_store.store(_fr07b_entry("L-holds", "feature_flag"))

    result = run_maintain_sync(_NS, backend=fr07b_store, config=_fr07b_config(fr07b_project))

    verification = result["passes"]["verification"]  # type: ignore[index]
    assert (verification["status"], verification["reason"], verification["entry_failures"]) == (
        "error",
        "entry_failures",
        1,
    )
    assert result["status"] == "error"
    assert result["last_maintained_at"] == ""
    healthy = fr07b_store.get("L-holds", namespace=_NS)
    assert healthy is not None
    assert healthy.verification_status == "verified"


def _persist_malformed_assertions(store: SQLiteBackend, entry_id: str) -> None:
    """Corrupt the STORED assertions column, as a torn or foreign write would."""
    store._conn.execute("UPDATE memories SET assertions = ? WHERE id = ?", ('[{"type": 7}]', entry_id))
    store._conn.commit()


@pytest.mark.parametrize("with_root", [True, False])
def test_fr07b_a_persisted_malformed_row_fails_the_pass_with_or_without_a_root(
    fr07b_store: SQLiteBackend, fr07b_project: Path, with_root: bool
) -> None:
    fr07b_store.store(_fr07b_entry("L-bad", "feature_flag"))
    fr07b_store.store(_fr07b_entry("L-holds", "feature_flag"))
    _persist_malformed_assertions(fr07b_store, "L-bad")

    config = _fr07b_config(fr07b_project if with_root else None)
    result = run_maintain_sync(_NS, backend=fr07b_store, config=config)

    verification = result["passes"]["verification"]  # type: ignore[index]
    # No root is a skip only when the sweep itself was clean.
    assert (verification["status"], verification["reason"], verification["entry_failures"]) == (
        "error",
        "entry_failures",
        1,
    )
    assert result["status"] == "error"
    assert result["last_maintained_at"] == ""


def test_fr07b_missing_root_directory_is_an_error_not_a_skip(fr07b_store: SQLiteBackend, tmp_path: Path) -> None:
    fr07b_store.store(_fr07b_entry("L-holds", "feature_flag"))

    result = run_maintain_sync(_NS, backend=fr07b_store, config=_fr07b_config(tmp_path / "absent"))

    assert result["passes"]["verification"]["status"] == "error"  # type: ignore[index]
    assert result["status"] == "error"
    row = fr07b_store.get("L-holds", namespace=_NS)
    assert row is not None
    assert row.verification_status is None


def test_fr07b_a_raising_sweep_marks_the_pass_error(
    fr07b_store: SQLiteBackend, fr07b_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("cursor did not advance")

    monkeypatch.setattr(verification_pass, "run_maintain_verify", boom)

    result = run_maintain_sync(_NS, backend=fr07b_store, config=_fr07b_config(fr07b_project))

    assert result["passes"]["verification"] == {"status": "error", "reason": "ValueError"}  # type: ignore[index]
    assert result["status"] == "error"
    assert result["last_maintained_at"] == ""


async def test_the_served_memory_maintain_runs_the_same_passes_in_the_same_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_maintain_sync (used above) is a test harness, not the served path: the REAL
    memory_maintain tool runs through _maintain_sweep.serve_maintain's lane-job orchestration,
    which resolves its own backend from MEMORY_STORAGE_PATH rather than an injected one. This
    proves that path calls the SAME decay/consolidation/verification/checkpoint pass functions,
    in the SAME order (decay -> consolidation -> verification -> wal_checkpoint), as
    run_maintain_sync uses above -- so the tests above are not proving order or identity for a
    copy that has since diverged from what is actually served."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    monkeypatch.delenv("MEMORY_PROJECT_ROOT", raising=False)  # verification pass runs for real, and skips (no root)

    from trw_memory.server import mcp
    from trw_memory.tools import maintain

    order: list[str] = []
    real_decay, real_consolidation, real_verification, real_checkpoint = (
        maintain._run_decay,
        maintain._run_consolidation,
        maintain._run_verification,
        maintain._run_checkpoint,
    )

    def _decay(backend: Any, namespace: str, config: MemoryConfig | None = None) -> dict[str, object]:
        order.append("decay")
        return real_decay(backend, namespace, config)

    def _consolidation(namespace: str, backend: Any, config: MemoryConfig, lane: Any = None) -> dict[str, object]:
        order.append("consolidation")
        return real_consolidation(namespace, backend, config, lane=lane)

    def _verification(namespace: str, backend: Any, config: MemoryConfig, **kwargs: Any) -> dict[str, object]:
        order.append("verification")
        return real_verification(namespace, backend, config, **kwargs)

    def _checkpoint(backend: Any) -> dict[str, object]:
        order.append("wal_checkpoint")
        return real_checkpoint(backend)

    real_security_maintenance = maintain._run_security_maintenance

    def _security_maintenance() -> dict[str, object]:
        order.append("security_maintenance")
        return real_security_maintenance()

    monkeypatch.setattr(maintain, "_run_decay", _decay)
    monkeypatch.setattr(maintain, "_run_consolidation", _consolidation)
    monkeypatch.setattr(maintain, "_run_verification", _verification)
    monkeypatch.setattr(maintain, "_run_checkpoint", _checkpoint)
    monkeypatch.setattr(maintain, "_run_security_maintenance", _security_maintenance)

    tool = await mcp.get_tool("memory_maintain")
    result = await asyncio.wait_for(tool.run({"namespace": "project:default"}), timeout=60.0)
    reply = result.structured_content
    assert isinstance(reply, dict)

    assert order == ["decay", "consolidation", "verification", "security_maintenance", "wal_checkpoint"]
    assert reply["status"] == "ok"
    assert reply["passes"]["verification"]["status"] == "skipped"  # no MEMORY_PROJECT_ROOT: real skip path


async def test_the_served_memory_maintain_drains_a_queued_security_maintenance_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B71-97, on the REAL served path (not run_maintain_sync -- see the test above for why that
    distinction matters): a log queued while security_maintenance_inline=False is compacted by the
    next real memory_maintain call, through _maintain_sweep.serve_maintain's own lane job, not just
    the test-only sync copy."""
    from trw_memory.security import runtime as security_runtime
    from trw_memory.security.audit import AuditLog

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
    monkeypatch.delenv("MEMORY_SINGLE_STORE_PATH", raising=False)
    monkeypatch.delenv("MEMORY_PROJECT_ROOT", raising=False)
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("MEMORY_AUDIT_LOG_PATH", str(audit_path))
    monkeypatch.setenv("MEMORY_AUDIT_RETENTION_DAYS", "365")
    monkeypatch.setenv("MEMORY_SECURITY_MAINTENANCE_INLINE", "false")

    security_runtime._AUDIT_MAINTENANCE_CACHE.clear()
    security_runtime._AUDIT_MAINTENANCE_QUEUE.clear()
    log = AuditLog(audit_path)
    for index in range(2):
        log.append("store", entry_id=f"M-{index}")
    lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    stale_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    for line in lines:
        line["ts"] = stale_ts
    audit_path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    security_runtime.ensure_security_maintenance(MemoryConfig())
    assert security_runtime.security_maintenance_status()["queued"] == 1
    assert len(AuditLog(audit_path).read_all()) == 2  # queued, not yet compacted

    from trw_memory.server import mcp as real_mcp

    tool = await real_mcp.get_tool("memory_maintain")
    result = await asyncio.wait_for(tool.run({"namespace": "project:default"}), timeout=60.0)
    reply = result.structured_content
    assert isinstance(reply, dict)

    assert reply["passes"]["security_maintenance"]["status"] == "ok"
    assert security_runtime.security_maintenance_status()["queued"] == 0
    # The stale queued records (M-0, M-1) are gone; a fresh audit event from THIS maintain call
    # (decay/consolidation append while audit_enabled) is expected and fine.
    remaining_ids = {record.id for record in AuditLog(audit_path).read_all()}
    assert not remaining_ids & {"M-0", "M-1"}
    security_runtime._AUDIT_MAINTENANCE_CACHE.clear()


# --- FR07 (a) and (c): the registered store and review tools ---

NS = "project:fr07"


@pytest.fixture
def store_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryConfig:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_QUARANTINE_DB_PATH", str(tmp_path / "quarantine.db"))
    config = MemoryConfig()
    assert config.quarantine_db_path == str(tmp_path / "quarantine.db")
    return config


def _server() -> FastMCP:
    mcp = FastMCP("fr07")
    register_store_tool(mcp)
    register_review_tool(mcp)
    return mcp


async def test_memory_store_learning_object_reaches_the_stored_row(store_env: MemoryConfig) -> None:
    learning = {
        "type": "workaround",
        "confidence": "medium",
        "task_type": "debug",
        "domain": ["sqlite"],
        "phase_origin": "implement",
        "phase_affinity": ["validate"],
        "team_origin": "lane-p",
        "protection_tier": "high",
        "anchors": [{"file": "src/db.py", "symbol_name": "connect", "symbol_type": "function"}],
        "nudge_line": "pin the driver first",
    }
    async with Client(_server()) as client:
        result = await client.call_tool(
            "memory_store", {"content": "pin the sqlite driver", "namespace": NS, "learning": learning}
        )

    with create_backend_from_config(store_env, NS) as backend:
        entry = backend.get(str(result.data["memory_id"]), namespace=NS)
    assert entry is not None
    stored = entry.model_dump(mode="json")
    for key in ("type", "confidence", "task_type", "domain", "phase_origin", "phase_affinity", "team_origin"):
        assert stored[key] == learning[key], key
    assert stored["protection_tier"] == "high"
    assert stored["nudge_line"] == "pin the driver first"
    assert [(a["file"], a["symbol_name"]) for a in stored["anchors"]] == [("src/db.py", "connect")]


async def test_memory_store_refuses_an_unknown_learning_key(store_env: MemoryConfig) -> None:
    async with Client(_server()) as client:
        with pytest.raises(ToolError):
            await client.call_tool("memory_store", {"content": "x", "namespace": NS, "learning": {"q_value": 0.9}})

    with create_backend_from_config(store_env, NS) as backend:
        assert backend.list_entries(namespace=NS) == []


def _quarantined(config: MemoryConfig, entry_id: str) -> None:
    store_quarantined_entry(config, make_entry(entry_id=entry_id, namespace=NS, content=f"suspicious {entry_id}"))
    with create_backend_from_config(config, NS) as backend:
        backend.store(make_entry(entry_id=entry_id, namespace=NS, content=f"suspicious {entry_id}"))


def _reviewer(config: MemoryConfig, entry_id: str) -> str:
    return get_status_history(config, entry_id, namespace=NS)[-1]["reviewer_id"]


async def test_memory_review_has_no_reviewer_parameter(store_env: MemoryConfig) -> None:
    async with Client(_server()) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert "reviewer_id" not in tools["memory_review"].inputSchema["properties"]
        _quarantined(store_env, "M-forged")
        with pytest.raises(ToolError):
            await client.call_tool(
                "memory_review",
                {"learning_id": "M-forged", "decision": "approve", "namespace": NS, "reviewer_id": "someone-else"},
            )


async def test_memory_review_in_process_records_the_os_user(store_env: MemoryConfig) -> None:
    _quarantined(store_env, "M-local")
    async with Client(_server()) as client:
        await client.call_tool("memory_review", {"learning_id": "M-local", "decision": "approve", "namespace": NS})

    assert _reviewer(store_env, "M-local") == f"os:{getpass.getuser()}"


async def test_memory_review_over_http_records_the_bearer_principal(
    store_env: MemoryConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The daemon's real streamable-HTTP app: bearer verified by the transport, principal recorded."""
    import httpx
    from fastmcp.client.transports import StreamableHttpTransport

    from trw_memory import server
    from trw_memory.daemon import DaemonPaths, mint_grant
    from trw_memory.daemon._serve import _build_app

    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    paths = DaemonPaths.resolve()
    secret = mint_grant(paths, [NS])  # PRD-CORE-298 FR02: a bearer is a namespace grant
    monkeypatch.setattr(server.mcp, "auth", server.mcp.auth)  # _build_app mutates the singleton
    monkeypatch.setattr(server.mcp, "middleware", list(server.mcp.middleware))
    app = _build_app(paths)
    _quarantined(store_env, "M-http")

    def _asgi_client(**kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), **kwargs)

    def _client(token: str) -> Client[Any]:
        from trw_memory.daemon._version_gate import VERSION_HEADER
        from trw_memory.daemon.client import _package_version

        return Client(
            StreamableHttpTransport(
                "http://daemon/mcp",
                auth=token,
                headers={VERSION_HEADER: _package_version()},
                httpx_client_factory=_asgi_client,
            )
        )

    async with app.router.lifespan_context(app):
        with pytest.raises(httpx.HTTPStatusError):
            async with _client("wrong-token") as client:
                await client.list_tools()
        async with _client(secret) as client:
            await client.call_tool("memory_review", {"learning_id": "M-http", "decision": "reject", "namespace": NS})

    assert _reviewer(store_env, "M-http") == "token:trw-memory-loopback"
