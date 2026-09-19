"""Write-rate limiter vs. batch and conversation ingest (learning L-8hyp).

The limiter (``max_memory_writes_per_minute``, 10/min/session by default) meters
an agent's decisions to write. It used to be charged once per ROW, so a
``bulk_store`` batch or a ``store_conversation`` call carrying a ``session_id``
lost every row after the tenth -- 79 of 504 turns in one LongMemEval question --
visible only as per-item statuses. These tests pin the repaired semantics:

* ``store_conversation`` is transcript ingest, not agent writes: every turn lands,
  in one 600-turn call or in 600 one-turn calls at ~20 calls/second.
* ``bulk_store`` charges ONE write per distinct writer session per call.
* a genuine flood (many separate writes under one session) is still refused, and
  a refused batch is refused whole, loudly: summary counts + a structured warning.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog.testing

from trw_memory._client_bulk_store import BulkStoreRequest
from trw_memory.client import MemoryClient
from trw_memory.exceptions import RateLimitError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.runtime import enforce_write_rate_limit, single_write_operation

_LIMIT = 10


@pytest.fixture()
def limited_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_MAX_MEMORY_WRITES_PER_MINUTE", str(_LIMIT))
    client = MemoryClient(namespace="default", mode="local")
    assert client._config.max_memory_writes_per_minute == _LIMIT  # the limiter is ON in these tests
    return client


def _turns(n: int) -> list[dict[str, str]]:
    return [{"speaker": "A" if i % 2 else "B", "content": f"turn {i} about topic {i * 7919 % 1009}"} for i in range(n)]


class _Clock:
    """Deterministic clock for the limiter: ``step`` seconds per charge."""

    def __init__(self, step: float) -> None:
        self.now = 1_000_000.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


@pytest.mark.integration
async def test_one_call_600_turn_conversation_stores_every_turn(limited_client: MemoryClient) -> None:
    summary = await limited_client.store_conversation(_turns(600), session_id="conv-1")
    assert (summary.total, summary.succeeded, summary.rejected) == (600, 600, 0)
    assert summary.rejected_reasons == {}
    backend = limited_client._get_backend()
    assert all(backend.get(it.memory_id, namespace="default") is not None for it in summary.items)


@pytest.mark.integration
async def test_one_turn_per_call_at_20_calls_per_second_stores_every_turn(
    limited_client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LOCOMO shim shape: one turn per call, 20 calls/second, one session id."""
    monkeypatch.setattr("trw_memory.security.runtime.time", _Clock(step=0.05))
    turns = _turns(600)
    stored = 0
    for i, turn in enumerate(turns):
        summary = await limited_client.store_conversation(
            [turn], session_id="locomo-conv-26", preceding=[turns[i - 1]["content"]] if i else ()
        )
        stored += summary.succeeded
        assert summary.rejected == 0, summary.rejected_reasons
    assert stored == 600


@pytest.mark.integration
async def test_conversation_id_still_reaches_row_metadata_and_provenance(limited_client: MemoryClient) -> None:
    summary = await limited_client.store_conversation(_turns(2), session_id="conv-prov")
    entry = limited_client._get_backend().get(summary.items[0].memory_id, namespace="default")
    assert entry is not None
    assert entry.metadata["session_id"] == "conv-prov"
    assert entry.metadata["provenance_session_id"] == "conv-prov"


@pytest.mark.integration
async def test_bulk_store_batch_is_one_write_per_session(limited_client: MemoryClient) -> None:
    reqs = [BulkStoreRequest(content=f"batch row {i} unique {i * 31}", session_id="agent-s") for i in range(50)]
    summary = await limited_client.bulk_store(reqs)
    assert (summary.succeeded, summary.rejected) == (50, 0)


@pytest.mark.integration
async def test_bulk_store_flood_of_calls_is_refused_whole_and_loudly(
    limited_client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eleven batch calls in a minute from one session: the eleventh is refused -- all of it."""
    monkeypatch.setattr("trw_memory.security.runtime.time", _Clock(step=0.5))
    for call in range(_LIMIT):
        ok = await limited_client.bulk_store([BulkStoreRequest(content=f"call {call} row", session_id="flood")])
        assert ok.rejected == 0
    over = [BulkStoreRequest(content=f"overflow row {i}", session_id="flood") for i in range(5)]
    over.append(BulkStoreRequest(content="another session's row", session_id="other"))
    with structlog.testing.capture_logs() as logs:
        summary = await limited_client.bulk_store(over)

    assert (summary.stored, summary.rejected) == (1, 5)  # the "flood" rows all refused; "other" admitted
    assert summary.rejected_reasons == {"RateLimitError": 5}
    assert [it.status for it in summary.items] == ["rejected"] * 5 + ["stored"]
    warnings = [e for e in logs if e["event"] == "bulk_store_rows_rejected"]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert (warnings[0]["rejected"], warnings[0]["total"]) == (5, 6)
    assert warnings[0]["reasons"] == {"RateLimitError": 5}
    done = [e for e in logs if e["event"] == "memory_bulk_stored"]
    assert done and done[0]["outcome"] == "partial"


@pytest.mark.integration
async def test_single_store_flood_is_still_refused(limited_client: MemoryClient) -> None:
    for i in range(_LIMIT):
        await limited_client.store(f"agent write {i} distinct {i * 13}", session_id="agent-flood")
    with pytest.raises(RateLimitError):
        await limited_client.store("one write too many", session_id="agent-flood")


@pytest.mark.integration
async def test_clean_batch_logs_no_rejection_warning(limited_client: MemoryClient) -> None:
    with structlog.testing.capture_logs() as logs:
        summary = await limited_client.bulk_store([BulkStoreRequest(content="fine row", session_id="s")])
    assert summary.rejected_reasons == {}
    assert not [e for e in logs if e["event"] == "bulk_store_rows_rejected"]
    assert [e["outcome"] for e in logs if e["event"] == "memory_bulk_stored"] == ["success"]


@pytest.mark.integration
def test_single_write_operation_charges_each_session_once_and_shares_refusal(tmp_path: Path) -> None:
    cfg = MemoryConfig(rate_limit_state_path=str(tmp_path / "rl.yaml"), max_memory_writes_per_minute=1)

    def charge(session: str) -> None:
        enforce_write_rate_limit(cfg, session_id=session, actor="a", namespace="default", entry_id="M")

    with single_write_operation():
        for _ in range(20):
            charge("s")  # one slot, however many rows
    with single_write_operation():
        with pytest.raises(RateLimitError) as first:
            charge("s")  # budget of 1 already spent by the previous operation
        with pytest.raises(RateLimitError) as repeat:
            charge("s")  # the refusal is shared by every later row of the session
        assert repeat.value.retry_after == first.value.retry_after
    with pytest.raises(RateLimitError):
        charge("s")  # outside a scope the limiter is per call, as before
