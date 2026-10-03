"""A recall query is user content: it reaches the platform only when the project turned team sync on.

Drives the real ``fetch_shared_memories`` with a stub httpx transport (the network boundary) and a project
directory whose ``.trw/config.yaml`` the fetch must read at call time.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.sync import fetch_shared_memories

pytestmark = pytest.mark.integration

_QUERY = "why does 10.0.0.5 refuse connections for ops@example.com, call 415-555-0134"


def _project(tmp_path: Path, config: str) -> MemoryConfig:
    (tmp_path / ".trw").mkdir()
    (tmp_path / ".trw" / "config.yaml").write_text(config, encoding="utf-8")
    return MemoryConfig(
        storage_backend="sqlite",
        storage_path=str(tmp_path),
        project_root=str(tmp_path),
        sync_enabled=True,
        platform_url="https://api.test.invalid",
    )


def _fetch(cfg: MemoryConfig, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Run one fetch and return every request body that went out, serialised."""
    monkeypatch.delenv("TRW_TEAM_SYNC_ENABLED", raising=False)
    monkeypatch.delenv("TRW_PLATFORM_CONTACT_ENABLED", raising=False)
    sent: list[str] = []
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    response = MagicMock(status_code=200)
    response.json.return_value = []
    client.post.return_value = response
    client.post.side_effect = lambda url, **kw: (sent.append(url + json.dumps(kw.get("json"))), response)[1]
    with patch("trw_memory.sync._remote_fetch.httpx.Client", return_value=client):
        fetch_shared_memories(_QUERY, cfg, admit=MagicMock())
    return sent


def test_query_stays_local_when_team_sync_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _fetch(_project(tmp_path, "platform_contact_enabled: true\n"), monkeypatch)
    assert sent == []


def test_query_stays_local_when_team_sync_is_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _fetch(_project(tmp_path, "team_sync_enabled: [not, a, bool]\n"), monkeypatch)
    assert sent == []


def test_query_leaves_redacted_when_team_sync_is_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _fetch(_project(tmp_path, "team_sync_enabled: true\n"), monkeypatch)
    assert len(sent) == 1
    body = sent[0]
    for raw in ("10.0.0.5", "ops@example.com", "415-555-0134"):
        assert raw not in body
    assert "refuse connections" in body  # the search intent survives the redaction


def test_team_sync_withdrawn_between_fetches_stops_the_next_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _project(tmp_path, "team_sync_enabled: true\n")
    assert len(_fetch(cfg, monkeypatch)) == 1
    (tmp_path / ".trw" / "config.yaml").write_text("team_sync_enabled: false\n", encoding="utf-8")
    assert _fetch(cfg, monkeypatch) == []
