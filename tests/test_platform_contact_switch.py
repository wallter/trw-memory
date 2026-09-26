"""rc11 F3: trw-mcp's platform contact switch stops trw-memory sync at the one place sync is derived.

With ``platform_contact_enabled: false``, ``learning_sharing_enabled: true`` and a platform URL, a publish
POST and a subscriber stream GET were still attempted: the publish, retry and subscribe paths consult only
``sync_enabled`` and ``platform_url``. The switch now turns ``sync_enabled`` off in ``MemoryConfig``, so
every one of them inherits it.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.sync._remote_publish import publish_memory_result
from trw_memory.sync.subscriber import SSESubscriber


class _RecordingClient:
    """Stands in for ``httpx.Client``: records every request and answers without a network."""

    requests: list[str] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> _RecordingClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def post(self, url: str, **_kwargs: object) -> httpx.Response:
        self.requests.append(f"POST {url}")
        return httpx.Response(201, json={"id": "remote-1"}, request=httpx.Request("POST", url))

    def stream(self, method: str, url: str, **_kwargs: object) -> object:
        self.requests.append(f"{method} {url}")
        raise httpx.ConnectError("no network in tests")


@pytest.mark.parametrize("contact", [False, True], ids=["contact-off", "contact-on"])
def test_the_platform_contact_switch_stops_publish_and_subscribe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contact: bool
) -> None:
    (tmp_path / ".trw").mkdir()
    (tmp_path / ".trw" / "config.yaml").write_text(
        f"platform_contact_enabled: {str(contact).lower()}\nlearning_sharing_enabled: true\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))  # no machine file: the project file decides
    monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://platform.example.com")
    monkeypatch.setattr(httpx, "Client", _RecordingClient)
    _RecordingClient.requests = []
    cfg = MemoryConfig()
    entry = MemoryEntry(id="L-1", content="a shareable learning", namespace="project:default", importance=0.9)

    publish_memory_result(entry, cfg)
    subscriber = SSESubscriber(cfg, on_event=lambda _data: None)
    subscriber.start()
    subscriber.stop()

    assert cfg.sync_enabled is contact
    if contact:
        assert "POST https://platform.example.com/v1/learnings" in " ".join(_RecordingClient.requests)
        assert subscriber._thread is not None
    else:
        assert _RecordingClient.requests == []
        assert subscriber._thread is None


def test_the_contact_switch_env_var_trw_mcp_reads_also_stops_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
    monkeypatch.setenv("TRW_PLATFORM_CONTACT_ENABLED", "false")

    assert MemoryConfig().sync_enabled is False


@pytest.mark.parametrize(
    ("machine", "project", "syncs"),
    [(False, None, False), (True, False, False), (False, True, True)],
    ids=["machine-off", "project-off-over-machine-on", "project-on-over-machine-off"],
)
def test_the_switch_resolves_machine_then_project_file_as_trw_mcp_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: bool, project: bool | None, syncs: bool
) -> None:
    """rc11 F3b: trw-mcp reads ``~/.trw/config.yaml`` and then the project file for this switch; trw-memory
    read only the project file, so a machine-wide ``false`` still published."""
    home, checkout = tmp_path / "home", tmp_path / "checkout"
    (home / ".trw").mkdir(parents=True)
    (checkout / ".trw").mkdir(parents=True)
    (home / ".trw" / "config.yaml").write_text(f"platform_contact_enabled: {str(machine).lower()}\n", encoding="utf-8")
    switch = "" if project is None else f"platform_contact_enabled: {str(project).lower()}\n"
    (checkout / ".trw" / "config.yaml").write_text(switch + "learning_sharing_enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TRW_PLATFORM_CONTACT_ENABLED", raising=False)
    monkeypatch.chdir(checkout)

    assert MemoryConfig().sync_enabled is syncs
