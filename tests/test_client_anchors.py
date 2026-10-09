"""``MemoryClient.store(anchors=..., source="distill")``: the public library door for code anchors.

The writer-side rule mirrors trw_learn's: anchors are repo-relative, a machine path is
refused before anything is written.
"""

from __future__ import annotations

import re

import pytest

from trw_memory.client import MemoryClient
from trw_memory.exceptions import SchemaValidationError
from trw_memory.models.memory import Anchor, MemoryEntry
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools.store import memory_store_impl

_GOOD = {"file": "src/pkg/mod.py", "symbol_name": "handler", "symbol_type": "function", "line_range": (3, 9)}


def _stored(client: MemoryClient, memory_id: str) -> MemoryEntry:
    entry = client._get_backend().get(memory_id, namespace=client.namespace)
    assert entry is not None
    return entry


async def test_anchors_round_trip_unchanged(memory_client: MemoryClient) -> None:
    result = await memory_client.store("anchored lesson", anchors=[Anchor(**_GOOD)])  # type: ignore[arg-type]
    entry = _stored(memory_client, result["memory_id"])
    assert [a.model_dump() for a in entry.anchors] == [Anchor(**_GOOD).model_dump()]  # type: ignore[arg-type]


async def test_dict_form_from_trw_learn_is_accepted(memory_client: MemoryClient) -> None:
    result = await memory_client.store("anchored lesson", anchors=[dict(_GOOD)])
    assert _stored(memory_client, result["memory_id"]).anchors[0].file == "src/pkg/mod.py"


async def test_omitting_anchors_stores_an_empty_list(memory_client: MemoryClient) -> None:
    result = await memory_client.store("plain lesson")
    assert _stored(memory_client, result["memory_id"]).anchors == []


async def test_update_without_anchors_keeps_them(memory_client: MemoryClient) -> None:
    first = await memory_client.store("anchored lesson", anchors=[dict(_GOOD)], entry_id="M-fixed")
    await memory_client.store("anchored lesson, reworded", entry_id=first["memory_id"])
    assert [a.file for a in _stored(memory_client, "M-fixed").anchors] == ["src/pkg/mod.py"]


@pytest.mark.parametrize(
    "bad",
    [
        {**_GOOD, "file": "/etc/passwd"},
        {**_GOOD, "file": "C:/work/mod.py"},
        {**_GOOD, "file": "c:\\work\\mod.py"},
        {**_GOOD, "file": "\\\\host\\share\\mod.py"},
        {**_GOOD, "file": "a/../../etc/passwd"},
        {**_GOOD, "file": "a\\..\\b.py"},
        {**_GOOD, "file": "src/mod\x00.py"},
        {**_GOOD, "file": "."},
        {**_GOOD, "file": 7},
        {**_GOOD, "file": None},
        {k: v for k, v in _GOOD.items() if k != "file"},
        {k: v for k, v in _GOOD.items() if k != "symbol_name"},
    ],
    ids=[
        "abs",
        "drive-fwd",
        "drive-back",
        "unc",
        "dotdot",
        "dotdot-back",
        "nul",
        "dot",
        "int",
        "none",
        "no-file",
        "no-symbol",
    ],
)
async def test_machine_paths_and_malformed_anchors_are_refused_and_nothing_is_written(
    memory_client: MemoryClient, bad: dict[str, object]
) -> None:
    before = memory_client._get_backend().count("default")
    with pytest.raises(SchemaValidationError) as caught:
        await memory_client.store("must not land", anchors=[bad])
    assert caught.value.failed_fields == ["anchors"]
    assert memory_client._get_backend().count("default") == before


async def test_refusal_names_the_offending_value(memory_client: MemoryClient) -> None:
    with pytest.raises(SchemaValidationError, match=re.escape("/srv/checkout/repo/mod.py")):
        await memory_client.store("must not land", anchors=[{**_GOOD, "file": "/srv/checkout/repo/mod.py"}])


async def test_a_model_built_without_validation_is_still_refused(memory_client: MemoryClient) -> None:
    sneaky = Anchor.model_construct(
        file="C:/x.py", symbol_name="f", symbol_type="function", signature="", line_range=None
    )
    before = memory_client._get_backend().count("default")
    with pytest.raises(SchemaValidationError):
        await memory_client.store("must not land", anchors=[sneaky])
    assert memory_client._get_backend().count("default") == before


async def test_more_than_three_anchors_or_a_non_list_is_refused(memory_client: MemoryClient) -> None:
    before = memory_client._get_backend().count("default")
    with pytest.raises(SchemaValidationError, match="at most 3"):
        await memory_client.store("x", anchors=[dict(_GOOD)] * 4)
    with pytest.raises(SchemaValidationError):
        await memory_client.store("x", anchors=dict(_GOOD))  # type: ignore[arg-type]
    assert memory_client._get_backend().count("default") == before


async def test_source_distill_is_accepted_and_comes_back(memory_client: MemoryClient) -> None:
    result = await memory_client.store("mined lesson", source="distill")
    assert _stored(memory_client, result["memory_id"]).source == "distill"


async def test_an_anchored_entry_is_found_by_its_file(memory_client: MemoryClient) -> None:
    """MemoryClient.recall has no file-path parameter; the backend's ``anchored_to`` is the lookup."""
    hit = await memory_client.store("names no file in its text", anchors=[dict(_GOOD)])
    await memory_client.store("unanchored")
    found = memory_client._get_backend().anchored_to("default", "src/pkg/mod.py", status=None, limit=10)
    assert [e.id for e in found] == [hit["memory_id"]]


def test_tool_surface_carries_anchors_and_distill_source(sqlite_memory_backend: StorageBackend) -> None:
    """The daemon's ``memory_store`` tool runs ``memory_store_impl``; it takes the same two things."""
    result = memory_store_impl(
        "mined lesson",
        "default",
        backend=sqlite_memory_backend,
        entry_id="M-tool",
        source="distill",
        anchors=[Anchor(file="src/pkg/mod.py", symbol_name="handler")],
    )
    assert result["status"] == "stored"
    entry = sqlite_memory_backend.get("M-tool", namespace="default")
    assert entry is not None
    assert entry.source == "distill"
    assert [a.symbol_name for a in entry.anchors] == ["handler"]


@pytest.mark.parametrize(
    "bad_file",
    ["C:/work/mod.py", "\\\\host\\share\\mod.py", "src/mod\x00.py", "a\\..\\b.py", "\\etc\\passwd", "."],
    ids=["drive", "unc", "nul", "dotdot-back", "backslash-abs", "dot"],
)
def test_tool_surface_refuses_the_same_anchor_paths_and_writes_nothing(
    sqlite_memory_backend: StorageBackend, bad_file: str
) -> None:
    before = sqlite_memory_backend.count("default")
    result = memory_store_impl(
        "must not land",
        "default",
        backend=sqlite_memory_backend,
        entry_id="M-bad",
        anchors=[Anchor(file=bad_file, symbol_name="f")],
    )
    assert result["status"] == "invalid"
    assert "anchors" in str(result["error"])
    assert sqlite_memory_backend.count("default") == before
    assert sqlite_memory_backend.get("M-bad", namespace="default") is None


_NEW_REFUSALS = {
    "uri": "file:///etc/passwd",
    "https": "https://host/x.py",
    "tilde": "~/.ssh/id_rsa",
    "backslash-rel": "a\\b.py",
    "backslash-dotdot": "a\\..\\b.py",
    "newline": "a\nb.py",
    "tab": "a\tb.py",
    "del": "a\x7fb.py",
    "pad-lead": " src/a.py",
    "pad-trail": "src/a.py ",
    "blank": "   ",
    "too-long": "a/" * 600,
    "empty-segment": "a//b.py",
    "trailing-slash": "src/",
    "dot-segment": "a/./b.py",
    "dotdot-segment": "a/../b.py",
    "double-dot-prefix": "./../b.py",
    "double-leading-dot": "././a.py",
}


@pytest.mark.parametrize("bad_file", list(_NEW_REFUSALS.values()), ids=list(_NEW_REFUSALS))
async def test_new_shapes_refused_through_the_library_door(memory_client: MemoryClient, bad_file: str) -> None:
    before = memory_client._get_backend().count("default")
    with pytest.raises(SchemaValidationError) as caught:
        await memory_client.store("must not land", anchors=[{**_GOOD, "file": bad_file}])
    assert caught.value.failed_fields == ["anchors"]
    assert memory_client._get_backend().count("default") == before


@pytest.mark.parametrize("bad_file", list(_NEW_REFUSALS.values()), ids=list(_NEW_REFUSALS))
def test_new_shapes_refused_through_the_tool_door(sqlite_memory_backend: StorageBackend, bad_file: str) -> None:
    before = sqlite_memory_backend.count("default")
    result = memory_store_impl(
        "must not land",
        "default",
        backend=sqlite_memory_backend,
        entry_id="M-bad2",
        anchors=[Anchor.model_construct(file=bad_file, symbol_name="f", symbol_type="function", signature="")],
    )
    assert result["status"] == "invalid"
    assert sqlite_memory_backend.count("default") == before
    assert sqlite_memory_backend.get("M-bad2", namespace="default") is None


async def test_one_leading_dot_slash_is_accepted_and_indexed_as_the_plain_path(memory_client: MemoryClient) -> None:
    result = await memory_client.store("lesson", anchors=[{**_GOOD, "file": "./src/a.py"}])
    assert _stored(memory_client, result["memory_id"]).anchors[0].file == "./src/a.py"
    found = memory_client._get_backend().anchored_to("default", "src/a.py", status=None, limit=10)
    assert [e.id for e in found] == [result["memory_id"]]


@pytest.mark.parametrize("name", ["src/a%2Fb.py", "%2e%2e/secret.py", "dir/100%.py"])
async def test_percent_encoded_text_is_a_literal_name(memory_client: MemoryClient, name: str) -> None:
    result = await memory_client.store("lesson", anchors=[{**_GOOD, "file": name}])
    assert _stored(memory_client, result["memory_id"]).anchors[0].file == name


async def test_echoed_value_is_truncated_and_repr_rendered(memory_client: MemoryClient) -> None:
    bad = "/x\n" + "y" * 500
    with pytest.raises(SchemaValidationError) as caught:
        await memory_client.store("must not land", anchors=[{**_GOOD, "file": bad}])
    message = str(caught.value)
    assert "\n" not in message
    assert "\\n" in message
    assert "y" * 116 in message
    assert "y" * 121 not in message
