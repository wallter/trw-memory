"""Fixture Hugging Face cache + network seam for the PRD-SEC-014 embedding tests.

Builds the real hub cache layout (``refs/`` + ``blobs/`` + symlinked
``snapshots/``) so the probe under test walks the same structure it walks in
production, and installs a socket-level seam that counts and refuses every
outbound connection attempt.
"""

from __future__ import annotations

import hashlib
import json
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

DEFAULT_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "0f1e2d3c4b5a69788796a5b4c3d2e1f009182736"

# Mirrors a real all-MiniLM-L6-v2 snapshot: JSON/text/safetensors only, no
# Python module — the FR03 invariant the default model must keep satisfying.
DEFAULT_SNAPSHOT_FILES: Mapping[str, str] = {
    "config.json": '{"hidden_size": 384, "model_type": "bert"}',
    "config_sentence_transformers.json": '{"__version__": {"sentence_transformers": "2.2.2"}}',
    "modules.json": '[{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}]',
    "sentence_bert_config.json": '{"max_seq_length": 256}',
    "tokenizer_config.json": '{"model_max_length": 512}',
    "vocab.txt": "[PAD]\n[UNK]\n",
    "model.safetensors": "not-real-weights",
}

# A repo that ships its own modeling code: loading it executes Hub-fetched Python.
REMOTE_CODE_SNAPSHOT_FILES: Mapping[str, str] = {
    **DEFAULT_SNAPSHOT_FILES,
    "config.json": '{"hidden_size": 768, "auto_map": {"AutoModel": "modeling_custom.CustomModel"}}',
    "modeling_custom.py": "class CustomModel:  # executed on load\n    pass\n",
}


def _repo_folder(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def build_model_cache(
    root: Path,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    files: Mapping[str, str] | None = None,
    revision: str = DEFAULT_REVISION,
) -> Path:
    """Create an HF hub cache under ``root/hub`` and return the snapshot dir.

    ``root`` is what the test exports as ``HF_HOME``.
    """
    payload = dict(DEFAULT_SNAPSHOT_FILES if files is None else files)
    repo_cache = root / "hub" / _repo_folder(repo_id)
    blobs = repo_cache / "blobs"
    snapshot = repo_cache / "snapshots" / revision
    (repo_cache / "refs").mkdir(parents=True, exist_ok=True)
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    (repo_cache / "refs" / "main").write_text(revision, encoding="utf-8")
    for name, content in payload.items():
        # Cache key only, never a security digest.
        blob = blobs / hashlib.sha1(f"{name}:{content}".encode()).hexdigest()
        blob.write_text(content, encoding="utf-8")
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
    return snapshot


def delete_blob_behind(snapshot: Path, filename: str) -> None:
    """Delete the blob a snapshot entry points at, leaving a dangling link."""
    (snapshot / filename).resolve().unlink()


def use_fixture_cache(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Point every hub cache env var at *root* and clear the offline switches.

    huggingface_hub freezes ``constants.HF_HUB_CACHE`` at *import* time, and
    transformers resolves checkpoints through it. Importing it here — before the
    env change — keeps a first import from freezing the temporary fixture path
    into the rest of the pytest session (which made a later real model load read
    a stub ``model.safetensors``). The monkeypatched constant is restored on
    teardown, so the override is scoped to the test.
    """
    from huggingface_hub import constants

    monkeypatch.setenv("HF_HOME", str(root))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(root / "hub"))
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRW_OFFLINE", "HF_HUB_OFFLINE", "MEMORY_LOCAL_ONLY"):
        monkeypatch.delenv(var, raising=False)


def simulated_hub_request() -> None:
    """Stand-in for the Hub revision check a network-capable load performs.

    Routed through ``socket.create_connection`` so it hits the same seam that
    intercepts real egress — a test that forgets to install :class:`NetworkSeam`
    fails loudly instead of dialing huggingface.co.
    """
    socket.create_connection(("huggingface.co", 443), timeout=0.001)


def install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: Exception | None = None,
) -> dict[str, object]:
    """Install a fake ``sentence_transformers`` module; return captured kwargs.

    The fake performs :func:`simulated_hub_request` whenever it is constructed
    with ``local_files_only=False``, which is what makes "did the loader decide
    to permit egress?" observable without loading a real model.
    """
    import sys
    import types

    captured: dict[str, object] = {}

    class _FakeSentenceTransformer:
        def __init__(
            self,
            model_name: str,
            *,
            local_files_only: bool = False,
            trust_remote_code: bool = False,
        ) -> None:
            captured["model_name"] = model_name
            captured["local_files_only"] = local_files_only
            captured["trust_remote_code"] = trust_remote_code
            if not local_files_only:
                simulated_hub_request()
            if error is not None:
                raise error

        def encode(self, *args: Any, **kwargs: Any) -> list[float]:
            return [0.0]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return captured


class NetworkSeam:
    """Counts and refuses outbound connection attempts."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError(f"network seam invoked: {args!r}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "create_connection", self)
        monkeypatch.setattr(socket.socket, "connect", self)
        monkeypatch.setattr(socket.socket, "connect_ex", self)


# A minimal but genuinely loadable sentence-transformers model, built from
# transformers primitives with no network access. Deliberately tiny (8 hidden
# units, 1 layer, 9-token vocab) so the real loader can be exercised in an
# ordinary test rather than behind a `slow` marker that default runs skip.
_TINY_VOCAB = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "warm", "cache", "hello", "world")
TINY_MODEL_DIM = 8
_TINY_MAX_SEQ = 32
_TINY_MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
]


def build_loadable_model_cache(
    root: Path,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    revision: str = DEFAULT_REVISION,
) -> Path:
    """Build a real, loadable ST model inside an HF-layout cache under ``root``.

    Unlike :func:`build_model_cache` (stub bytes, for probe-state assertions),
    every file here is produced by transformers itself, so the genuine
    ``SentenceTransformer`` can load the result. That is what makes a zero-egress
    assertion meaningful: a fake loader cannot reveal what the real one does with
    the arguments it is handed, and the real one ignored ``local_files_only``.

    Requires the ``[embeddings]`` extra; callers guard with ``importorskip``.
    """
    from transformers import BertConfig, BertModel, BertTokenizerFast

    repo_cache = root / "hub" / _repo_folder(repo_id)
    snapshot = repo_cache / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (repo_cache / "refs").mkdir(parents=True, exist_ok=True)
    (repo_cache / "refs" / "main").write_text(revision, encoding="utf-8")

    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(_TINY_VOCAB) + "\n", encoding="utf-8")
    config = BertConfig(
        vocab_size=len(_TINY_VOCAB),
        hidden_size=TINY_MODEL_DIM,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=TINY_MODEL_DIM,
        max_position_embeddings=_TINY_MAX_SEQ,
    )
    BertModel(config).save_pretrained(snapshot)
    BertTokenizerFast(vocab_file=str(vocab_file), model_max_length=_TINY_MAX_SEQ).save_pretrained(snapshot)

    (snapshot / "modules.json").write_text(json.dumps(_TINY_MODULES), encoding="utf-8")
    (snapshot / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": _TINY_MAX_SEQ, "do_lower_case": False}),
        encoding="utf-8",
    )
    pooling = snapshot / "1_Pooling"
    pooling.mkdir(exist_ok=True)
    (pooling / "config.json").write_text(
        json.dumps(
            {
                "word_embedding_dimension": TINY_MODEL_DIM,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
            }
        ),
        encoding="utf-8",
    )
    return snapshot
