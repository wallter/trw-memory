"""Loaded-state identity and provider integration; no filesystem ancestry policy."""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from trw_memory.embeddings._declared_space import DECLARED_ENCODING_PREFIX, declared_embedding_space
from trw_memory.embeddings._hf_cache import CacheProbe, CacheState
from trw_memory.embeddings._loaded_state import _PACKAGES, dependency_versions, loaded_state_digest
from trw_memory.embeddings.local import LocalEmbeddingProvider


class FakeTensor:
    """CPU tensor protocol double; real-Torch coverage is separately conditional."""

    def __init__(self, values):
        self.array = np.asarray(values, dtype=np.float32)
        self.device = SimpleNamespace(type="cpu")
        self.layout = "strided"
        self.is_quantized = False
        self._version = 0

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def is_complex(self):
        return np.iscomplexobj(self.array)

    def element_size(self):
        return self.array.itemsize

    def numel(self):
        return self.array.size

    def detach(self):
        return self

    def reshape(self, *shape):
        result = FakeTensor([])
        result.array = self.array.reshape(*shape)
        return result

    def __getitem__(self, key):
        result = FakeTensor([])
        result.array = self.array[key]
        return result

    def view(self, dtype):
        result = FakeTensor([])
        result.array = self.array.view(dtype)
        return result

    def numpy(self):
        return self.array


def register_torch_double(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Tensor=FakeTensor,
            nn=SimpleNamespace(Module=FakeModel),
            strided="strided",
            uint8=np.uint8,
            isfinite=lambda tensor: np.isfinite(tensor.array),
        ),
    )


class FakeModel:
    backend = "torch"

    def __init__(self):
        self.weights = FakeTensor([1.0, 2.0])
        self.buffer = FakeTensor([3.0])

    def named_parameters(self, **kwargs):
        return [("weights", self.weights)]

    def named_buffers(self, **kwargs):
        return [("buffer", self.buffer)]

    prompts = {"query": "query: "}
    default_prompt_name = None
    max_seq_length = 256
    truncate_dim = None
    dtype = "torch.float32"

    def encode(self, text, **kwargs):
        return [1.0, 0.0]


def test_loaded_state_changes_for_weights_buffers_names_and_shapes(monkeypatch):
    register_torch_double(monkeypatch)
    model = FakeModel()
    first = loaded_state_digest(model)
    assert first is not None and first == loaded_state_digest(FakeModel())
    model.weights = FakeTensor([9.0, 2.0])
    assert loaded_state_digest(model) != first
    model = FakeModel()
    model.buffer = FakeTensor([8.0])
    assert loaded_state_digest(model) != first
    model = FakeModel()
    model.named_parameters = lambda **kwargs: [("renamed", model.weights)]
    assert loaded_state_digest(model) != first
    model = FakeModel()
    model.weights.array = model.weights.array.reshape(1, 2)
    assert loaded_state_digest(model) != first


@pytest.mark.parametrize("change", ["nan", "cuda", "meta", "sparse", "quantized", "noncontiguous"])
def test_unsupported_state_stays_unknown_without_transfer(monkeypatch, change):
    register_torch_double(monkeypatch)
    model = FakeModel()
    if change == "nan":
        model.weights = FakeTensor([float("nan")])
    elif change in ("cuda", "meta"):
        model.weights.device.type = change
        model.weights.detach = lambda: pytest.fail("nonCPU state accessed")
    elif change == "sparse":
        model.weights.layout = "sparse"
    elif change == "quantized":
        model.weights.is_quantized = True
    else:
        model.weights.is_contiguous = lambda: False
    assert loaded_state_digest(model) is None


def test_loaded_and_installed_dependency_versions_must_agree(monkeypatch):
    monkeypatch.setattr("trw_memory.embeddings._loaded_state.importlib.metadata.version", lambda name: "1.0")
    for name in _PACKAGES:
        monkeypatch.setitem(sys.modules, name.replace("-", "_"), SimpleNamespace(__version__="1.0"))
    assert dependency_versions() == dict.fromkeys(_PACKAGES, "1.0")
    sys.modules["transformers"].__version__ = "other"
    assert dependency_versions() is None


def test_real_torch_state_includes_nonpersistent_buffers():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2, device="cpu")
    model.register_buffer("nonpersistent", torch.tensor([1.0], device="cpu"), persistent=False)
    first = loaded_state_digest(model)
    assert first is not None and first == loaded_state_digest(model)
    with torch.no_grad():
        model.nonpersistent.add_(1)
    assert first != loaded_state_digest(model)


@pytest.mark.parametrize("qualification", ["known", "unknown", "error"])
def test_local_provider_uses_runtime_identity_without_file_identity(monkeypatch, qualification):
    register_torch_double(monkeypatch)
    model = FakeModel()
    monkeypatch.setattr(
        "trw_memory.embeddings.local._hide_broken_torchcodec_for_sentence_transformers", contextlib.nullcontext
    )
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *a, **kw: model)
    )
    monkeypatch.setattr("trw_memory.embeddings.local.dependency_versions", lambda: {"fixture": "1"})
    guard = object()

    def capture(*args):
        if qualification == "error":
            raise RuntimeError("optional identity unavailable")
        return ("b" * 64, guard) if qualification == "known" else None

    monkeypatch.setattr("trw_memory.embeddings.local.capture_runtime_identity", capture)
    monkeypatch.setattr(
        "trw_memory.embeddings.local.runtime_identity_matches", lambda m, d, g: m is model and g is guard
    )
    provider = LocalEmbeddingProvider("model-name", dim=2)
    monkeypatch.setattr(
        provider, "_probe_cache", lambda: CacheProbe(CacheState.COMPLETE, snapshot_path="/not/a/real/path")
    )
    assert provider.embed("input") == [1.0, 0.0]
    descriptor = provider.embedding_space()
    assert descriptor is not None
    if qualification != "known":
        # No measured identity: the provider still names the model it loaded
        # (declared tier), which never equals a measured descriptor.
        assert descriptor == declared_embedding_space("model-name", "", 2)
        assert descriptor.encoding.startswith(DECLARED_ENCODING_PREFIX)
    else:
        assert descriptor.artifact_sha256 == loaded_state_digest(model)
        assert descriptor.encoding == "trw-loaded-encoder-v2:" + "b" * 64
        monkeypatch.setattr(
            "trw_memory.embeddings.local.capture_runtime_identity", lambda *a: pytest.fail("recapture on getter")
        )
        monkeypatch.setattr(
            "trw_memory.embeddings.local.loaded_state_digest", lambda *a: pytest.fail("state scan on getter")
        )
        assert provider.embedding_space() == descriptor
        provider._model = FakeModel()
        assert provider.embedding_space() is None
