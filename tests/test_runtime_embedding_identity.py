"""Runtime semantic identity tests with canonical-module protocol doubles.

No model imports, files, or inference. Actual supported MiniLM activation is a
separate integration proof; these doubles exercise the fingerprint contracts.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from trw_memory.embeddings._runtime_identity import capture_runtime_identity, runtime_identity_matches


class Node:
    def get_config_dict(self):
        return self.settings

    def extra_repr(self):
        return ""


def canonical(monkeypatch, module, name):
    cls = type(name, (Node,), {"__module__": module})
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{name: cls}))
    return cls


@pytest.fixture
def encoder(monkeypatch):
    root = canonical(monkeypatch, "sentence_transformers.sentence_transformer.model", "SentenceTransformer")()
    transformer = canonical(monkeypatch, "sentence_transformers.base.modules.transformer", "Transformer")()
    pooling = canonical(monkeypatch, "sentence_transformers.sentence_transformer.modules.pooling", "Pooling")()
    normalize = canonical(monkeypatch, "sentence_transformers.base.modules.normalize", "Normalize")()
    bert = canonical(monkeypatch, "transformers.models.bert.modeling_bert", "BertModel")()
    tokenizer = canonical(monkeypatch, "transformers.models.bert.tokenization_bert", "BertTokenizer")()
    formatter = canonical(monkeypatch, "sentence_transformers.base.modality", "InputFormatter")()
    backend = canonical(monkeypatch, "tokenizers", "Tokenizer")()
    backend.data = {"model": {"type": "WordPiece", "vocab": {"hello": 1}}, "normalizer": {"lowercase": True}}
    backend.to_str = lambda: json.dumps(backend.data)
    tokenizer.backend_tokenizer = backend
    tokenizer.is_fast = True
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = 512
    tokenizer.model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    tokenizer.split_special_tokens = False
    tokenizer.special_tokens_map = {"pad_token": "[PAD]"}
    tokenizer.pad_token_id = 0
    tokenizer.pad_token_type_id = 0
    formatter.processor = tokenizer
    formatter.supported_modalities = ["text"]
    formatter.model_type = "bert"
    formatter.message_format = "structured"
    bert.config = SimpleNamespace(_attn_implementation="sdpa", to_dict=lambda: {"model_type": "bert", "hidden_size": 2})
    transformer.backend = "torch"
    transformer.transformer_task = "feature-extraction"
    transformer.modality_config = {"text": {}}
    transformer.model = bert
    transformer.processor = transformer.tokenizer = tokenizer
    transformer.input_formatter = formatter
    transformer.settings = {"processing_kwargs": {"text": {"truncation": True}}, "unpad_inputs": False}
    pooling.settings = {"pooling_mode": "mean", "include_prompt": True}
    normalize.settings = {"module_input_name": "sentence_embedding", "module_output_name": "sentence_embedding"}
    root._modules = {"0": transformer, "1": pooling, "2": normalize}
    root.backend = "torch"
    root.dtype = "torch.float32"
    root.prompts = {"query": ""}
    root.default_prompt_name = None
    root.truncate_dim = None
    root.module_kwargs = None
    root.named_modules = lambda: [("", root), ("0", transformer), ("1", pooling), ("2", normalize), ("0.model", bert)]
    return root


VERSIONS = {"sentence-transformers": "6.0.1", "transformers": "5.16.1", "torch": "2.14.0+cpu"}


def capture(model):
    result = capture_runtime_identity(model, 2, VERSIONS)
    assert result is not None
    return result


def test_stable_identity_and_version_identity_not_version_allowlist(encoder):
    first = capture(encoder)
    assert capture(encoder)[0] == first[0]
    assert runtime_identity_matches(encoder, 2, first[1])
    other = capture_runtime_identity(encoder, 2, {**VERSIONS, "transformers": "different-version"})
    assert other is not None and other[0] != first[0]


@pytest.mark.parametrize(
    "setting", ["vocab", "pooling", "attention", "padding", "processing", "prompt", "normalization"]
)
def test_effective_semantic_changes_alter_identity(encoder, setting):
    first, guard = capture(encoder)
    transformer = encoder._modules["0"]
    if setting == "vocab":
        transformer.processor.backend_tokenizer.data["model"]["vocab"]["new"] = 2
    elif setting == "pooling":
        encoder._modules["1"].settings["pooling_mode"] = "max"
    elif setting == "attention":
        transformer.model.config._attn_implementation = "eager"
    elif setting == "padding":
        transformer.processor.padding_side = "left"
    elif setting == "processing":
        transformer.settings["processing_kwargs"]["text"]["truncation"] = False
    elif setting == "normalization":
        encoder._modules["2"].settings["module_input_name"] = "token_embeddings"
    else:
        encoder.prompts["query"] = "prefix: "
    assert capture(encoder)[0] != first
    if setting != "vocab":
        assert not runtime_identity_matches(encoder, 2, guard)
    # In-place tokenizer vocabulary mutation violates the immutable lifetime
    # contract. The scan-free getter deliberately does not claim to detect it.


def test_transient_backend_batch_settings_do_not_change_identity(encoder):
    first = capture(encoder)[0]
    encoder._modules["0"].processor.backend_tokenizer.data.update(
        {"padding": {"length": 17}, "truncation": {"max_length": 21}}
    )
    assert capture(encoder)[0] == first


def test_getter_does_not_serialize_tokenizer_or_model_graph(encoder):
    _, guard = capture(encoder)

    def forbidden():
        raise AssertionError("unexpected expensive operation")

    encoder._modules["0"].processor.backend_tokenizer.to_str = forbidden
    encoder.named_modules = forbidden
    encoder._modules["0"].model.config.to_dict = forbidden
    assert runtime_identity_matches(encoder, 2, guard)


@pytest.mark.parametrize("kind", ["hooks", "compiled", "adapter", "multimodal", "processor", "custom-code"])
def test_unsupported_runtime_is_unknown(encoder, kind):
    if kind == "hooks":
        encoder._forward_hooks = {1: object()}
    elif kind == "compiled":
        encoder._orig_mod = object()
    elif kind == "adapter":
        encoder._modules["0"].model.peft_config = {"adapter": "x"}
    elif kind == "multimodal":
        encoder._modules["0"].modality_config["image"] = {}
    elif kind == "processor":
        encoder._modules["0"].processor = SimpleNamespace()
    else:
        encoder.trust_remote_code = True
    assert capture_runtime_identity(encoder, 2, VERSIONS) is None


def test_component_replacement_and_bad_guard_are_rejected(encoder):
    _, guard = capture(encoder)
    encoder._modules["1"] = type(encoder._modules["1"])()
    encoder._modules["1"].settings = {"pooling_mode": "mean", "include_prompt": True}
    assert not runtime_identity_matches(encoder, 2, guard)
    assert not runtime_identity_matches(encoder, 2, object())


def test_load_origin_is_not_effective_identity(encoder):
    config = encoder._modules["0"].model.config
    config.to_dict = lambda: {"model_type": "bert", "hidden_size": 2, "_name_or_path": "/first/copy"}
    first = capture(encoder)[0]
    config.to_dict = lambda: {"model_type": "bert", "hidden_size": 2, "_name_or_path": "/second/copy"}
    assert capture(encoder)[0] == first
    config.to_dict = lambda: {"model_type": "bert", "hidden_size": 3, "_name_or_path": "/second/copy"}
    assert capture(encoder)[0] != first
