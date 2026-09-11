"""Effective identity of a loaded standard text/BERT encoder, not its files.

Source-grounded against ST 6.0.1 / Transformers 5.16.1. Versions enter identity,
not an equality allowlist. Unknown runtime shapes remain unknown. Parameters and
buffers are hashed separately by the owner. Capture occurs once per immutable
provider lifetime; the getter detects component swaps and supported small setting
changes, not arbitrary in-place weight/tokenizer/code mutation. No imports of
model libraries, file access, model loading or inference occur here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any, cast

_ROOT = ("sentence_transformers.sentence_transformer.model", "SentenceTransformer")
_TRANSFORMER = ("sentence_transformers.base.modules.transformer", "Transformer")
_POOLING = ("sentence_transformers.sentence_transformer.modules.pooling", "Pooling")
_NORMALIZE = ("sentence_transformers.base.modules.normalize", "Normalize")
_BERT = ("transformers.models.bert.modeling_bert", "BertModel")
_TOKENIZER = ("transformers.models.bert.tokenization_bert", "BertTokenizer")
_FORMATTER = ("sentence_transformers.base.modality", "InputFormatter")
_HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")


@dataclass(frozen=True)
class RuntimeGuard:
    """Request-independent strong references and captured small configuration."""

    model: object
    objects: tuple[object, ...]
    dimensions: int
    settings_json: str


def _exact(value: object, reference: tuple[str, str]) -> bool:
    cls = getattr(sys.modules.get(reference[0]), reference[1], None)
    return isinstance(cls, type) and type(value) is cls


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clean(value: object) -> bool:
    return not (
        any(getattr(value, key, None) for key in _HOOKS)
        or getattr(value, "_orig_mod", None) is not None
        or getattr(value, "_compiled_call_impl", None) is not None
        or getattr(value, "peft_config", None)
        or any(key in vars(value) for key in ("forward", "encode", "preprocess"))
    )


def _objects(model: object) -> tuple[object, ...] | None:
    if not _exact(model, _ROOT) or not _clean(model) or getattr(model, "backend", None) != "torch":
        return None
    modules = getattr(model, "_modules", None)
    if not isinstance(modules, dict) or list(modules) not in (["0", "1"], ["0", "1", "2"]):
        return None
    parts = tuple(modules.values())
    for part, reference in zip(parts, (_TRANSFORMER, _POOLING, _NORMALIZE), strict=False):
        if not _exact(part, reference) or not _clean(part):
            return None
    transformer = parts[0]
    bert, processor = transformer.model, transformer.processor
    formatter = transformer.input_formatter
    if (
        not _exact(bert, _BERT)
        or not _clean(bert)
        or not _exact(processor, _TOKENIZER)
        or not _exact(formatter, _FORMATTER)
        or transformer.tokenizer is not processor
        or formatter.processor is not processor
        or formatter.supported_modalities != ["text"]
        or set(transformer.modality_config) != {"text"}
        or transformer.transformer_task != "feature-extraction"
        or transformer.backend != "torch"
        or not processor.is_fast
        or getattr(processor, "chat_template", None)
        or getattr(model, "trust_remote_code", False)
    ):
        return None
    backend = processor.backend_tokenizer
    if not _exact(backend, ("tokenizers", "Tokenizer")):
        return None
    return (*parts, bert, processor, backend, formatter)


# Reflective optional-library boundary: _objects establishes canonical runtime
# classes without importing Torch/Transformers (including their type stubs).
def _settings(model: Any, dimensions: int, objects: tuple[Any, ...]) -> str:
    transformer, pooling = objects[:2]
    processor, formatter = transformer.processor, transformer.input_formatter
    config = transformer.model.config
    normalization = model._modules.get("2")
    # get_config_dict is upstream's effective runtime configuration surface;
    # tokenizer backend, vocabulary and tensor state are deliberately not scanned.
    return _json(
        {
            "dimensions": dimensions,
            "dtype": str(model.dtype),
            "root": {
                "prompts": model.prompts,
                "default_prompt_name": model.default_prompt_name,
                "truncate_dim": model.truncate_dim,
                "module_kwargs": model.module_kwargs,
            },
            "transformer": transformer.get_config_dict(),
            "pooling": pooling.get_config_dict(),
            "normalize": normalization.get_config_dict() if normalization is not None else None,
            "attention": getattr(config, "_attn_implementation", None),
            "formatter": {
                "model_type": formatter.model_type,
                "message_format": formatter.message_format,
                "supported_modalities": formatter.supported_modalities,
            },
            "tokenizer": {
                "padding_side": processor.padding_side,
                "truncation_side": processor.truncation_side,
                "model_max_length": processor.model_max_length,
                "model_input_names": processor.model_input_names,
                "split_special_tokens": processor.split_special_tokens,
                "special_tokens_map": processor.special_tokens_map,
                "pad_token_id": processor.pad_token_id,
                "pad_token_type_id": processor.pad_token_type_id,
            },
            "call": {
                "method": "encode",
                "input": "str-or-list[str]",
                "normalize_embeddings": True,
                "precision": "float32",
                "output_value": "sentence_embedding",
                "task": None,
            },
        }
    )


def capture_runtime_identity(
    model: object, dimensions: int, versions: dict[str, str]
) -> tuple[str, RuntimeGuard] | None:
    """Capture effective nontensor identity once; unsupported shapes fail open."""
    try:
        if (
            type(dimensions) is not int
            or dimensions < 1
            or not versions
            or not all(isinstance(key, str) and isinstance(value, str) and value for key, value in versions.items())
        ):
            return None
        objects = _objects(model)
        if objects is None:
            return None
        loaded = cast("Any", model)  # canonical classes already checked above
        graph = []
        for name, module in loaded.named_modules():
            cls = type(module)
            if not _clean(module) or not _exact(module, (cls.__module__, cls.__name__)):
                return None
            if not cls.__module__.startswith(
                ("torch.nn.modules.", "transformers.models.bert.", "transformers.activations", "sentence_transformers.")
            ):
                return None
            graph.append([name, cls.__module__, cls.__name__, module.extra_repr()])
        settings = _settings(model, dimensions, objects)
        tokenizer = json.loads(loaded._modules["0"].processor.backend_tokenizer.to_str())
        if not isinstance(tokenizer, dict):
            return None
        # The canonical tokenizer call overwrites both for every batch. Their
        # transient last-batch values are not immutable tokenizer semantics.
        tokenizer.pop("padding", None)
        tokenizer.pop("truncation", None)
        config = loaded._modules["0"].model.config.to_dict()
        # Transformers 5.16.1 configuration_utils.name_or_path records the
        # from_pretrained origin, not BERT forward configuration. Equivalent
        # loaded encoders copied to different directories share an identity.
        config.pop("_name_or_path", None)
        manifest = _json(
            {
                "contract": "trw-loaded-text-bert-v1",
                "dependencies": versions,
                "settings": json.loads(settings),
                "model_config": config,
                "module_graph": graph,
                "tokenizer_backend": tokenizer,
            }
        )
        return hashlib.sha256(manifest.encode()).hexdigest(), RuntimeGuard(model, objects, dimensions, settings)
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):  # trw-fail-silent-allow: None is the typed unknown-identity signal; no identity means no qualified provenance, which is the fail-closed direction
        return None


def runtime_identity_matches(model: object, dimensions: int, guard: object) -> bool:
    """Cheap lifetime guard; no tokenizer serialization or tensor/file scan."""
    if not isinstance(guard, RuntimeGuard) or model is not guard.model or dimensions != guard.dimensions:
        return False
    try:
        objects = _objects(model)
        return (
            objects is not None
            and len(objects) == len(guard.objects)
            and all(current is old for current, old in zip(objects, guard.objects, strict=True))
            and _settings(model, dimensions, objects) == guard.settings_json
        )
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):  # trw-fail-silent-allow: False means "not provably the same runtime", which forces re-derivation -- the safe direction for a lifetime guard
        return False
