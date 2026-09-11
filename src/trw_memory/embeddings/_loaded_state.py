"""Measured loaded tensor state and dependency versions; no filesystem scans."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys

_PACKAGES = ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy", "safetensors")


def dependency_versions() -> dict[str, str] | None:
    """Resolve loaded implementation's package versions once, never on recall."""
    try:
        versions: dict[str, str] = {}
        for name in _PACKAGES:
            installed = importlib.metadata.version(name)
            module = sys.modules.get(name.replace("-", "_"))
            loaded = getattr(module, "__version__", None)
            # A long-lived process may hold old modules after an installation
            # changes on disk. Installed metadata alone cannot identify its code.
            if not isinstance(loaded, str) or loaded != installed:
                return None
            versions[name] = loaded
        return versions
    except importlib.metadata.PackageNotFoundError:  # trw-fail-silent-allow: None is the typed unknown-identity signal; an unidentifiable runtime must not qualify a vector, so this fails closed
        return None


def loaded_state_digest(model: object) -> str | None:
    """Hash actual loaded CPU parameters and buffers once, without transfers.

    Missing checkpoint keys can be initialized randomly by the loader. Artifact
    identity alone therefore cannot identify loaded state. Include nonpersistent
    buffers, canonical names/dtypes/shapes, and bytes in bounded chunks. Refuse
    unsupported tensors rather than copying GPU/meta/sparse state to CPU. The
    getter never calls this scan. Ordinary concurrent tensor/component mutation
    is detected; malicious mutation bypassing Torch version counters is not.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not isinstance(model, torch.nn.Module):
            return None

        def state() -> list[tuple[str, object]]:
            return sorted(
                [
                    ("parameter:" + name, value)
                    for name, value in model.named_parameters(recurse=True, remove_duplicate=False)
                ]
                + [
                    ("buffer:" + name, value)
                    for name, value in model.named_buffers(recurse=True, remove_duplicate=False)
                ],
                key=lambda item: item[0],
            )

        entries = state()
        if not entries or len({name for name, _ in entries}) != len(entries):
            return None
        digest = hashlib.sha256(b"trw-loaded-cpu-state-v1\0")
        snapshots = []
        total_bytes = 0
        for name, tensor in entries:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or tensor.is_quantized
                or not tensor.is_contiguous()
                or tensor.is_complex()
            ):
                return None
            stamp = (tensor._version, str(tensor.dtype), tuple(tensor.shape))
            snapshots.append((name, tensor, stamp))
            header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            flat = tensor.detach().reshape(-1)
            step = max(1, (1024 * 1024) // tensor.element_size())
            for offset in range(0, tensor.numel(), step):
                chunk = flat[offset : offset + step]
                if not bool(torch.isfinite(chunk).all().item()):
                    return None
                block = chunk.view(torch.uint8).numpy().tobytes()
                digest.update(block)
                total_bytes += len(block)
        after = state()
        if len(after) != len(snapshots):
            return None
        for (name, tensor), (old_name, old_tensor, stamp) in zip(after, snapshots, strict=True):
            if name != old_name or tensor is not old_tensor:
                return None
            if (tensor._version, str(tensor.dtype), tuple(tensor.shape)) != stamp:
                return None
        return digest.hexdigest() if total_bytes else None
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):  # trw-fail-silent-allow: None is the typed unknown-identity signal -- a digest that cannot be computed must not be treated as a match
        return None
