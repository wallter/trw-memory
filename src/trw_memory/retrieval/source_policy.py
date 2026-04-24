"""Source-aware recall policy helpers for multi-source retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from collections.abc import Mapping, Sequence

SourceFamily = str

DEFAULT_SOURCE_WEIGHTS: dict[SourceFamily, float] = {
    "git_distilled": 0.75,
    "instruction_rule": 0.95,
    "semantic_memory": 0.9,
    "lifecycle": 0.55,
    "episodic": 0.45,
    "unknown": 1.0,
}


def classify_source_family(result: Mapping[str, Any]) -> SourceFamily:
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        explicit = str(metadata.get("source_kind", "")).strip()
        if explicit == "git":
            return "git_distilled"
        if explicit in {"instruction_rule", "semantic_memory", "lifecycle", "episodic"}:
            return explicit
        source = str(metadata.get("source", "")).strip()
        if source.startswith("distilled:git:"):
            return "git_distilled"
        if source.startswith("distilled:bulletin:"):
            return "lifecycle"
    tags = result.get("tags", []) or []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        if tag.startswith("source_kind:"):
            family = tag.split(":", 1)[1]
            if family == "git":
                return "git_distilled"
            if family in {"instruction_rule", "semantic_memory", "lifecycle", "episodic"}:
                return family
        if tag.startswith(("distill:", "distilled:")):
            return "git_distilled"
        if tag == "change_bulletin":
            return "lifecycle"
    return "unknown"


def resolve_expiry(result: Mapping[str, Any]) -> str:
    raw = result.get("expires")
    if isinstance(raw, str) and raw:
        return raw
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        meta_expiry = metadata.get("expires")
        if isinstance(meta_expiry, str):
            return meta_expiry
    return ""


def is_expired_result(result: Mapping[str, Any], *, now: datetime | None = None) -> bool:
    raw = resolve_expiry(result)
    if not raw:
        return False
    try:
        expiry = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    current = now or datetime.now(timezone.utc)
    return expiry <= current


def apply_source_policy(
    results: Sequence[Mapping[str, Any]],
    *,
    include_distilled: bool = True,
    distilled_weight: float | None = None,
    include_source_kinds: list[str] | None = None,
    exclude_source_kinds: list[str] | None = None,
    source_weights: dict[str, float] | None = None,
    exclude_expired: bool = True,
) -> list[dict[str, Any]]:
    include_set = set(include_source_kinds or [])
    exclude_set = set(exclude_source_kinds or [])
    weights = dict(DEFAULT_SOURCE_WEIGHTS)
    if source_weights:
        weights.update(source_weights)
    if distilled_weight is not None:
        weights["git_distilled"] = distilled_weight

    filtered: list[dict[str, Any]] = []
    for result in results:
        family = classify_source_family(result)
        if family == "git_distilled" and not include_distilled:
            continue
        if include_set and family not in include_set:
            continue
        if family in exclude_set:
            continue
        if exclude_expired and family in {"lifecycle", "episodic"} and is_expired_result(result):
            continue
        weight = weights.get(family, 1.0)
        if weight <= 0.0:
            continue
        if weight >= 1.0 - 1e-9:
            filtered.append(dict(result))
            continue
        adjusted = dict(result)
        adjusted["score"] = float(result.get("score", 0.0)) * weight
        filtered.append(adjusted)
    filtered.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return filtered


__all__ = [
    "DEFAULT_SOURCE_WEIGHTS",
    "apply_source_policy",
    "classify_source_family",
    "is_expired_result",
    "resolve_expiry",
]
