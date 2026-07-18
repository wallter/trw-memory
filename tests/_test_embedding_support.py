"""Reusable deterministic embedding provider for memory tests."""

from __future__ import annotations


class StubEmbedder:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self._vectors: dict[str, list[float]] = {}

    def set_vector(self, text: str, vector: list[float]) -> None:
        self._vectors[text] = vector

    def embed(self, text: str) -> list[float] | None:
        if not self._available:
            return None
        if text in self._vectors:
            return self._vectors[text]
        return [float(ord(char)) / 128.0 for char in text[:3].ljust(3)]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def available(self) -> bool:
        return self._available

    def dim(self) -> int:
        return 3
