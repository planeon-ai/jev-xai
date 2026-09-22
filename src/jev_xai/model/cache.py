"""In-process call cache. The cassette is the durable form of the same idea."""

from __future__ import annotations

from jev_xai.types import Prediction


class MemoryCache:
    def __init__(self) -> None:
        self._items: dict[str, Prediction] = {}

    def get(self, key: str) -> Prediction | None:
        return self._items.get(key)

    def put(self, key: str, prediction: Prediction) -> None:
        self._items[key] = prediction
