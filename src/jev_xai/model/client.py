"""Async-first model client: cache, cassette, batching, budgets, retries."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import anyio

from jev_xai.async_compat import run_sync
from jev_xai.canonical import content_hash
from jev_xai.config.schema import ModelClientConfig
from jev_xai.errors import BudgetExceededError, ModelCallError
from jev_xai.model.cache import MemoryCache
from jev_xai.model.cassette import Cassette
from jev_xai.replay.seeding import generator
from jev_xai.types import CostEnvelope, Prediction, coerce_prediction


class ModelClient:
    """The only object explainers call.

    Sync user code is run via ``anyio.to_thread``. A sync facade raises
    :class:`jev_xai.errors.JevXaiUsageError` inside a running event loop.
    """

    def __init__(
        self,
        model: Any,
        config: ModelClientConfig,
        *,
        fingerprint: str,
        seed: int = 0,
        cassette: Cassette | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.fingerprint = fingerprint
        self.seed = seed
        self.cassette = cassette
        self.memory = MemoryCache()
        self.noise_floor: float | None = None
        self._calls = 0
        self._hits = 0
        self._wall_ms = 0.0
        self._ignore_cache = 0
        self._lock: anyio.Lock | None = None
        self._rng = generator(seed)

    @contextmanager
    def ignore_cache(self) -> Iterator[None]:
        """Call the model instead of the cache, and do not write those calls back.

        Stability measurement uses this so a warm cache cannot look like a
        repeated experiment. The cassette is left as it was.
        """

        self._ignore_cache += 1
        try:
            yield
        finally:
            self._ignore_cache -= 1

    def cache_key(self, instance: Mapping[str, Any]) -> str:
        return content_hash({"fingerprint": self.fingerprint, "input": dict(instance)})

    def cost(self) -> CostEnvelope:
        return CostEnvelope(
            n_model_calls=self._calls,
            cache_hits=self._hits,
            wall_ms=self._wall_ms,
        )

    def _lock_for(self) -> anyio.Lock:
        if self._lock is None:
            self._lock = anyio.Lock()
        return self._lock

    def _lookup(self, key: str) -> Prediction | None:
        mode = self.config.cache_mode
        if self._ignore_cache or mode == "off":
            return None
        if mode == "memory":
            return self.memory.get(key)
        if self.cassette is None:
            return self.memory.get(key)
        found = self.cassette.get(key)
        if found is not None:
            self.memory.put(key, found)
        return found

    def _store(self, key: str, prediction: Prediction) -> None:
        mode = self.config.cache_mode
        if self._ignore_cache or mode == "off":
            return
        self.memory.put(key, prediction)
        if mode == "cassette" and self.cassette is not None:
            self.cassette.put(key, prediction)

    def force_store(self, key: str, prediction: Prediction) -> None:
        """Persist a call even when the cache mode is off, so evidence replay works."""

        self.memory.put(key, prediction)
        if self.cassette is not None:
            self.cassette.put(key, prediction)

    async def predict(self, instance: Mapping[str, Any], *, use_cache: bool = True) -> Prediction:
        started = time.perf_counter()
        key = self.cache_key(instance)
        try:
            if use_cache:
                async with self._lock_for():
                    cached = self._lookup(key)
                    if cached is not None:
                        self._hits += 1
                        return cached
            prediction = await self._invoke_with_retry(instance)
            if use_cache:
                async with self._lock_for():
                    self._store(key, prediction)
            return prediction
        finally:
            self._wall_ms += (time.perf_counter() - started) * 1000

    def predict_sync(self, instance: Mapping[str, Any]) -> Prediction:
        return run_sync(self.predict, instance)

    async def predict_many(self, rows: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        results: list[Prediction | None] = [None] * len(rows)
        missing: list[int] = []
        for index, row in enumerate(rows):
            key = self.cache_key(row)
            cached = self._lookup(key)
            if cached is not None:
                self._hits += 1
                results[index] = cached
            else:
                missing.append(index)
        if not missing:
            return [item for item in results if item is not None]
        if self._has_batch():
            size = max(1, self.config.batch_size)
            for start in range(0, len(missing), size):
                chunk = missing[start : start + size]
                preds = await self._invoke_batch([rows[index] for index in chunk])
                async with self._lock_for():
                    self._reserve_call()
                for index, prediction in zip(chunk, preds, strict=True):
                    self._store(self.cache_key(rows[index]), prediction)
                    results[index] = prediction
        else:
            limiter = anyio.CapacityLimiter(max(1, self.config.concurrency))

            async def one(index: int) -> None:
                async with limiter:
                    results[index] = await self.predict(rows[index])

            async with anyio.create_task_group() as group:
                for index in missing:
                    group.start_soon(one, index)
        return [item for item in results if item is not None]

    def _has_batch(self) -> bool:
        probe = getattr(self.model, "supports_batch", None)
        if callable(probe):
            return bool(probe())
        return callable(getattr(self.model, "apredict_batch", None)) or callable(
            getattr(self.model, "predict_batch", None)
        )

    def _reserve_call(self) -> None:
        if self.config.max_calls is not None and self._calls >= self.config.max_calls:
            raise BudgetExceededError(
                f"model call budget exhausted ({self.config.max_calls} calls)"
            )
        self._calls += 1

    async def _invoke_with_retry(self, instance: Mapping[str, Any]) -> Prediction:
        attempts = self.config.retries + 1
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self._lock_for():
                    self._reserve_call()
                prediction: Prediction | None = None
                with anyio.move_on_after(self.config.timeout_s) as scope:
                    prediction = await self._invoke(instance)
                if prediction is None or scope.cancel_called:
                    raise TimeoutError(f"model call exceeded {self.config.timeout_s}s")
                return prediction
            except BudgetExceededError:
                raise
            except TimeoutError as exc:
                last = exc
            except Exception as exc:
                last = exc
            if attempt + 1 < attempts:
                delay = (0.01 * (2**attempt)) * float(self._rng.uniform(0.5, 1.0))
                await anyio.sleep(delay)
        raise ModelCallError(f"model call failed after {attempts} attempts: {last}") from last

    async def _invoke(self, instance: Mapping[str, Any]) -> Prediction:
        if callable(getattr(self.model, "apredict", None)):
            raw = await self.model.apredict(instance)
        else:
            raw = await anyio.to_thread.run_sync(self.model.predict, dict(instance))
        prediction = coerce_prediction(raw)
        if prediction.probability is None and _supports_proba(self.model):
            extra = await self._invoke_proba(instance)
            prediction = _merge_proba(prediction, extra)
        return prediction

    async def _invoke_proba(self, instance: Mapping[str, Any]) -> Any:
        if callable(getattr(self.model, "apredict_proba", None)):
            return await self.model.apredict_proba(instance)
        return await anyio.to_thread.run_sync(self.model.predict_proba, dict(instance))

    async def _invoke_batch(self, rows: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        if callable(getattr(self.model, "apredict_batch", None)):
            raw_rows = await self.model.apredict_batch(rows)
        else:
            raw_rows = await anyio.to_thread.run_sync(self.model.predict_batch, list(rows))
        return [coerce_prediction(item) for item in raw_rows]


def _supports_proba(model: Any) -> bool:
    probe = getattr(model, "supports_proba", None)
    if callable(probe):
        return bool(probe())
    return callable(getattr(model, "predict_proba", None))


def _merge_proba(prediction: Prediction, raw: Any) -> Prediction:
    if isinstance(raw, Mapping):
        probabilities = {str(key): float(value) for key, value in raw.items()}
        probability = probabilities.get(prediction.label, prediction.probability)
        return prediction.model_copy(
            update={"probabilities": probabilities, "probability": probability}
        )
    return prediction
