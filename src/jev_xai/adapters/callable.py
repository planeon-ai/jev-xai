"""Wrap any callable as a DecisionModel."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from jev_xai.canonical import content_hash


class CallableAdapter:
    """Adapter for a plain function, sync or async.

    This is the wrap pattern a multi-agent harness uses when the decision is
    already a Python callable.
    """

    def __init__(
        self,
        predict_fn: Callable[[Mapping[str, Any]], Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        proba_fn: Callable[[Mapping[str, Any]], Mapping[str, float]] | None = None,
        async_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        batch_fn: Callable[[Sequence[Mapping[str, Any]]], Sequence[Any]] | None = None,
        async_batch_fn: Callable[[Sequence[Mapping[str, Any]]], Any] | None = None,
    ) -> None:
        self._predict = predict_fn
        self._proba = proba_fn
        self._async = async_fn
        self._batch = batch_fn
        self._async_batch = async_batch_fn
        self._batch_enabled = batch_fn is not None or async_batch_fn is not None
        meta = dict(metadata or {})
        meta.setdefault("provider", "callable")
        meta.setdefault("model_name", getattr(predict_fn, "__name__", "callable"))
        meta.setdefault("model_version", "unspecified")
        if "fingerprint" not in meta:
            meta["fingerprint"] = content_hash(
                {
                    "provider": meta.get("provider"),
                    "model_name": meta.get("model_name"),
                    "model_version": meta.get("model_version"),
                    "artifact_hash": meta.get("artifact_hash"),
                }
            )
        self._metadata = meta

    def predict(self, x: Mapping[str, Any]) -> Any:
        return self._predict(x)

    def predict_proba(self, x: Mapping[str, Any]) -> Mapping[str, float]:
        if self._proba is None:
            from jev_xai.errors import CapabilityError

            raise CapabilityError(
                "calibrated_probabilities",
                "pass proba_fn to CallableAdapter or return probability from predict",
            )
        return self._proba(x)

    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    def supports_proba(self) -> bool:
        return self._proba is not None

    def supports_batch(self) -> bool:
        return self._batch_enabled

    async def apredict(self, x: Mapping[str, Any]) -> Any:
        if self._async is not None:
            return await self._async(x)
        return self._predict(x)

    def predict_batch(self, rows: Sequence[Mapping[str, Any]]) -> Sequence[Any]:
        if self._batch is None:
            return [self._predict(row) for row in rows]
        return self._batch(rows)

    async def apredict_batch(self, rows: Sequence[Mapping[str, Any]]) -> Sequence[Any]:
        if self._async_batch is not None:
            result: Sequence[Any] = await self._async_batch(rows)
            return result
        return self.predict_batch(rows)
