"""Thin JEV adapter.

No official JEV SDK is part of this distribution. Until one is published,
``JEVAdapter`` wraps a caller-supplied client that is either callable or
exposes ``predict``. See ``docs/research-landscape.md``.
"""

from __future__ import annotations

from typing import Any

from jev_xai.adapters.callable import CallableAdapter
from jev_xai.errors import JevXaiUsageError


class JEVAdapter(CallableAdapter):
    """Wrap a JEV-like client without importing a vendor SDK."""

    def __init__(
        self,
        client: Any,
        *,
        provider: str = "jev",
        model_name: str = "jev",
        model_version: str = "unspecified",
        artifact_hash: str | None = None,
    ) -> None:
        predict = _bind_predict(client)
        proba = getattr(client, "predict_proba", None)
        async_fn = getattr(client, "apredict", None)
        batch = getattr(client, "predict_batch", None)
        async_batch = getattr(client, "apredict_batch", None)
        extra: dict[str, Any] = {}
        if hasattr(client, "metadata"):
            extra = dict(client.metadata())
        meta: dict[str, Any] = {
            "provider": provider,
            "model_name": model_name,
            "model_version": model_version,
            "artifact_hash": artifact_hash,
        }
        meta.update(extra)
        super().__init__(
            predict,
            metadata=meta,
            proba_fn=proba if callable(proba) else None,
            async_fn=async_fn if callable(async_fn) else None,
            batch_fn=batch if callable(batch) else None,
            async_batch_fn=async_batch if callable(async_batch) else None,
        )


def _bind_predict(client: Any) -> Any:
    if hasattr(client, "predict") and callable(client.predict):
        return client.predict
    if callable(client):
        return client
    raise JevXaiUsageError(
        "JEV client must be callable or expose predict(). "
        "No JEV SDK is bundled; pass the object your harness already calls."
    )


def metadata_of(model: Any) -> dict[str, Any]:
    if hasattr(model, "metadata") and callable(model.metadata):
        return dict(model.metadata())
    return {"provider": "unknown", "model_name": "unknown", "model_version": "unknown"}
