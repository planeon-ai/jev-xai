"""Model identity used as the cache key prefix."""

from __future__ import annotations

from typing import Any

from jev_xai.adapters.jev import metadata_of
from jev_xai.canonical import content_hash


def model_fingerprint(model: Any) -> str:
    meta = metadata_of(model)
    explicit = meta.get("fingerprint")
    if explicit:
        return str(explicit)
    return content_hash(
        {
            "provider": meta.get("provider"),
            "model_name": meta.get("model_name"),
            "model_version": meta.get("model_version"),
            "artifact_hash": meta.get("artifact_hash"),
        }
    )
