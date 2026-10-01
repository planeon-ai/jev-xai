"""Redact or externalize decision inputs before they land in a pack."""

from __future__ import annotations

from typing import Any

from jev_xai.canonical import canonical_json, content_hash
from jev_xai.config.schema import EvidenceConfig
from jev_xai.errors import JevXaiError


def prepare_input(
    instance: dict[str, Any],
    config: EvidenceConfig,
    *,
    store: Any | None = None,
) -> tuple[Any | None, bool, bool]:
    """Return ``(stored_input, reachable, externalized)``.

    Oversized payloads stay reachable when ``store`` writes the body. The record
    then holds ``{"external_hash": ...}`` and replay loads that object back.
    """

    if config.redaction_mode == "hash_only":
        return None, False, False
    payload: dict[str, Any] = dict(instance)
    if config.redaction_mode == "redacted":
        for field in config.redact_fields:
            if field in payload:
                payload[field] = "[REDACTED]"
    encoded = canonical_json(payload).encode("utf-8")
    if len(encoded) > config.max_inline_input_bytes:
        digest = content_hash(payload)
        if store is not None:
            key = store.put_object(payload)
            if key != digest:
                raise JevXaiError("evidence store key did not match the externalized input hash")
            return {"external_hash": digest}, True, True
        return {"external_hash": digest}, False, True
    return payload, True, False
