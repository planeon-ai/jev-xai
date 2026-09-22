"""Redact or externalize decision inputs before they land in a pack."""

from __future__ import annotations

from typing import Any

from jev_xai.canonical import canonical_json, content_hash
from jev_xai.config.schema import EvidenceConfig


def prepare_input(
    instance: dict[str, Any], config: EvidenceConfig
) -> tuple[Any | None, bool, bool]:
    """Return ``(stored_input, reachable, externalized)``."""

    if config.redaction_mode == "hash_only":
        return None, False, False
    payload: dict[str, Any] = dict(instance)
    if config.redaction_mode == "redacted":
        for field in config.redact_fields:
            if field in payload:
                payload[field] = "[REDACTED]"
    encoded = canonical_json(payload).encode("utf-8")
    if len(encoded) > config.max_inline_input_bytes:
        return {"external_hash": content_hash(payload)}, False, True
    return payload, True, False
