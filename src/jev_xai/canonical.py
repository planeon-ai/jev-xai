"""Canonical JSON and content hashing.

Float policy: each finite float is emitted with ``format(value, ".17g")``,
which is stable across CPython versions for the same IEEE value. Strings are
Unicode NFC. Object keys are sorted. There is no insignificant whitespace.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any


def _emit(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats cannot be canonicalized")
        return format(value, ".17g")
    if isinstance(value, str):
        return json.dumps(unicodedata.normalize("NFC", value), ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_emit(item) for item in value) + "]"
    if isinstance(value, dict):
        items = sorted(
            (unicodedata.normalize("NFC", str(key)), item) for key, item in value.items()
        )
        body = ",".join(_emit(key) + ":" + _emit(item) for key, item in items)
        return "{" + body + "}"
    if hasattr(value, "model_dump"):
        dumped: Any = value.model_dump(mode="json")
        return _emit(dumped)
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON form of ``value``."""

    return _emit(value)


def blake2b_hex(data: bytes, *, digest_size: int = 32) -> str:
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


def content_hash(value: Any) -> str:
    """blake2b over the UTF-8 canonical JSON of ``value``."""

    return blake2b_hex(canonical_json(value).encode("utf-8"))
