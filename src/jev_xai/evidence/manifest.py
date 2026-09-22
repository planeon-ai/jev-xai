"""Merkle manifest over pack members."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_xai.canonical import content_hash
from jev_xai.evidence.schema import SCHEMA_VERSION
from jev_xai.limitations import LIMITATIONS
from jev_xai.version import __version__


def merkle_root(members: dict[str, str]) -> str:
    """Hash of the sorted ``(name, content_hash)`` pairs."""

    rows = [{"name": name, "hash": members[name]} for name in sorted(members)]
    return content_hash(rows)


def build_manifest(
    members: dict[str, str], *, config_hash_value: str | None = None
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "jev_xai_version": __version__,
        "config_hash": config_hash_value,
        "members": [{"name": name, "hash": members[name]} for name in sorted(members)],
        "merkle_root": merkle_root(members),
        "limitations": LIMITATIONS,
    }


def verify_manifest(directory: Path) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = {item["name"]: item["hash"] for item in manifest.get("members", [])}
    if merkle_root(members) != manifest.get("merkle_root"):
        return False
    for name, digest in members.items():
        path = directory / name
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8")
        if name.endswith(".json"):
            actual = content_hash(json.loads(text))
        else:
            actual = content_hash({"text": text})
        if actual != digest:
            return False
    return True
