"""Content-addressed evidence store. Writes are atomic (temp file, then rename)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_xai.canonical import canonical_json, content_hash
from jev_xai.evidence.manifest import build_manifest, verify_manifest
from jev_xai.evidence.schema import DecisionRecord, read_record


class EvidenceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects = root / "objects"
        self.packs = root / "packs"
        self.index = root / "index.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)

    def put_object(self, payload: Any) -> str:
        encoded = canonical_json(payload)
        key = content_hash(payload)
        path = self.objects / key[:2] / key
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
        return key

    def get_object(self, key: str) -> Any:
        path = self.objects / key[:2] / key
        return json.loads(path.read_text(encoding="utf-8"))

    def append_index(self, decision_id: str, object_key: str) -> None:
        line = canonical_json({"decision_id": decision_id, "object": object_key})
        with self.index.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_pack(self, decision_id: str, members: dict[str, Any]) -> Path:
        """Write named JSON members and a manifest with a Merkle root."""

        directory = self.packs / decision_id
        directory.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for name, payload in members.items():
            text = payload if isinstance(payload, str) else canonical_json(payload)
            path = directory / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
            hashes[name] = content_hash(
                json.loads(text) if name.endswith(".json") else {"text": text}
            )
        manifest = build_manifest(hashes)
        manifest_path = directory / "manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(canonical_json(manifest), encoding="utf-8")
        temporary.replace(manifest_path)
        return directory

    def read_record(self, path: Path) -> DecisionRecord:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return read_record(payload)


def verify_pack(directory: Path) -> bool:
    return verify_manifest(directory)
