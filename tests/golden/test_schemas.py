"""Published JSON Schemas stay aligned with the pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

from jev_xai.evidence.schema import DecisionRecord, PackManifest

ROOT = Path(__file__).resolve().parents[2]


def test_published_schemas_match_the_models() -> None:
    decision = json.loads((ROOT / "schemas" / "decision_record.v1.json").read_text(encoding="utf-8"))
    pack = json.loads((ROOT / "schemas" / "evidence_pack.v1.json").read_text(encoding="utf-8"))
    assert decision == DecisionRecord.model_json_schema()
    assert pack == PackManifest.model_json_schema()
