"""JSONL ingestion from a host harness."""

from __future__ import annotations

from pathlib import Path

from jev_xai.adapters.capabilities import diagnose
from jev_xai.sources.jsonl import JsonlSource


def test_jsonl_import_is_record_only(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        '{"input": {"prompt": "hello"}, "label": "SAFE", "probability": 0.9, '
        '"model_name": "guard", "trace_id": "tr", "agent_id": "router", "step_index": 2}\n',
        encoding="utf-8",
    )
    records = JsonlSource(path).load()
    assert len(records) == 1
    assert records[0].output.label == "SAFE"
    assert records[0].trace.agent_id == "router"
    assert records[0].input_reachable
    diagnosis = diagnose(None, input_reachable=True, source_only=True, has_corpus=True)
    assert diagnosis.tier == 0
    missing = next(item for item in diagnosis.explainers if item.name == "ablation")
    assert "model_reinvocation" in missing.missing
