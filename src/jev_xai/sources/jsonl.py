"""JSONL importer for decision logs from an existing observability harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev_xai.canonical import content_hash
from jev_xai.evidence.schema import (
    SCHEMA_VERSION,
    DecisionRecord,
    ModelInfo,
    OutputInfo,
    RuntimeInfo,
    TraceInfo,
)
from jev_xai.sources.base import RecordSource


class JsonlSource(RecordSource):
    """Map one JSON object per line onto :class:`DecisionRecord`.

    Expected keys: ``input``, ``label`` or ``output.label``, optional
    ``probability``, ``model_name``, ``model_version``, ``provider``,
    ``policy_version``, and trace fields. Imported records are tier 0 unless
    the caller later attaches a live model.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        super().__init__(_read(self._path))

    @property
    def path(self) -> Path:
        return self._path


def _read(path: Path) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(_convert(payload))
    return records


def _convert(payload: dict[str, Any]) -> DecisionRecord:
    raw_input = payload.get("input")
    raw_output = payload.get("output")
    output: dict[str, Any] = raw_output if isinstance(raw_output, dict) else {}
    label = str(payload.get("label") or output.get("label") or "UNKNOWN")
    probability = payload.get("probability", output.get("probability"))
    model = ModelInfo(
        provider=str(payload.get("provider") or "imported"),
        model_name=str(payload.get("model_name") or "imported"),
        model_version=str(payload.get("model_version") or "unspecified"),
        artifact_hash=payload.get("artifact_hash"),
        fingerprint=str(
            payload.get("fingerprint")
            or content_hash(
                {
                    "provider": payload.get("provider"),
                    "model_name": payload.get("model_name"),
                    "model_version": payload.get("model_version"),
                }
            )
        ),
    )
    trace = TraceInfo(
        trace_id=payload.get("trace_id"),
        span_id=payload.get("span_id"),
        agent_id=payload.get("agent_id"),
        step_index=payload.get("step_index"),
        parent_decision_id=payload.get("parent_decision_id"),
    )
    return DecisionRecord(
        schema_version=SCHEMA_VERSION,
        decision_id=str(payload.get("decision_id") or content_hash(payload)[:32]),
        timestamp=str(payload.get("timestamp") or "1970-01-01T00:00:00Z"),
        model=model,
        input=raw_input,
        input_hash=str(payload.get("input_hash") or content_hash(raw_input)),
        output=OutputInfo(
            label=label,
            probability=float(probability) if probability is not None else None,
        ),
        threshold=payload.get("threshold"),
        policy_version=payload.get("policy_version"),
        runtime=RuntimeInfo(
            python_version="imported",
            package_lock_hash="imported",
            seed=int(payload.get("seed") or 0),
        ),
        explainer_config={"source": "jsonl", "config_hash": None},
        trace=trace,
        input_reachable=raw_input is not None,
    )
