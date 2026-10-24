"""Versioned decision record. ``schema_version`` 1.x is readable."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jev_xai.errors import SchemaVersionError
from jev_xai.types import CostEnvelope, ProbeSummary

SCHEMA_VERSION = "1.0.0"


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ModelInfo(_Model):
    provider: str = "unknown"
    model_name: str = "unknown"
    model_version: str = "unknown"
    artifact_hash: str | None = None
    fingerprint: str


class OutputInfo(_Model):
    label: str
    probability: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class RuntimeInfo(_Model):
    python_version: str
    package_lock_hash: str
    seed: int
    container_digest: str | None = None
    hardware: str | None = None


class TraceInfo(_Model):
    trace_id: str | None = None
    span_id: str | None = None
    agent_id: str | None = None
    step_index: int | None = None
    parent_decision_id: str | None = None


class DecisionRecord(_Model):
    """One replayable decision.

    ``input`` is null when the pack was written hash-only. An oversized payload
    is ``{"external_hash": ...}``; it is reachable when that object was written
    to an evidence store. ``input_hash`` is always of the pre-redaction input.
    """

    schema_version: str = SCHEMA_VERSION
    decision_id: str
    timestamp: str
    model: ModelInfo
    input: Any | None = None
    input_hash: str
    input_reachable: bool = True
    input_externalized: bool = False
    output: OutputInfo
    threshold: float | None = None
    policy_version: str | None = None
    runtime: RuntimeInfo
    explainer_config: dict[str, Any] = Field(default_factory=dict)
    explanations: dict[str, Any] = Field(default_factory=dict)
    cost: CostEnvelope = Field(default_factory=CostEnvelope)
    trace: TraceInfo = Field(default_factory=TraceInfo)
    cassette_key: str | None = None
    reproducibility: ProbeSummary | None = None


class PackManifest(_Model):
    """Merkle manifest stored beside an audit pack."""

    schema_version: str
    jev_xai_version: str
    config_hash: str | None = None
    members: list[dict[str, str]]
    merkle_root: str
    limitations: str


def assert_compatible(version: str) -> None:
    """Accept older 1.x records. Reject any other major version."""

    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts) or parts[0] != "1":
        raise SchemaVersionError(
            f"unsupported schema_version {version!r}; this release reads 1.x records"
        )


def read_record(payload: dict[str, Any] | DecisionRecord) -> DecisionRecord:
    if isinstance(payload, DecisionRecord):
        assert_compatible(payload.schema_version)
        return payload
    version = str(payload.get("schema_version", ""))
    assert_compatible(version)
    return DecisionRecord.model_validate(payload)
