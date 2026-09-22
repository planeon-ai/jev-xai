"""Cross-version comparison and mismatch attribution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from jev_xai.config.loader import config_hash
from jev_xai.config.schema import JevXaiConfig
from jev_xai.evidence.schema import DecisionRecord
from jev_xai.model.fingerprint import model_fingerprint
from jev_xai.replay.replay import ReplayEngine, ReplayResult

NONSTATIONARY_KEYS = {"timestamp", "time", "now", "date", "request_time", "as_of"}


def classify_mismatch(
    record: DecisionRecord,
    *,
    live_fingerprint: str,
    live_package_hash: str,
    live_config_hash: str,
    matched: bool,
) -> str | None:
    """Name the most likely reason a live replay disagrees with the record."""

    if matched:
        return None
    if record.model.fingerprint != live_fingerprint:
        return "model_version_changed"
    if record.runtime.package_lock_hash != live_package_hash:
        return "environment_changed"
    recorded = record.explainer_config.get("config_hash")
    if isinstance(recorded, str) and recorded != live_config_hash:
        return "config_changed"
    if isinstance(record.input, dict) and any(
        str(key).lower() in NONSTATIONARY_KEYS for key in record.input
    ):
        return "input_nonstationary"
    return "model_nondeterministic"


class DiffRow(BaseModel):
    decision_id: str
    original_label: str
    replay_label: str
    flipped: bool
    mismatch_reason: str | None = None
    config_hash_matches: bool = True
    original_probability: float | None = None
    replay_probability: float | None = None


class DiffReport(BaseModel):
    n_records: int
    n_flipped: int
    rows: list[DiffRow] = Field(default_factory=list)
    config_hash_warning: str | None = None
    from_fingerprint: str | None = None
    to_fingerprint: str | None = None


async def cross_version_diff(
    records: list[DecisionRecord],
    model: Any,
    config: JevXaiConfig,
) -> DiffReport:
    """Re-run stored records against ``model`` and report decision flips."""

    engine = ReplayEngine(config)
    live_hash = config_hash(config)
    rows: list[DiffRow] = []
    warnings: list[str] = []
    for record in records:
        recorded_hash = record.explainer_config.get("config_hash")
        if isinstance(recorded_hash, str) and recorded_hash != live_hash:
            warnings.append(record.decision_id)
        if not record.input_reachable or not isinstance(record.input, dict):
            rows.append(
                DiffRow(
                    decision_id=record.decision_id,
                    original_label=record.output.label,
                    replay_label=record.output.label,
                    flipped=False,
                    mismatch_reason="input_evidence_reachable",
                    config_hash_matches=recorded_hash in (None, live_hash),
                )
            )
            continue
        result: ReplayResult = await engine.replay(record, model=model, mode="current")
        rows.append(
            DiffRow(
                decision_id=record.decision_id,
                original_label=result.original_label,
                replay_label=result.replay_label,
                flipped=result.original_label != result.replay_label,
                mismatch_reason=result.mismatch_reason,
                config_hash_matches=result.config_hash_matches,
                original_probability=result.original_probability,
                replay_probability=result.replay_probability,
            )
        )
    warning = None
    if warnings:
        warning = (
            "config_hash does not match the recorded explainer config for "
            + ", ".join(warnings)
            + "; decision diffs are still reported, but they are not an isolated model comparison"
        )
    fingerprints = {record.model.fingerprint for record in records}
    from_fp = next(iter(fingerprints)) if len(fingerprints) == 1 else None
    return DiffReport(
        n_records=len(records),
        n_flipped=sum(1 for row in rows if row.flipped),
        rows=rows,
        config_hash_warning=warning,
        from_fingerprint=from_fp,
        to_fingerprint=model_fingerprint(model) if records else None,
    )
