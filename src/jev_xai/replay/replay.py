"""Replay modes. Evidence replay and behavioral reproduction are different claims."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from jev_xai.config.loader import config_hash
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import ReplayMismatchError
from jev_xai.evidence.schema import DecisionRecord
from jev_xai.model.cassette import Cassette
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint
from jev_xai.replay.environment import runtime_fingerprint
from jev_xai.types import CostEnvelope, Prediction


class ReplayResult(BaseModel):
    claim: Literal["evidence_replay", "behavioral_reproduction"]
    mode: str
    matched: bool
    mismatch_reason: str | None = None
    original_label: str
    replay_label: str
    original_probability: float | None = None
    replay_probability: float | None = None
    cost: CostEnvelope = Field(default_factory=CostEnvelope)
    config_hash_matches: bool = True
    note: str = ""


class ReplayEngine:
    def __init__(self, config: JevXaiConfig, *, cassette: Cassette | None = None) -> None:
        self.config = config
        self.cassette = cassette

    async def replay(
        self,
        record: DecisionRecord,
        *,
        model: Any | None = None,
        mode: str | None = None,
    ) -> ReplayResult:
        selected = mode or self.config.replay.mode
        if selected == "exact":
            return self._evidence(record)
        if model is None:
            raise ReplayMismatchError(
                "behavioral reproduction requires a live model",
                reason="model_reinvocation",
            )
        if not record.input_reachable or record.input is None or not isinstance(record.input, dict):
            raise ReplayMismatchError(
                "input evidence is not reachable; only evidence replay is possible",
                reason="input_evidence_reachable",
            )
        return await self._behavioral(record, model)

    def _evidence(self, record: DecisionRecord) -> ReplayResult:
        prediction: Prediction | None = None
        if self.cassette is not None and record.cassette_key:
            prediction = self.cassette.get(record.cassette_key)
        if prediction is None:
            prediction = Prediction(
                label=record.output.label,
                probability=record.output.probability,
                probabilities=record.output.probabilities,
            )
        return ReplayResult(
            claim="evidence_replay",
            mode="exact",
            matched=prediction.label == record.output.label,
            original_label=record.output.label,
            replay_label=prediction.label,
            original_probability=record.output.probability,
            replay_probability=prediction.probability,
            cost=CostEnvelope(),
            note="Evidence replay proves what was recorded. It is not proof the model still decides this way.",
        )

    async def _behavioral(self, record: DecisionRecord, model: Any) -> ReplayResult:
        fingerprint = model_fingerprint(model)
        client = ModelClient(
            model,
            self.config.model,
            fingerprint=fingerprint,
            seed=self.config.seed,
            cassette=self.cassette,
        )
        assert isinstance(record.input, dict)
        prediction = await client.predict(record.input, use_cache=False)
        tolerance = self.config.reproducibility.probability_tolerance
        label_ok = prediction.label == record.output.label
        if record.output.probability is None or prediction.probability is None:
            proba_ok = True
        else:
            proba_ok = abs(prediction.probability - record.output.probability) <= tolerance
        matched = label_ok and proba_ok
        from jev_xai.replay.diff import classify_mismatch

        live_hash = config_hash(self.config)
        recorded_hash = record.explainer_config.get("config_hash")
        reason = classify_mismatch(
            record,
            live_fingerprint=fingerprint,
            live_package_hash=runtime_fingerprint(self.config.seed).package_lock_hash,
            live_config_hash=live_hash,
            matched=matched,
        )
        return ReplayResult(
            claim="behavioral_reproduction",
            mode="current",
            matched=matched,
            mismatch_reason=reason,
            original_label=record.output.label,
            replay_label=prediction.label,
            original_probability=record.output.probability,
            replay_probability=prediction.probability,
            cost=client.cost(),
            config_hash_matches=recorded_hash in (None, live_hash),
            note="Behavioral reproduction re-invokes the model. A mismatch is classified, not hidden.",
        )


def raise_on_mismatch(result: ReplayResult) -> ReplayResult:
    if not result.matched:
        raise ReplayMismatchError(
            f"replay mismatch ({result.mismatch_reason})",
            reason=result.mismatch_reason,
        )
    return result
