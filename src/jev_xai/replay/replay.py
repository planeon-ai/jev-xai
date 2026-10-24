"""Replay modes. Evidence replay and behavioral reproduction are different claims."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from jev_xai.config.loader import config_hash
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import JevXaiUsageError, ReplayMismatchError
from jev_xai.evidence.schema import DecisionRecord
from jev_xai.evidence.store import EvidenceStore
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
    def __init__(
        self,
        config: JevXaiConfig,
        *,
        cassette: Cassette | None = None,
        store: EvidenceStore | None = None,
    ) -> None:
        self.config = config
        self.cassette = cassette
        self.store = store

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
        if selected == "counterfactual":
            raise JevXaiUsageError(
                "counterfactual is not a replay mode. "
                "Confirmation is replay_confirmed on a counterfactual candidate."
            )
        if selected not in {"current", "cross"}:
            raise JevXaiUsageError("replay mode must be exact, current, or cross")
        if model is None:
            raise ReplayMismatchError(
                "behavioral reproduction requires a live model",
                reason="model_reinvocation",
            )
        instance = self._resolve_input(record)
        return await self._behavioral(record, model, instance, mode=selected)

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

    def _resolve_input(self, record: DecisionRecord) -> dict[str, Any]:
        if record.input_externalized:
            pointer = record.input
            if not isinstance(pointer, dict) or "external_hash" not in pointer:
                raise ReplayMismatchError(
                    "externalized record is missing external_hash",
                    reason="input_evidence_reachable",
                )
            if self.store is None:
                raise ReplayMismatchError(
                    "externalized input requires the evidence store used at record time",
                    reason="input_evidence_reachable",
                )
            try:
                loaded = self.store.get_object(str(pointer["external_hash"]))
            except FileNotFoundError as exc:
                raise ReplayMismatchError(
                    "externalized input is not in the evidence store",
                    reason="input_evidence_reachable",
                ) from exc
            if not isinstance(loaded, dict):
                raise ReplayMismatchError(
                    "externalized input is not a JSON object",
                    reason="input_evidence_reachable",
                )
            return loaded
        if not record.input_reachable or not isinstance(record.input, dict):
            raise ReplayMismatchError(
                "input evidence is not reachable; only evidence replay is possible",
                reason="input_evidence_reachable",
            )
        return record.input

    async def _behavioral(
        self, record: DecisionRecord, model: Any, instance: dict[str, Any], *, mode: str
    ) -> ReplayResult:
        fingerprint = model_fingerprint(model)
        client = ModelClient(
            model,
            self.config.model,
            fingerprint=fingerprint,
            seed=self.config.seed,
            cassette=self.cassette,
        )
        prediction = await client.predict(instance, use_cache=False)
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
            resolved_input=instance,
        )
        if mode == "cross":
            note = (
                "Cross-version replay re-invokes this record on the supplied model. "
                "A corpus report is cross_version_diff."
            )
        else:
            note = "Behavioral reproduction re-invokes the model. A mismatch is classified, not hidden."
        return ReplayResult(
            claim="behavioral_reproduction",
            mode=mode,
            matched=matched,
            mismatch_reason=reason,
            original_label=record.output.label,
            replay_label=prediction.label,
            original_probability=record.output.probability,
            replay_probability=prediction.probability,
            cost=client.cost(),
            config_hash_matches=recorded_hash in (None, live_hash),
            note=note,
        )


def raise_on_mismatch(result: ReplayResult) -> ReplayResult:
    if not result.matched:
        raise ReplayMismatchError(
            f"replay mismatch ({result.mismatch_reason})",
            reason=result.mismatch_reason,
        )
    return result
