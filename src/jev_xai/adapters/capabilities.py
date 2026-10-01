"""Capability tiers: what the host can prove, and which explainers that unlocks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from jev_xai.explainers.context import ExplainContext


class ExplainerAvailability(BaseModel):
    name: str
    enabled: bool
    missing: list[str] = Field(default_factory=list)
    fix: str = ""
    note: str = ""


class Diagnosis(BaseModel):
    tier: int
    tier_name: str
    prerequisites: dict[str, bool]
    explainers: list[ExplainerAvailability]
    notes: list[str] = Field(default_factory=list)


_TIER_NAMES = {
    0: "record_only",
    1: "behavioral",
    2: "graded",
    3: "longitudinal",
}

_FIXES = {
    "model_reinvocation": "pass a live DecisionModel; a JSONL import alone cannot re-invoke the model",
    "input_evidence_reachable": "persist the decision input, or set redaction_mode off hash_only",
    "feature_spec": "pass a FeatureSpec so ablation can mask individual fields",
    "calibrated_probabilities": "return probability from predict, or implement predict_proba",
    "model_fingerprint": "include fingerprint or model_version in metadata()",
    "record_corpus": "record a directory of decisions before cross-version replay",
}


def has_probabilities(model: Any) -> bool:
    probe = getattr(model, "supports_proba", None)
    if callable(probe):
        return bool(probe())
    return callable(getattr(model, "predict_proba", None)) or callable(
        getattr(model, "apredict_proba", None)
    )


def has_fingerprint(model: Any | None) -> bool:
    if model is None or not callable(getattr(model, "metadata", None)):
        return False
    meta = dict(model.metadata())
    return bool(meta.get("fingerprint") or meta.get("model_version"))


def has_reinvocation(model: Any | None) -> bool:
    if model is None:
        return False
    return callable(getattr(model, "predict", None)) or callable(model)


def diagnose(
    model: Any | None = None,
    *,
    input_reachable: bool = True,
    context: ExplainContext | None = None,
    has_corpus: bool = False,
    source_only: bool = False,
) -> Diagnosis:
    """Probe what this host can support. The result is reported, not guessed."""

    reinvocation = has_reinvocation(model) and not source_only
    probabilities = has_probabilities(model) and reinvocation
    fingerprint = has_fingerprint(model)
    feature_spec = bool(context and context.features)
    prerequisites = {
        "model_reinvocation": reinvocation,
        "input_evidence_reachable": input_reachable,
        "feature_spec": feature_spec,
        "calibrated_probabilities": probabilities,
        "model_fingerprint": fingerprint,
        "record_corpus": has_corpus,
    }
    tier = 0
    if reinvocation and input_reachable:
        tier = 1
        if probabilities:
            tier = 2
            if fingerprint and has_corpus:
                tier = 3
    notes: list[str] = []
    if reinvocation and input_reachable and not feature_spec:
        notes.append(
            "no FeatureSpec: ablation is limited to a single whole-input mask; "
            "anchors and permutation stay off"
        )
    if reinvocation and input_reachable and not probabilities:
        notes.append("no probabilities: ablation reports label flips and omits delta-P")

    def availability(
        name: str, enabled: bool, missing: list[str], note: str = ""
    ) -> ExplainerAvailability:
        fix = " ".join(_FIXES[item] for item in missing)
        return ExplainerAvailability(
            name=name, enabled=enabled, missing=missing, fix=fix, note=note
        )

    behavioral_missing = [
        key
        for key, present in (
            ("model_reinvocation", reinvocation),
            ("input_evidence_reachable", input_reachable),
        )
        if not present
    ]
    graded_missing = list(behavioral_missing)
    if not probabilities:
        graded_missing.append("calibrated_probabilities")
    longitudinal_missing = list(graded_missing)
    if not fingerprint:
        longitudinal_missing.append("model_fingerprint")
    if not has_corpus:
        longitudinal_missing.append("record_corpus")

    spec_missing = list(behavioral_missing)
    if not feature_spec:
        spec_missing.append("feature_spec")
    explainers = [
        availability("record", True, []),
        availability(
            "evidence_replay",
            True,
            [],
            note="replays the cassette; this is not proof the model still decides the same way",
        ),
        availability("ablation", tier >= 1, behavioral_missing),
        availability("counterfactual", tier >= 1, behavioral_missing),
        availability(
            "anchors",
            tier >= 1 and feature_spec,
            spec_missing,
            note="precision is estimated with model samples; coverage does not call the model",
        ),
        availability(
            "permutation",
            tier >= 1 and feature_spec,
            spec_missing,
            note="importance uses class probability when the model returns one",
        ),
        availability("stability", tier >= 2, graded_missing),
        availability("cross_version_replay", tier >= 3, longitudinal_missing),
    ]
    return Diagnosis(
        tier=tier,
        tier_name=_TIER_NAMES[tier],
        prerequisites=prerequisites,
        explainers=explainers,
        notes=notes,
    )


def require(diagnosis: Diagnosis, explainer: str) -> None:
    """Raise CapabilityError naming the first missing prerequisite."""

    from jev_xai.errors import CapabilityError

    for item in diagnosis.explainers:
        if item.name == explainer and not item.enabled:
            missing = item.missing[0] if item.missing else explainer
            raise CapabilityError(
                missing, item.fix or _FIXES.get(missing, "see docs/prerequisites.md")
            )
