"""Shared result types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """One decision from a model."""

    label: str
    probability: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)

    def class_probability(self, label: str | None = None) -> float | None:
        key = self.label if label is None else label
        if key in self.probabilities:
            return self.probabilities[key]
        if label is None:
            return self.probability
        return None


class CostEnvelope(BaseModel):
    """Query cost attached to every result."""

    n_model_calls: int = 0
    cache_hits: int = 0
    wall_ms: float = 0.0

    def delta(self, earlier: CostEnvelope) -> CostEnvelope:
        return CostEnvelope(
            n_model_calls=self.n_model_calls - earlier.n_model_calls,
            cache_hits=self.cache_hits - earlier.cache_hits,
            wall_ms=self.wall_ms - earlier.wall_ms,
        )


class ProbeSummary(BaseModel):
    """Measured reproducibility of one input."""

    runs: int
    reproduction_rate: float
    probability_sigma: float
    noise_floor: float
    labels: list[str] = Field(default_factory=list)
    probabilities: list[float] = Field(default_factory=list)


def coerce_prediction(raw: Any) -> Prediction:
    """Normalize adapter return values into a :class:`Prediction`."""

    from jev_xai.errors import ModelCallError

    if isinstance(raw, Prediction):
        return raw
    if isinstance(raw, str):
        return Prediction(label=raw)
    if isinstance(raw, tuple) and raw:
        label = str(raw[0])
        probability = float(raw[1]) if len(raw) > 1 and raw[1] is not None else None
        return Prediction(label=label, probability=probability)
    if isinstance(raw, dict):
        if "label" not in raw:
            raise ModelCallError("prediction dict is missing 'label'")
        probabilities = {
            str(key): float(val) for key, val in dict(raw.get("probabilities") or {}).items()
        }
        probability = raw.get("probability")
        prob_value = float(probability) if probability is not None else None
        label = str(raw["label"])
        if prob_value is None and label in probabilities:
            prob_value = probabilities[label]
        return Prediction(label=label, probability=prob_value, probabilities=probabilities)
    raise ModelCallError(f"unrecognized prediction type {type(raw).__name__}")
