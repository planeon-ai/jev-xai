"""Nested, hashable configuration for every jev-xai engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelClientConfig(_Strict):
    concurrency: int = 8
    batch_size: int = 16
    max_calls: int | None = None
    timeout_s: float = 30.0
    retries: int = 2
    cache_mode: Literal["off", "memory", "cassette"] = "memory"


class MaskingPolicy(_Strict):
    structured: Literal["drop_key", "null", "neutral", "background"] = "neutral"
    text: Literal["deletion", "mask_token", "neutral_token"] = "mask_token"
    neutral_numeric: float = 0.0
    neutral_text: str = ""
    mask_token: str = "[MASK]"


class AblationConfig(_Strict):
    masking: MaskingPolicy = Field(default_factory=MaskingPolicy)
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)
    text_span_granularity: Literal["token", "sentence", "char_chunk"] = "token"
    max_spans: int = 32
    text_chunk_chars: int = 32


class CounterfactualConfig(_Strict):
    max_features_changed: int = 3
    grid_quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    max_candidates: int = 5
    distance_metric: Literal["normalized_l1", "hamming"] = "normalized_l1"
    immutable_features: list[str] = Field(default_factory=list)
    allowed_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    categorical_values: dict[str, list[Any]] = Field(default_factory=dict)
    numeric_grid: dict[str, list[float]] = Field(default_factory=dict)
    call_budget: int = 200


class ScoreWeights(_Strict):
    """Weights for ``stability_score_v1``. They must sum to 1.

    ``stability_score_v1`` =
    ``rank_correlation`` * ``(rho + 1) / 2``
    + ``feature_overlap`` * Jaccard top-k
    + ``attribution_stability`` * ``1 / (1 + mean variance of deltas)``
    + ``counterfactual_consistency`` * modal-set agreement.
    """

    rank_correlation: float = 0.4
    feature_overlap: float = 0.3
    attribution_stability: float = 0.2
    counterfactual_consistency: float = 0.1

    @model_validator(mode="after")
    def _sum_to_one(self) -> ScoreWeights:
        total = (
            self.rank_correlation
            + self.feature_overlap
            + self.attribution_stability
            + self.counterfactual_consistency
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("stability score weights must sum to 1")
        return self


class StabilityConfig(_Strict):
    runs: int = 20
    top_k: int = 5
    rank_metric: Literal["spearman", "kendall"] = "spearman"
    failure_tolerance: float = 0.2
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)


class ReproducibilityConfig(_Strict):
    repeat_probe_runs: int = 5
    noise_floor_sigma_k: float = 2.0
    probability_tolerance: float = 1e-6
    min_reproduction_rate: float = 0.9
    on_below_noise_floor: Literal["flag", "drop", "error"] = "flag"


def _default_formats() -> list[Literal["json", "md", "html", "table"]]:
    return ["json", "md", "html"]


class AnchorConfig(_Strict):
    """Greedy anchor search. ``precision`` is the stopping threshold."""

    precision: float = 0.95
    max_size: int = 4
    samples: int = 16
    coverage_samples: int = 64
    numeric_band: float = 0.2
    call_budget: int = 200


class PermutationConfig(_Strict):
    """Local permutation importance. ``call_budget`` applies per instance."""

    repeats: int = 8
    call_budget: int = 200


class ReplayConfig(_Strict):
    mode: Literal["exact", "current", "cross", "counterfactual"] = "exact"


class EvidenceConfig(_Strict):
    redaction_mode: Literal["full", "redacted", "hash_only"] = "full"
    redact_fields: list[str] = Field(default_factory=list)
    max_inline_input_bytes: int = 64_000
    formats: list[Literal["json", "md", "html", "table"]] = Field(default_factory=_default_formats)


class JevXaiConfig(_Strict):
    """Resolved configuration. Hash this object; do not pass loose kwargs."""

    seed: int = 42
    threshold: float | None = None
    policy_version: str | None = None
    model: ModelClientConfig = Field(default_factory=ModelClientConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    counterfactual: CounterfactualConfig = Field(default_factory=CounterfactualConfig)
    anchors: AnchorConfig = Field(default_factory=AnchorConfig)
    permutation: PermutationConfig = Field(default_factory=PermutationConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
