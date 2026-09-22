"""Repeat-probe a single input and derive the noise floor used by every explainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jev_xai.config.schema import JevXaiConfig
from jev_xai.model.client import ModelClient
from jev_xai.stability.metrics import sample_stdev
from jev_xai.types import ProbeSummary


class ReproducibilityProbe:
    """Fresh calls, never served from the cache.

    ``reproduction_rate`` is the fraction of repeats whose label matches the
    reference decision. ``noise_floor`` is ``k * probability_sigma``.
    """

    def __init__(self, config: JevXaiConfig) -> None:
        self.config = config

    async def measure(
        self,
        client: ModelClient,
        instance: Mapping[str, Any],
        *,
        reference_label: str,
    ) -> ProbeSummary:
        runs = self.config.reproducibility.repeat_probe_runs
        labels: list[str] = []
        probabilities: list[float] = []
        for _ in range(runs):
            prediction = await client.predict(dict(instance), use_cache=False)
            labels.append(prediction.label)
            if prediction.probability is not None:
                probabilities.append(prediction.probability)
        rate = (sum(1 for label in labels if label == reference_label) / runs) if runs else 1.0
        sigma = sample_stdev(probabilities)
        floor = self.config.reproducibility.noise_floor_sigma_k * sigma
        client.noise_floor = floor
        return ProbeSummary(
            runs=runs,
            reproduction_rate=rate,
            probability_sigma=sigma,
            noise_floor=floor,
            labels=labels,
            probabilities=probabilities,
        )
