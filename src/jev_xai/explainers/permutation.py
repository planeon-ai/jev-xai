"""Permutation importance. Local sensitivity, plus a mean across a dataset.

Importance is the mean drop in P(original label). When the model returns no
probability, importance is omitted and ``label_flip_rate`` is the evidence.
A feature that is not sampled is ``unmeasured``; null importance is not a zero effect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jev_xai.adapters.capabilities import diagnose, require
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import JevXaiUsageError
from jev_xai.explainers.base import Explainer, ExplanationResult
from jev_xai.explainers.context import ExplainContext
from jev_xai.explainers.sampling import draw_alternative, is_perturbable
from jev_xai.model.client import ModelClient
from jev_xai.replay.seeding import generator
from jev_xai.types import CostEnvelope


class PermutationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    importance: float | None = None
    label_flip_rate: float = 0.0
    n_samples: int = 0
    unmeasured: bool = False


class PermutationResult(ExplanationResult):
    explainer: str = "permutation"
    label: str = ""
    rows: list[PermutationRow] = Field(default_factory=list)
    scope: Literal["local", "dataset"] = "local"


class PermutationExplainer(Explainer):
    """Permute one mutable feature at a time. Immutable features are skipped."""

    name = "permutation"

    def __init__(self, config: JevXaiConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed

    async def explain(
        self,
        client: ModelClient,
        instance: Mapping[str, Any],
        context: ExplainContext | None = None,
    ) -> PermutationResult:
        context = context or ExplainContext()
        require(diagnose(client.model, input_reachable=True, context=context), "permutation")
        started = client.cost()
        return await self._explain(client, dict(instance), context, started, self.seed)

    async def explain_dataset(
        self,
        client: ModelClient,
        rows: Sequence[Mapping[str, Any]],
        context: ExplainContext | None = None,
    ) -> PermutationResult:
        """Mean local importance and flip rate across ``rows``. Budget is per row."""

        if not rows:
            raise JevXaiUsageError("explain_dataset requires at least one row")
        context = context or ExplainContext()
        started = client.cost()
        parts: list[PermutationResult] = []
        for index, row in enumerate(rows):
            child = self.seed + index
            parts.append(await self._explain(client, dict(row), context, client.cost(), child))
        return _average(parts, client.cost().delta(started), client.noise_floor)

    async def _explain(
        self,
        client: ModelClient,
        instance: dict[str, Any],
        context: ExplainContext,
        started: CostEnvelope,
        seed: int,
    ) -> PermutationResult:
        require(diagnose(client.model, input_reachable=True, context=context), "permutation")
        original = await client.predict(instance)
        baseline = original.class_probability()
        cfg = self.config.permutation
        rng = generator(seed)
        rows: list[PermutationRow] = []
        for spec in context.features:
            if not spec.mutable:
                continue
            original_value = instance.get(spec.name)
            if not is_perturbable(spec, original_value):
                rows.append(_unmeasured(spec.name))
                continue
            if _remaining(client, started.n_model_calls, cfg.call_budget) < 1:
                rows.append(_unmeasured(spec.name))
                continue
            deltas: list[float] = []
            flips = 0
            used = 0
            for _ in range(cfg.repeats):
                if _remaining(client, started.n_model_calls, cfg.call_budget) < 1:
                    break
                trial = dict(instance)
                trial[spec.name] = draw_alternative(spec, original_value, rng)
                prediction = await client.predict(trial)
                used += 1
                if prediction.label != original.label:
                    flips += 1
                after = prediction.class_probability(original.label)
                if baseline is not None and after is not None:
                    deltas.append(baseline - after)
            if used == 0:
                rows.append(_unmeasured(spec.name))
                continue
            importance = sum(deltas) / len(deltas) if baseline is not None and deltas else None
            rows.append(
                PermutationRow(
                    feature=spec.name,
                    importance=importance,
                    label_flip_rate=flips / used,
                    n_samples=used,
                )
            )
        return PermutationResult(
            label=original.label,
            rows=rows,
            scope="local",
            noise_floor=client.noise_floor,
            cost=client.cost().delta(started),
        )


def _average(
    parts: list[PermutationResult],
    cost: CostEnvelope,
    noise_floor: float | None,
) -> PermutationResult:
    names = [row.feature for row in parts[0].rows]
    merged: list[PermutationRow] = []
    for name in names:
        importances: list[float] = []
        flips: list[float] = []
        count = 0
        measured = False
        for part in parts:
            found = next(item for item in part.rows if item.feature == name)
            if not found.unmeasured:
                measured = True
            if found.importance is not None:
                importances.append(found.importance)
            flips.append(found.label_flip_rate)
            count += found.n_samples
        importance = sum(importances) / len(importances) if importances else None
        merged.append(
            PermutationRow(
                feature=name,
                importance=importance,
                label_flip_rate=sum(flips) / len(flips) if flips else 0.0,
                n_samples=count,
                unmeasured=not measured,
            )
        )
    labels = {part.label for part in parts}
    label = parts[0].label if len(labels) == 1 else "mixed"
    return PermutationResult(
        label=label,
        rows=merged,
        scope="dataset",
        noise_floor=noise_floor,
        cost=cost,
    )


def _unmeasured(feature: str) -> PermutationRow:
    """No draw was made, so a zero importance would not be evidence."""

    return PermutationRow(feature=feature, importance=None, n_samples=0, unmeasured=True)


def _remaining(client: ModelClient, started_calls: int, budget: int) -> int:
    used = client.cost().n_model_calls - started_calls
    return max(0, budget - used)
