"""Greedy coordinate descent over a quantile grid, then a sparsity polish."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from jev_xai.adapters.capabilities import diagnose, require
from jev_xai.config.schema import CounterfactualConfig, JevXaiConfig
from jev_xai.explainers.base import Explainer, ExplanationResult
from jev_xai.explainers.context import ExplainContext, FeatureSpec
from jev_xai.model.client import ModelClient
from jev_xai.replay.seeding import generator
from jev_xai.types import Prediction


class CounterfactualChange(ExplanationResult):
    explainer: str = "counterfactual"
    feature: str
    before: Any = None
    after: Any = None


class CounterfactualCandidate(ExplanationResult):
    explainer: str = "counterfactual"
    changes: list[CounterfactualChange] = Field(default_factory=list)
    label: str
    probability: float | None = None
    distance: float
    sparsity: int
    margin: float | None = None
    flipped: bool
    below_noise_floor: bool = False
    replay_confirmed: bool = False


class CounterfactualResult(ExplanationResult):
    explainer: str = "counterfactual"
    original_label: str
    desired_label: str | None = None
    candidates: list[CounterfactualCandidate] = Field(default_factory=list)
    success_rate: float = 0.0
    algorithm: str = "greedy_coordinate_descent"


class CounterfactualExplainer(Explainer):
    """Minimal sparse search under user-supplied ranges. No density model."""

    name = "counterfactual"

    def __init__(self, config: JevXaiConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed

    async def explain(
        self,
        client: ModelClient,
        instance: Mapping[str, Any],
        context: ExplainContext | None = None,
    ) -> CounterfactualResult:
        context = context or ExplainContext()
        require(diagnose(client.model, input_reachable=True, context=context), "counterfactual")
        started = client.cost()
        budget = self.config.counterfactual.call_budget
        original_input = dict(instance)
        original = await client.predict(original_input)
        desired = context.target_label
        rng = generator(self.seed)
        seeds = [
            int(value)
            for value in rng.integers(0, 2**31 - 1, size=self.config.counterfactual.max_candidates)
        ]
        candidates: list[CounterfactualCandidate] = []
        for offset, child in enumerate(seeds):
            if client.cost().n_model_calls - started.n_model_calls >= budget:
                break
            found = await self._search(
                client,
                original_input,
                original,
                context,
                desired,
                seed=child,
                calls_at_start=started.n_model_calls,
            )
            if found is not None:
                candidates.append(found)
            if offset + 1 >= self.config.counterfactual.max_candidates:
                break
        unique = _dedupe(candidates)
        successes = sum(1 for item in unique if item.flipped)
        rate = successes / len(unique) if unique else 0.0
        return CounterfactualResult(
            original_label=original.label,
            desired_label=desired,
            candidates=unique,
            success_rate=rate,
            noise_floor=client.noise_floor,
            cost=client.cost().delta(started),
        )

    async def _search(
        self,
        client: ModelClient,
        original: dict[str, Any],
        baseline: Prediction,
        context: ExplainContext,
        desired: str | None,
        *,
        seed: int,
        calls_at_start: int,
    ) -> CounterfactualCandidate | None:
        current = dict(original)
        changes: list[CounterfactualChange] = []
        cfg = self.config.counterfactual
        rng = generator(seed)
        features = _mutable_features(original, context, cfg)
        order = list(features)
        rng.shuffle(order)
        for _step in range(cfg.max_features_changed):
            if client.cost().n_model_calls - calls_at_start >= cfg.call_budget:
                break
            best_feature: str | None = None
            best_value: Any = None
            best_prediction: Prediction | None = None
            best_distance = float("inf")
            for feature in order:
                if any(change.feature == feature for change in changes):
                    continue
                for value in _grid(feature, original.get(feature), cfg, context):
                    if value == current.get(feature):
                        continue
                    if client.cost().n_model_calls - calls_at_start >= cfg.call_budget:
                        break
                    trial = dict(current)
                    trial[feature] = value
                    prediction = await client.predict(trial)
                    distance = _distance(original, trial, context, cfg)
                    flipped = _flipped(prediction, baseline.label, desired)
                    better = flipped and (
                        best_prediction is None
                        or not _flipped(best_prediction, baseline.label, desired)
                        or distance < best_distance
                    )
                    closer = (
                        not flipped
                        and best_prediction is not None
                        and not _flipped(best_prediction, baseline.label, desired)
                        and _score(prediction, desired, baseline.label)
                        > _score(best_prediction, desired, baseline.label)
                    )
                    if best_prediction is None or better or closer:
                        if flipped or best_prediction is None or closer or better:
                            best_feature = feature
                            best_value = value
                            best_prediction = prediction
                            best_distance = distance
            if best_feature is None or best_prediction is None:
                break
            current[best_feature] = best_value
            changes.append(
                CounterfactualChange(
                    feature=best_feature, before=original.get(best_feature), after=best_value
                )
            )
            if _flipped(best_prediction, baseline.label, desired):
                break
        if not changes:
            return None
        changes = await self._polish(
            client, original, current, changes, baseline.label, desired, context, cfg
        )
        final = await client.predict(current)
        confirmed = _flipped(final, baseline.label, desired)
        margin = _margin(final, baseline)
        below = (
            client.noise_floor is not None
            and margin is not None
            and abs(margin) < client.noise_floor
        )
        return CounterfactualCandidate(
            changes=changes,
            label=final.label,
            probability=final.probability,
            distance=_distance(original, _apply(original, changes), context, cfg),
            sparsity=len(changes),
            margin=margin,
            flipped=confirmed,
            below_noise_floor=below,
            replay_confirmed=confirmed,
            noise_floor=client.noise_floor,
        )

    async def _polish(
        self,
        client: ModelClient,
        original: dict[str, Any],
        current: dict[str, Any],
        changes: list[CounterfactualChange],
        original_label: str,
        desired: str | None,
        context: ExplainContext,
        cfg: CounterfactualConfig,
    ) -> list[CounterfactualChange]:
        del context, cfg
        kept = list(changes)
        for change in list(kept):
            trial_changes = [item for item in kept if item.feature != change.feature]
            trial = _apply(original, trial_changes)
            prediction = await client.predict(trial)
            if _flipped(prediction, original_label, desired):
                kept = trial_changes
                current.clear()
                current.update(trial)
        return kept


def _apply(original: Mapping[str, Any], changes: list[CounterfactualChange]) -> dict[str, Any]:
    updated = dict(original)
    for change in changes:
        updated[change.feature] = change.after
    return updated


def _mutable_features(
    instance: Mapping[str, Any], context: ExplainContext, cfg: CounterfactualConfig
) -> list[str]:
    if context.features:
        names = [
            spec.name
            for spec in context.features
            if spec.mutable and spec.name not in cfg.immutable_features
        ]
        return names
    return [key for key in instance if key not in cfg.immutable_features]


def _grid(
    feature: str, value: Any, cfg: CounterfactualConfig, context: ExplainContext
) -> list[Any]:
    spec = context.feature(feature)
    if feature in cfg.immutable_features or (spec is not None and not spec.mutable):
        return []
    if spec is not None and spec.kind == "boolean":
        return [True, False]
    if feature in cfg.categorical_values:
        return list(cfg.categorical_values[feature])
    if spec is not None and spec.allowed_values:
        return list(spec.allowed_values)
    if feature in cfg.numeric_grid:
        return list(cfg.numeric_grid[feature])
    if _is_numeric(value, spec):
        low, high = _range_for(feature, value, cfg, spec)
        return [low + (high - low) * quantile for quantile in cfg.grid_quantiles]
    return []


def _is_numeric(value: Any, spec: FeatureSpec | None) -> bool:
    if spec is not None and spec.kind == "numeric":
        return True
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _range_for(
    feature: str, value: Any, cfg: CounterfactualConfig, spec: FeatureSpec | None
) -> tuple[float, float]:
    if feature in cfg.allowed_ranges:
        return cfg.allowed_ranges[feature]
    if spec is not None and spec.allowed_range is not None:
        return spec.allowed_range
    numeric = (
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
    )
    low, high = numeric * 0.5, numeric * 1.5
    if low == high:
        return numeric - 1.0, numeric + 1.0
    return low, high


def _flipped(prediction: Prediction, original_label: str, desired: str | None) -> bool:
    if desired is not None:
        return prediction.label == desired
    return prediction.label != original_label


def _score(prediction: Prediction, desired: str | None, original_label: str) -> float:
    if desired is not None:
        value = prediction.class_probability(desired)
        if value is not None:
            return value
    value = prediction.class_probability(original_label)
    if value is None:
        return 0.0
    return 1.0 - value


def _margin(final: Prediction, baseline: Prediction) -> float | None:
    if final.probability is None or baseline.probability is None:
        return None
    return final.probability - baseline.probability


def _distance(
    original: Mapping[str, Any],
    trial: Mapping[str, Any],
    context: ExplainContext,
    cfg: CounterfactualConfig,
) -> float:
    names = _mutable_features(original, context, cfg) or list(original)
    if not names:
        return 0.0
    if cfg.distance_metric == "hamming":
        return sum(1 for name in names if original.get(name) != trial.get(name)) / len(names)
    total = 0.0
    for name in names:
        before = original.get(name)
        after = trial.get(name)
        spec = context.feature(name)
        if (
            isinstance(before, (int, float))
            and isinstance(after, (int, float))
            and not isinstance(before, bool)
            and not isinstance(after, bool)
        ):
            low, high = _range_for(name, before, cfg, spec)
            span = high - low or 1.0
            total += abs(float(after) - float(before)) / span
        elif before != after:
            total += 1.0
    return total / len(names)


def _dedupe(candidates: list[CounterfactualCandidate]) -> list[CounterfactualCandidate]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    unique: list[CounterfactualCandidate] = []
    for candidate in candidates:
        key = tuple(sorted((change.feature, repr(change.after)) for change in candidate.changes))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    unique.sort(key=lambda item: (not item.flipped, item.sparsity, item.distance))
    return unique
