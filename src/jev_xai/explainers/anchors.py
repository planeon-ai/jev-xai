"""Greedy anchors: a short rule that keeps the predicted label.

Precision is the fraction of seeded perturbations, with the rule held, that
still predict the original label. Coverage is the fraction of unconditional
perturbations that satisfy the rule, and it does not call the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from jev_xai.adapters.capabilities import diagnose, require
from jev_xai.config.schema import JevXaiConfig
from jev_xai.explainers.base import Explainer, ExplanationResult
from jev_xai.explainers.context import ExplainContext
from jev_xai.explainers.sampling import draw_value, is_perturbable, numeric_band
from jev_xai.model.client import ModelClient
from jev_xai.replay.seeding import generator


class AnchorPredicate(BaseModel):
    """One condition in an anchor. ``eq`` is exact; ``within`` is a numeric band."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    op: Literal["eq", "within"]
    value: Any = None
    low: float | None = None
    high: float | None = None


class AnchorResult(ExplanationResult):
    explainer: str = "anchors"
    label: str = ""
    predicates: list[AnchorPredicate] = Field(default_factory=list)
    precision: float = 0.0
    coverage: float = 1.0
    sufficient: bool = False
    samples: int = 0
    algorithm: str = "greedy_anchor"


class AnchorExplainer(Explainer):
    """Grow a rule until precision reaches the threshold or the budget stops it.

    Immutable features and features with a single allowed value are not
    candidates. The best rule found so far is returned even when it is not
    sufficient.
    """

    name = "anchors"

    def __init__(self, config: JevXaiConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed

    async def explain(
        self,
        client: ModelClient,
        instance: Mapping[str, Any],
        context: ExplainContext | None = None,
    ) -> AnchorResult:
        context = context or ExplainContext()
        require(diagnose(client.model, input_reachable=True, context=context), "anchors")
        started = client.cost()
        original = await client.predict(dict(instance))
        cfg = self.config.anchors
        rng = generator(self.seed)
        chosen: list[AnchorPredicate] = []
        candidates = _candidates(dict(instance), context, cfg.numeric_band)
        precision = 0.0
        samples = 0
        sufficient = False
        while len(chosen) < cfg.max_size:
            remaining = _remaining(client, started.n_model_calls, cfg.call_budget)
            if remaining < cfg.samples or not _pending(candidates, chosen):
                break
            parent_precision, worlds = await _precision(
                client,
                dict(instance),
                context,
                {item.feature: item for item in chosen},
                original.label,
                rng,
                cfg.samples,
            )
            samples += len(worlds)
            precision = parent_precision
            if precision >= cfg.precision and not chosen:
                sufficient = True
                break
            added, precision, spent = await _best_addition(
                client,
                worlds,
                context,
                candidates,
                chosen,
                original.label,
                rng,
                _remaining(client, started.n_model_calls, cfg.call_budget),
                parent_precision,
            )
            samples += spent
            if added is None:
                break
            chosen.append(added)
            if precision >= cfg.precision:
                sufficient = True
                break
        coverage = _coverage(dict(instance), context, chosen, rng, cfg.coverage_samples)
        return AnchorResult(
            label=original.label,
            predicates=chosen,
            precision=precision,
            coverage=coverage,
            sufficient=sufficient,
            samples=samples,
            noise_floor=client.noise_floor,
            cost=client.cost().delta(started),
        )


def _candidates(
    instance: Mapping[str, Any], context: ExplainContext, fraction: float
) -> list[AnchorPredicate]:
    found: list[AnchorPredicate] = []
    for spec in context.features:
        original = instance.get(spec.name)
        if not is_perturbable(spec, original):
            continue
        if spec.kind == "numeric":
            low, high = numeric_band(spec, original, fraction)
            found.append(
                AnchorPredicate(
                    feature=spec.name,
                    op="within",
                    value=original,
                    low=low,
                    high=high,
                )
            )
        else:
            value = bool(original) if spec.kind == "boolean" else original
            found.append(AnchorPredicate(feature=spec.name, op="eq", value=value))
    return found


def _pending(candidates: list[AnchorPredicate], chosen: list[AnchorPredicate]) -> bool:
    chosen_names = {item.feature for item in chosen}
    return any(item.feature not in chosen_names for item in candidates)


async def _precision(
    client: ModelClient,
    instance: dict[str, Any],
    context: ExplainContext,
    held: dict[str, AnchorPredicate],
    label: str,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, list[dict[str, Any]]]:
    worlds: list[dict[str, Any]] = []
    hits = 0
    for _ in range(samples):
        world = _world(instance, context, held, rng)
        prediction = await client.predict(world)
        worlds.append(world)
        if prediction.label == label:
            hits += 1
    if not worlds:
        return 0.0, worlds
    return hits / len(worlds), worlds


async def _best_addition(
    client: ModelClient,
    worlds: list[dict[str, Any]],
    context: ExplainContext,
    candidates: list[AnchorPredicate],
    chosen: list[AnchorPredicate],
    label: str,
    rng: np.random.Generator,
    remaining: int,
    parent_precision: float,
) -> tuple[AnchorPredicate | None, float, int]:
    """Pick the candidate that strictly raises precision on the same worlds."""

    chosen_names = {item.feature for item in chosen}
    best: AnchorPredicate | None = None
    best_precision = parent_precision
    spent = 0
    for candidate in candidates:
        if candidate.feature in chosen_names:
            continue
        spec = context.feature(candidate.feature)
        if spec is None:
            continue
        if remaining - spent < len(worlds):
            break
        hits = 0
        for world in worlds:
            trial = dict(world)
            trial[candidate.feature] = draw_value(
                spec,
                world.get(candidate.feature),
                rng,
                hold=_hold(candidate),
            )
            prediction = await client.predict(trial)
            spent += 1
            if prediction.label == label:
                hits += 1
        precision = hits / len(worlds) if worlds else 0.0
        if precision > best_precision:
            best = candidate
            best_precision = precision
    return best, best_precision, spent


def _world(
    instance: Mapping[str, Any],
    context: ExplainContext,
    held: dict[str, AnchorPredicate],
    rng: np.random.Generator,
) -> dict[str, Any]:
    world = dict(instance)
    for spec in context.features:
        predicate = held.get(spec.name)
        world[spec.name] = draw_value(
            spec,
            instance.get(spec.name),
            rng,
            hold=_hold(predicate) if predicate is not None else None,
        )
    return world


def _hold(predicate: AnchorPredicate) -> tuple[Any, ...]:
    if predicate.op == "within":
        return ("within", predicate.low, predicate.high)
    return ("eq", predicate.value)


def _coverage(
    instance: Mapping[str, Any],
    context: ExplainContext,
    predicates: list[AnchorPredicate],
    rng: np.random.Generator,
    samples: int,
) -> float:
    if not predicates or samples <= 0:
        return 1.0
    hits = 0
    for _ in range(samples):
        world = _world(instance, context, {}, rng)
        if all(_matches(world, predicate) for predicate in predicates):
            hits += 1
    return hits / samples


def _matches(world: Mapping[str, Any], predicate: AnchorPredicate) -> bool:
    got = world.get(predicate.feature)
    if predicate.op == "eq":
        return bool(got == predicate.value)
    if got is None or predicate.low is None or predicate.high is None:
        return False
    return predicate.low <= float(got) <= predicate.high


def _remaining(client: ModelClient, started_calls: int, budget: int) -> int:
    used = client.cost().n_model_calls - started_calls
    return max(0, budget - used)
