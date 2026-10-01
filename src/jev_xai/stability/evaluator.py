"""Repeat an explainer and score how stable the evidence is.

``stability_score_v1`` is defined in :class:`jev_xai.config.schema.ScoreWeights`.
Decision reproducibility is not part of this score; it lives on the replay probe.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import anyio
from pydantic import Field

from jev_xai.config.loader import config_hash
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import JevXaiError
from jev_xai.explainers.ablation import AblationResult
from jev_xai.explainers.anchors import AnchorResult
from jev_xai.explainers.base import Explainer, ExplanationResult
from jev_xai.explainers.context import ExplainContext
from jev_xai.explainers.counterfactual import CounterfactualResult
from jev_xai.explainers.lime import LimeResult, LimeRow
from jev_xai.explainers.permutation import PermutationResult, PermutationRow
from jev_xai.explainers.shap import ShapResult, ShapRow
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint
from jev_xai.replay.seeding import spawn_seeds
from jev_xai.stability.metrics import jaccard, rank_correlation, sample_stdev
from jev_xai.types import CostEnvelope


class StabilityResult(ExplanationResult):
    explainer: str = "stability"
    runs: int
    n_failed_runs: int = 0
    failures: list[str] = Field(default_factory=list)
    rank_correlation: float = 0.0
    feature_overlap: float = 0.0
    attribution_variance: float = 0.0
    counterfactual_consistency: float = 0.0
    stability_score: float = 0.0
    formula: str = "stability_score_v1"
    explanations: list[dict[str, Any]] = Field(default_factory=list)


class StabilityEvaluator:
    def __init__(self, config: JevXaiConfig) -> None:
        self.config = config

    async def evaluate(
        self,
        explainer: Explainer,
        model: Any,
        instance: Mapping[str, Any],
        *,
        runs: int | None = None,
        context: ExplainContext | None = None,
        client: ModelClient | None = None,
    ) -> StabilityResult:
        total = self.config.stability.runs if runs is None else runs
        seeds = spawn_seeds(self.config.seed, total)
        started_cost = CostEnvelope()
        results: list[ExplanationResult | None] = [None] * total
        failures: list[str] = []
        shared = client or _client(model, self.config)

        async def one(index: int, seed: int) -> None:
            try:
                bound = _with_seed(explainer, seed)
                results[index] = await bound.explain(shared, dict(instance), context)
            except Exception as exc:
                failures.append(f"run {index}: {exc}")

        with shared.ignore_cache():
            async with anyio.create_task_group() as group:
                for index, seed in enumerate(seeds):
                    group.start_soon(one, index, seed)

        failed = len(failures)
        if total and failed / total > self.config.stability.failure_tolerance:
            raise JevXaiError(
                f"{failed} of {total} stability runs failed, above tolerance "
                f"{self.config.stability.failure_tolerance}"
            )
        successful = [item for item in results if item is not None]
        score = _score(successful, self.config)
        return StabilityResult(
            runs=total,
            n_failed_runs=failed,
            failures=failures,
            rank_correlation=score["rank_correlation"],
            feature_overlap=score["feature_overlap"],
            attribution_variance=score["attribution_variance"],
            counterfactual_consistency=score["counterfactual_consistency"],
            stability_score=score["stability_score"],
            explanations=[item.model_dump(mode="json") for item in successful],
            cost=shared.cost().delta(started_cost),
            config_hash=config_hash(self.config),
            noise_floor=shared.noise_floor,
        )

    async def evaluate_sequential(
        self,
        explainer: Explainer,
        model: Any,
        instance: Mapping[str, Any],
        *,
        runs: int | None = None,
        context: ExplainContext | None = None,
        client: ModelClient | None = None,
    ) -> StabilityResult:
        """Same seeds as :meth:`evaluate`, executed one at a time. Used to prove order independence."""

        total = self.config.stability.runs if runs is None else runs
        seeds = spawn_seeds(self.config.seed, total)
        shared = client or _client(model, self.config)
        collected: list[ExplanationResult] = []
        failures: list[str] = []
        with shared.ignore_cache():
            for index, seed in enumerate(seeds):
                try:
                    collected.append(
                        await _with_seed(explainer, seed).explain(shared, dict(instance), context)
                    )
                except Exception as exc:
                    failures.append(f"run {index}: {exc}")
        if total and len(failures) / total > self.config.stability.failure_tolerance:
            raise JevXaiError("stability failure tolerance exceeded")
        score = _score(collected, self.config)
        return StabilityResult(
            runs=total,
            n_failed_runs=len(failures),
            failures=failures,
            rank_correlation=score["rank_correlation"],
            feature_overlap=score["feature_overlap"],
            attribution_variance=score["attribution_variance"],
            counterfactual_consistency=score["counterfactual_consistency"],
            stability_score=score["stability_score"],
            explanations=[item.model_dump(mode="json") for item in collected],
            cost=shared.cost(),
            config_hash=config_hash(self.config),
        )


def _client(model: Any, config: JevXaiConfig) -> ModelClient:
    return ModelClient(
        model,
        config.model,
        fingerprint=model_fingerprint(model),
        seed=config.seed,
    )


def _with_seed(explainer: Explainer, seed: int) -> Explainer:
    if not hasattr(explainer, "seed"):
        return explainer
    config = getattr(explainer, "config")  # noqa: B009
    clone = explainer.__class__(config, seed=seed)  # type: ignore[call-arg]
    return clone


def _vectors(
    results: list[ExplanationResult],
) -> tuple[list[list[str]], list[dict[str, float]], list[frozenset[str]]]:
    rankings: list[list[str]] = []
    attributions: list[dict[str, float]] = []
    change_sets: list[frozenset[str]] = []
    for result in results:
        if isinstance(result, AblationResult):
            measured = [row for row in result.rows if not row.noop]
            ordered = sorted(
                measured,
                key=lambda row: abs(row.delta_p) if row.delta_p is not None else -1.0,
                reverse=True,
            )
            rankings.append([row.feature for row in ordered])
            attributions.append({row.feature: row.delta_p or 0.0 for row in measured})
            change_sets.append(frozenset())
        elif isinstance(result, CounterfactualResult):
            if result.candidates:
                features = [change.feature for change in result.candidates[0].changes]
            else:
                features = []
            rankings.append(features)
            attributions.append({name: float(index + 1) for index, name in enumerate(features)})
            change_sets.append(frozenset(features))
        elif isinstance(result, AnchorResult):
            names = [predicate.feature for predicate in result.predicates]
            rankings.append(names)
            attributions.append({name: result.precision for name in names})
            change_sets.append(frozenset(names))
        elif isinstance(result, PermutationResult):
            ranked = _sort_permutations(result.rows)
            rankings.append([row.feature for row in ranked])
            attributions.append({row.feature: row.importance or 0.0 for row in result.rows})
            change_sets.append(frozenset())
        elif isinstance(result, ShapResult):
            ranked_shap = _sort_shap(result.rows)
            rankings.append([row.feature for row in ranked_shap])
            attributions.append({row.feature: row.value for row in result.rows})
            change_sets.append(frozenset())
        elif isinstance(result, LimeResult):
            ranked_lime = _sort_lime(result.rows)
            rankings.append([row.feature for row in ranked_lime])
            attributions.append({row.feature: row.weight for row in result.rows})
            change_sets.append(frozenset())
        else:
            rankings.append([])
            attributions.append({})
            change_sets.append(frozenset())
    return rankings, attributions, change_sets


def _score(results: list[ExplanationResult], config: JevXaiConfig) -> dict[str, float]:
    weights = config.stability.score_weights
    if len(results) < 2:
        return {
            "rank_correlation": 0.0,
            "feature_overlap": 0.0,
            "attribution_variance": 0.0,
            "counterfactual_consistency": 0.0,
            "stability_score": 0.0,
        }
    rankings, attributions, change_sets = _vectors(results)
    correlations: list[float] = []
    overlaps: list[float] = []
    universe = sorted({name for item in attributions for name in item})
    for left in range(len(results)):
        for right in range(left + 1, len(results)):
            left_ranks = _rank_vector(rankings[left], universe)
            right_ranks = _rank_vector(rankings[right], universe)
            correlations.append(
                rank_correlation(left_ranks, right_ranks, config.stability.rank_metric)
            )
            k = config.stability.top_k
            overlaps.append(jaccard(set(rankings[left][:k]), set(rankings[right][:k])))
    mean_corr = sum(correlations) / len(correlations) if correlations else 0.0
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    variances = []
    for name in universe:
        series = [item.get(name, 0.0) for item in attributions]
        variances.append(sample_stdev(series) ** 2)
    mean_var = sum(variances) / len(variances) if variances else 0.0
    attr_stability = 1.0 / (1.0 + mean_var)
    if any(change_sets):
        modal, count = Counter(change_sets).most_common(1)[0]
        del modal
        consistency = count / len(change_sets)
    else:
        consistency = 1.0
    score = (
        weights.rank_correlation * ((mean_corr + 1) / 2)
        + weights.feature_overlap * mean_overlap
        + weights.attribution_stability * attr_stability
        + weights.counterfactual_consistency * consistency
    )
    return {
        "rank_correlation": mean_corr,
        "feature_overlap": mean_overlap,
        "attribution_variance": mean_var,
        "counterfactual_consistency": consistency,
        "stability_score": max(0.0, min(1.0, score)),
    }


def _sort_permutations(rows: list[PermutationRow]) -> list[PermutationRow]:
    return sorted(rows, key=_permutation_rank, reverse=True)


def _sort_shap(rows: list[ShapRow]) -> list[ShapRow]:
    return sorted(rows, key=lambda row: abs(row.value), reverse=True)


def _sort_lime(rows: list[LimeRow]) -> list[LimeRow]:
    return sorted(rows, key=lambda row: abs(row.weight), reverse=True)


def _permutation_rank(row: PermutationRow) -> float:
    if row.importance is None:
        return -1.0
    return abs(row.importance)


def _rank_vector(ranking: list[str], universe: list[str]) -> list[float]:
    position = {name: float(index) for index, name in enumerate(ranking)}
    worst = float(len(ranking) + 1)
    return [position.get(name, worst) for name in universe]


async def explain_with_stability(
    explainer: Explainer,
    model: Any,
    instance: Mapping[str, Any],
    config: JevXaiConfig,
    *,
    runs: int | None = None,
    context: ExplainContext | None = None,
) -> StabilityResult:
    """Public helper around :class:`StabilityEvaluator`."""

    return await StabilityEvaluator(config).evaluate(
        explainer, model, instance, runs=runs, context=context
    )


def assert_stable(result: StabilityResult, min_score: float = 0.8) -> None:
    """Fail a test or a CI gate when explanation stability drops."""

    if result.stability_score < min_score:
        raise AssertionError(
            f"stability_score_v1 {result.stability_score:.3f} is below {min_score:.3f}"
        )
