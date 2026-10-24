"""Optional LIME tabular adapter. The ``lime`` package is not a core dependency."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import anyio
import numpy as np
from pydantic import Field

from jev_xai.adapters.capabilities import diagnose, require
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import JevXaiError
from jev_xai.explainers.base import Explainer, ExplanationResult
from jev_xai.explainers.context import ExplainContext
from jev_xai.explainers.tabular import (
    DirectCalls,
    background_instances,
    encode_row,
    load_extra,
    tabular_columns,
)
from jev_xai.model.client import ModelClient
from jev_xai.types import CostEnvelope


class LimeRow(ExplanationResult):
    explainer: str = "lime"
    feature: str
    weight: float


class LimeResult(ExplanationResult):
    explainer: str = "lime"
    label: str = ""
    rows: list[LimeRow] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    note: str = ""
    algorithm: str = "lime_tabular"


class LimeExplainer(Explainer):
    """LIME on the same tabular columns as :class:`jev_xai.explainers.shap.ShapExplainer`.

    The surrogate is regression on P(original label). Sample calls use
    ``model.predict`` and are not written to the cassette.
    """

    name = "lime"

    def __init__(self, config: JevXaiConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed

    async def explain(
        self,
        client: ModelClient,
        instance: Mapping[str, Any],
        context: ExplainContext | None = None,
    ) -> LimeResult:
        context = context or ExplainContext()
        require(diagnose(client.model, input_reachable=True, context=context), "lime")
        specs, skipped = tabular_columns(context)
        if not specs:
            raise JevXaiError("lime needs at least one numeric, boolean, or categorical feature")
        rows, note = background_instances(context)
        started = client.cost()
        original = await client.predict(dict(instance))
        background = np.asarray([encode_row(row, specs) for row in rows], dtype=float)
        point = np.asarray(encode_row(instance, specs), dtype=float)
        counter = DirectCalls(self.config.lime.call_budget)
        names = [spec.name for spec in specs]
        width = min(self.config.lime.num_features, len(names))
        samples = min(self.config.lime.num_samples, self.config.lime.call_budget)

        def predict_fn(matrix: np.ndarray) -> np.ndarray:
            return counter.probabilities(client.model, instance, specs, matrix, original.label)

        def run() -> list[tuple[int, float]]:
            return lime_weights(
                predict_fn,
                background,
                point,
                names,
                num_features=width,
                num_samples=samples,
                seed=self.seed,
            )

        clock = time.perf_counter()
        weights = await anyio.to_thread.run_sync(run)
        wall_ms = (time.perf_counter() - clock) * 1000
        delta = client.cost().delta(started)
        return LimeResult(
            label=original.label,
            rows=[
                LimeRow(feature=names[index], weight=weight)
                for index, weight in weights
                if 0 <= index < len(names)
            ],
            skipped=skipped,
            note=note,
            noise_floor=client.noise_floor,
            cost=CostEnvelope(
                n_model_calls=delta.n_model_calls + counter.calls,
                cache_hits=delta.cache_hits,
                wall_ms=delta.wall_ms + wall_ms,
            ),
        )


def lime_weights(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    instance: np.ndarray,
    names: list[str],
    *,
    num_features: int,
    num_samples: int,
    seed: int,
    module: Any | None = None,
) -> list[tuple[int, float]]:
    """Run LIME tabular regression. Pass ``module`` in tests to avoid importing ``lime``."""

    lime_tabular = load_extra("lime.lime_tabular", "lime") if module is None else module
    explainer = lime_tabular.LimeTabularExplainer(
        background,
        feature_names=names,
        mode="regression",
        discretize_continuous=False,
        random_state=seed,
    )
    explained = explainer.explain_instance(
        instance,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
    )
    mapping = explained.as_map()
    if not mapping:
        return []
    pairs = mapping[next(iter(mapping))]
    return [(int(index), float(weight)) for index, weight in pairs]
