"""Optional KernelSHAP adapter. The ``shap`` package is not a core dependency."""

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


class ShapRow(ExplanationResult):
    explainer: str = "shap"
    feature: str
    value: float


class ShapResult(ExplanationResult):
    explainer: str = "shap"
    label: str = ""
    rows: list[ShapRow] = Field(default_factory=list)
    base_value: float | None = None
    skipped: list[str] = Field(default_factory=list)
    note: str = ""
    algorithm: str = "kernel_shap"


class ShapExplainer(Explainer):
    """KernelSHAP on numeric, boolean, and declared categorical columns.

    Sample calls use ``model.predict`` inside the SHAP process. They are counted
    on ``cost`` and are not stored in the cassette. KernelSHAP draws from the
    global NumPy RNG; that state is saved and restored around the call.
    """

    name = "shap"

    def __init__(self, config: JevXaiConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed

    async def explain(
        self,
        client: ModelClient,
        instance: Mapping[str, Any],
        context: ExplainContext | None = None,
    ) -> ShapResult:
        context = context or ExplainContext()
        require(diagnose(client.model, input_reachable=True, context=context), "shap")
        specs, skipped = tabular_columns(context)
        if not specs:
            raise JevXaiError("shap needs at least one numeric, boolean, or categorical feature")
        rows, note = background_instances(context)
        started = client.cost()
        original = await client.predict(dict(instance))
        background = np.asarray([encode_row(row, specs) for row in rows], dtype=float)
        point = np.asarray(encode_row(instance, specs), dtype=float)
        counter = DirectCalls(self.config.shap.call_budget)
        names = [spec.name for spec in specs]

        def predict_fn(matrix: np.ndarray) -> np.ndarray:
            return counter.probabilities(client.model, instance, specs, matrix, original.label)

        def run() -> tuple[list[float], float | None]:
            return kernel_shap_values(
                predict_fn,
                background,
                point,
                nsamples=min(self.config.shap.nsamples, self.config.shap.call_budget),
                seed=self.seed,
            )

        clock = time.perf_counter()
        values, base_value = await anyio.to_thread.run_sync(run)
        if len(values) != len(names):
            raise JevXaiError(f"shap returned {len(values)} values for {len(names)} features")
        wall_ms = (time.perf_counter() - clock) * 1000
        delta = client.cost().delta(started)
        return ShapResult(
            label=original.label,
            rows=[
                ShapRow(feature=name, value=value)
                for name, value in zip(names, values, strict=True)
            ],
            base_value=base_value,
            skipped=skipped,
            note=note,
            noise_floor=client.noise_floor,
            cost=CostEnvelope(
                n_model_calls=delta.n_model_calls + counter.calls,
                cache_hits=delta.cache_hits,
                wall_ms=delta.wall_ms + wall_ms,
            ),
        )


def kernel_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    instance: np.ndarray,
    *,
    nsamples: int,
    seed: int,
    module: Any | None = None,
) -> tuple[list[float], float | None]:
    """Run KernelSHAP. Pass ``module`` in tests to avoid importing ``shap``."""

    shap = load_extra("shap", "shap") if module is None else module
    state = np.random.get_state()
    try:
        np.random.seed(seed % (2**32))
        explainer = shap.KernelExplainer(predict_fn, background)
        try:
            raw = explainer.shap_values(instance, nsamples=nsamples, silent=True)
        except TypeError:
            raw = explainer.shap_values(instance, nsamples=nsamples)
        expected = getattr(explainer, "expected_value", None)
    finally:
        np.random.set_state(state)
    array = np.asarray(raw, dtype=float).reshape(-1)
    base = None if expected is None else float(np.asarray(expected, dtype=float).reshape(-1)[0])
    return [float(value) for value in array], base
