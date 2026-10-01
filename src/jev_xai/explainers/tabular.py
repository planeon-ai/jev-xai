"""Tabular matrices for optional attribution extras.

SHAP and LIME call ``model.predict`` directly. Those samples are counted on the
result, and they are not written to the cassette.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from jev_xai.errors import BudgetExceededError, JevXaiUsageError
from jev_xai.explainers.context import ExplainContext, FeatureSpec
from jev_xai.types import coerce_prediction


def extra_installed(module: str) -> bool:
    """True when an optional distribution can be imported."""

    return importlib.util.find_spec(module) is not None


def load_extra(module: str, extra: str) -> Any:
    """Import an optional module or raise a usage error that names the extra."""

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise JevXaiUsageError(
            f"{extra} is optional and is not installed. Install it with "
            f"`pip install jev-xai[{extra}]`."
        ) from exc


def tabular_columns(context: ExplainContext) -> tuple[list[FeatureSpec], list[str]]:
    """Keep numeric, boolean, and categorical columns. Skip text and open categoricals."""

    kept: list[FeatureSpec] = []
    skipped: list[str] = []
    for spec in context.features:
        if spec.kind == "text" or (spec.kind == "categorical" and not spec.allowed_values):
            skipped.append(spec.name)
        else:
            kept.append(spec)
    return kept, skipped


def background_instances(context: ExplainContext) -> tuple[list[dict[str, Any]], str]:
    """Rows the attribution is relative to. A single ``background`` dict is a coarse baseline."""

    if context.background_rows:
        return [dict(row) for row in context.background_rows], ""
    if context.background:
        return [dict(context.background)], (
            "one-row background; attributions are relative to that single baseline"
        )
    return [], ""


def encode_row(instance: Mapping[str, Any], specs: Sequence[FeatureSpec]) -> list[float]:
    encoded: list[float] = []
    for spec in specs:
        raw = instance.get(spec.name)
        if spec.kind == "boolean":
            encoded.append(1.0 if raw else 0.0)
        elif spec.kind == "numeric":
            encoded.append(0.0 if raw is None else float(raw))
        else:
            choices = list(spec.allowed_values or [])
            encoded.append(float(choices.index(raw)) if raw in choices else -1.0)
    return encoded


def decode_row(
    base: Mapping[str, Any], specs: Sequence[FeatureSpec], vector: np.ndarray
) -> dict[str, Any]:
    row = dict(base)
    for spec, number in zip(specs, vector, strict=True):
        value = float(number)
        if spec.kind == "boolean":
            row[spec.name] = value >= 0.5
        elif spec.kind == "numeric":
            row[spec.name] = value
        else:
            choices = list(spec.allowed_values or [])
            if not choices:
                continue
            index = min(max(int(round(value)), 0), len(choices) - 1)
            row[spec.name] = choices[index]
    return row


class DirectCalls:
    """Count synchronous ``model.predict`` calls made by an attribution library."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.calls = 0

    def probabilities(
        self,
        model: Any,
        base: Mapping[str, Any],
        specs: Sequence[FeatureSpec],
        matrix: np.ndarray,
        label: str,
    ) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(matrix, dtype=float))
        self.calls += int(rows.shape[0])
        if self.calls > self.budget:
            raise BudgetExceededError(f"attribution call budget exhausted ({self.budget} calls)")
        values: list[float] = []
        for vector in rows:
            prediction = coerce_prediction(model.predict(decode_row(base, specs, vector)))
            probability = prediction.class_probability(label)
            values.append(0.0 if probability is None else float(probability))
        return np.asarray(values, dtype=float)
