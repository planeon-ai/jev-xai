"""Perturbation domains declared by FeatureSpec. Does not touch the global RNG."""

from __future__ import annotations

from typing import Any

import numpy as np

from jev_xai.explainers.context import FeatureSpec


def is_perturbable(spec: FeatureSpec, original: Any) -> bool:
    """True when the spec allows at least one value other than the instance."""

    if not spec.mutable:
        return False
    if spec.kind == "boolean":
        return len(_bool_choices(spec)) > 1
    if spec.kind == "numeric":
        if spec.allowed_range is None:
            return False
        low, high = spec.allowed_range
        return high > low
    return len(_discrete_choices(spec, original)) > 1


def draw_value(
    spec: FeatureSpec,
    original: Any,
    rng: np.random.Generator,
    *,
    hold: tuple[Any, ...] | None = None,
) -> Any:
    """Draw one value. ``hold`` is ``("eq", value)`` or ``("within", low, high)``."""

    if hold is not None:
        return _held(hold, rng)
    if not spec.mutable:
        return original
    if spec.kind == "boolean":
        choices = _bool_choices(spec)
        return choices[int(rng.integers(0, len(choices)))]
    if spec.kind == "numeric":
        if spec.allowed_range is None:
            return original
        low, high = spec.allowed_range
        return _uniform(rng, low, high)
    choices = _discrete_choices(spec, original)
    return choices[int(rng.integers(0, len(choices)))]


def draw_alternative(spec: FeatureSpec, original: Any, rng: np.random.Generator) -> Any:
    """Draw a value different from ``original``. Caller checks :func:`is_perturbable`."""

    if spec.kind == "boolean":
        choices = [item for item in _bool_choices(spec) if item is not bool(original)]
        if not choices:
            return bool(original)
        return choices[int(rng.integers(0, len(choices)))]
    if spec.kind == "numeric" and spec.allowed_range is not None:
        low, high = spec.allowed_range
        current = _as_float(original, low)
        for _ in range(8):
            value = _uniform(rng, low, high)
            if value != current:
                return value
        return float(high if current == float(low) else low)
    choices = [item for item in _discrete_choices(spec, original) if item != original]
    if not choices:
        return original
    return choices[int(rng.integers(0, len(choices)))]


def numeric_band(spec: FeatureSpec, value: Any, fraction: float) -> tuple[float, float]:
    """A window of ``fraction`` of ``allowed_range``, centered on ``value`` and clamped."""

    if spec.allowed_range is None:
        number = _as_float(value, 0.0)
        return number, number
    low, high = spec.allowed_range
    center = min(max(_as_float(value, (low + high) / 2), low), high)
    half = (high - low) * fraction / 2
    return max(low, center - half), min(high, center + half)


def _held(hold: tuple[Any, ...], rng: np.random.Generator) -> Any:
    op = hold[0]
    if op == "eq":
        return hold[1]
    low = float(hold[1])
    high = float(hold[2])
    return _uniform(rng, low, high)


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    if high <= low:
        return float(low)
    return float(rng.uniform(low, high))


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _bool_choices(spec: FeatureSpec) -> list[bool]:
    if spec.allowed_values is None:
        return [False, True]
    unique: list[bool] = []
    for item in spec.allowed_values:
        coerced = bool(item)
        if coerced not in unique:
            unique.append(coerced)
    return unique or [False]


def _discrete_choices(spec: FeatureSpec, original: Any) -> list[Any]:
    if spec.allowed_values:
        return list(spec.allowed_values)
    return [original]
