"""Ablation and occlusion. Delta P is reported per masking strategy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from jev_xai.adapters.capabilities import diagnose, require
from jev_xai.config.schema import AblationConfig, JevXaiConfig
from jev_xai.errors import CapabilityError, JevXaiError
from jev_xai.explainers.base import Explainer, ExplanationResult
from jev_xai.explainers.context import ExplainContext, FeatureSpec
from jev_xai.model.client import ModelClient

_DROP = object()


class AblationRow(ExplanationResult):
    explainer: str = "ablation"
    feature: str
    scope: Literal["single", "group", "text_span", "whole_input"]
    masking_strategy: str
    baseline_probability: float | None = None
    ablated_probability: float | None = None
    delta_p: float | None = None
    label_flipped: bool = False
    below_noise_floor: bool = False
    noop: bool = False
    original_label: str = ""
    ablated_label: str = ""


class AblationResult(ExplanationResult):
    explainer: str = "ablation"
    rows: list[AblationRow] = Field(default_factory=list)
    suppressed_rows: list[AblationRow] = Field(default_factory=list)
    masking_policy: dict[str, Any] = Field(default_factory=dict)


class AblationExplainer(Explainer):
    """Single-feature, grouped, structured-field, and text-span ablation."""

    name = "ablation"

    def __init__(self, config: JevXaiConfig) -> None:
        self.config = config

    async def explain(
        self,
        client: ModelClient,
        instance: Mapping[str, Any],
        context: ExplainContext | None = None,
    ) -> AblationResult:
        context = context or ExplainContext()
        diagnosis = diagnose(
            client.model,
            input_reachable=True,
            context=context,
        )
        require(diagnosis, "ablation")
        started = client.cost()
        original = await client.predict(dict(instance))
        baseline = original.class_probability()
        rows = await self._rows(client, dict(instance), context, original.label, baseline)
        kept, suppressed = _apply_noise_policy(
            rows, self.config.reproducibility.on_below_noise_floor
        )
        return AblationResult(
            rows=kept,
            suppressed_rows=suppressed,
            masking_policy=self.config.ablation.masking.model_dump(mode="json"),
            noise_floor=client.noise_floor,
            cost=client.cost().delta(started),
        )

    async def _rows(
        self,
        client: ModelClient,
        instance: dict[str, Any],
        context: ExplainContext,
        original_label: str,
        baseline: float | None,
    ) -> list[AblationRow]:
        if not context.features:
            masked = _mask_whole(instance, self.config.ablation)
            return [
                await self._measure(
                    client,
                    instance,
                    masked,
                    feature="whole_input",
                    scope="whole_input",
                    strategy="whole_input",
                    original_label=original_label,
                    baseline=baseline,
                )
            ]
        rows: list[AblationRow] = []
        strategy = self.config.ablation.masking.structured
        for spec in context.features:
            masked = _mask_fields(instance, [spec], strategy, context, self.config.ablation)
            rows.append(
                await self._measure(
                    client,
                    instance,
                    masked,
                    feature=spec.name,
                    scope="single",
                    strategy=strategy,
                    original_label=original_label,
                    baseline=baseline,
                )
            )
            if spec.kind == "text":
                rows.extend(
                    await self._text_spans(
                        client, instance, spec, original_label, baseline, context
                    )
                )
        groups = {**self.config.ablation.feature_groups, **context.groups}
        for name, members in groups.items():
            specs = [spec for spec in context.features if spec.name in members]
            masked = _mask_fields(instance, specs, strategy, context, self.config.ablation)
            rows.append(
                await self._measure(
                    client,
                    instance,
                    masked,
                    feature=f"group:{name}",
                    scope="group",
                    strategy=strategy,
                    original_label=original_label,
                    baseline=baseline,
                )
            )
        return rows

    async def _text_spans(
        self,
        client: ModelClient,
        instance: dict[str, Any],
        spec: FeatureSpec,
        original_label: str,
        baseline: float | None,
        context: ExplainContext,
    ) -> list[AblationRow]:
        text = str(instance.get(spec.name, ""))
        policy = self.config.ablation.masking.text
        rows: list[AblationRow] = []
        for index, (start, end) in enumerate(_spans(text, self.config.ablation)):
            updated = dict(instance)
            updated[spec.name] = _mask_span(text, start, end, policy, self.config.ablation)
            rows.append(
                await self._measure(
                    client,
                    instance,
                    updated,
                    feature=f"text:{spec.name}:{index}",
                    scope="text_span",
                    strategy=policy,
                    original_label=original_label,
                    baseline=baseline,
                )
            )
        return rows

    async def _measure(
        self,
        client: ModelClient,
        original: Mapping[str, Any],
        masked: Mapping[str, Any],
        *,
        feature: str,
        scope: Literal["single", "group", "text_span", "whole_input"],
        strategy: str,
        original_label: str,
        baseline: float | None,
    ) -> AblationRow:
        if _same_input(original, masked):
            return AblationRow(
                feature=feature,
                scope=scope,
                masking_strategy=strategy,
                baseline_probability=baseline,
                delta_p=None,
                label_flipped=False,
                below_noise_floor=False,
                noop=True,
                original_label=original_label,
                ablated_label=original_label,
                noise_floor=client.noise_floor,
            )
        prediction = await client.predict(masked)
        ablated = prediction.class_probability(original_label)
        delta = None if baseline is None or ablated is None else baseline - ablated
        below = _below_floor(delta, client.noise_floor)
        return AblationRow(
            feature=feature,
            scope=scope,
            masking_strategy=strategy,
            baseline_probability=baseline,
            ablated_probability=ablated,
            delta_p=delta,
            label_flipped=prediction.label != original_label,
            below_noise_floor=below,
            original_label=original_label,
            ablated_label=prediction.label,
            noise_floor=client.noise_floor,
        )


def _same_input(original: Mapping[str, Any], masked: Mapping[str, Any]) -> bool:
    """True when the mask left every field as it was, so a call would not be evidence."""

    return dict(original) == dict(masked)


def _below_floor(delta: float | None, floor: float | None) -> bool:
    if delta is None or floor is None:
        return False
    return abs(delta) < floor


def _apply_noise_policy(
    rows: list[AblationRow], policy: str
) -> tuple[list[AblationRow], list[AblationRow]]:
    if policy == "error" and any(row.below_noise_floor for row in rows):
        raise JevXaiError(
            "ablation delta is below the measured noise floor; "
            "raise noise_floor_sigma_k or set on_below_noise_floor to 'flag'"
        )
    if policy == "drop":
        return (
            [row for row in rows if not row.below_noise_floor],
            [row for row in rows if row.below_noise_floor],
        )
    return rows, []


def _mask_whole(instance: Mapping[str, Any], config: AblationConfig) -> dict[str, Any]:
    del instance, config
    return {}


def _neutral(spec: FeatureSpec, config: AblationConfig, context: ExplainContext) -> Any:
    if spec.baseline is not None:
        return spec.baseline
    if spec.kind == "numeric":
        return config.masking.neutral_numeric
    if spec.kind == "boolean":
        return False
    if spec.kind == "text":
        return config.masking.neutral_text
    return None


def _mask_fields(
    instance: Mapping[str, Any],
    specs: list[FeatureSpec],
    strategy: str,
    context: ExplainContext,
    config: AblationConfig,
) -> dict[str, Any]:
    updated = dict(instance)
    for spec in specs:
        if strategy == "drop_key":
            updated.pop(spec.name, None)
        elif strategy == "null":
            updated[spec.name] = None
        elif strategy == "background":
            if not context.background or spec.name not in context.background:
                raise CapabilityError(
                    "feature_spec",
                    f"background masking needs context.background['{spec.name}']",
                )
            updated[spec.name] = context.background[spec.name]
        else:
            updated[spec.name] = _neutral(spec, config, context)
    return updated


def _spans(text: str, config: AblationConfig) -> list[tuple[int, int]]:
    if config.text_span_granularity == "token":
        found = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    elif config.text_span_granularity == "sentence":
        found = [(match.start(), match.end()) for match in re.finditer(r"[^.!?]+[.!?]?", text)]
        found = [(start, end) for start, end in found if text[start:end].strip()]
    else:
        size = max(1, config.text_chunk_chars)
        found = [(index, min(index + size, len(text))) for index in range(0, len(text), size)]
    return found[: config.max_spans]


def _mask_span(text: str, start: int, end: int, policy: str, config: AblationConfig) -> str:
    if policy == "deletion":
        return text[:start] + text[end:]
    token = config.masking.mask_token if policy == "mask_token" else config.masking.neutral_text
    return text[:start] + token + text[end:]
