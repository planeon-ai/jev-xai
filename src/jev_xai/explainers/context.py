"""Feature declarations and the typed context passed to every explainer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureSpec(BaseModel):
    """One input field the explainer is allowed to perturb."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["numeric", "categorical", "boolean", "text"] = "categorical"
    mutable: bool = True
    allowed_values: list[Any] | None = None
    allowed_range: tuple[float, float] | None = None
    baseline: Any = None


class ExplainContext(BaseModel):
    """Typed replacement for an untyped context dict."""

    model_config = ConfigDict(extra="forbid")

    features: list[FeatureSpec] = Field(default_factory=list)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    target_label: str | None = None
    background: dict[str, Any] | None = None
    text_field: str | None = None

    def feature(self, name: str) -> FeatureSpec | None:
        for spec in self.features:
            if spec.name == name:
                return spec
        return None
