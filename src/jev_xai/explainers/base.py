"""Shared explainer machinery. Explainers are ABCs; the model boundary is a Protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from jev_xai.async_compat import run_sync
from jev_xai.explainers.context import ExplainContext
from jev_xai.model.client import ModelClient
from jev_xai.types import CostEnvelope


class ExplanationResult(BaseModel):
    explainer: str
    cost: CostEnvelope = Field(default_factory=CostEnvelope)
    noise_floor: float | None = None
    config_hash: str | None = None


class Explainer(ABC):
    """Internal base class. Callers depend on :meth:`explain`, not on a vendor SDK."""

    name: str

    @abstractmethod
    async def explain(
        self,
        client: ModelClient,
        instance: Mapping[str, Any],
        context: ExplainContext | None = None,
    ) -> ExplanationResult:
        """Produce an explanation for one instance."""

    def explain_sync(
        self,
        client: ModelClient,
        instance: Mapping[str, Any],
        context: ExplainContext | None = None,
    ) -> ExplanationResult:
        return run_sync(self.explain, client, instance, context)
