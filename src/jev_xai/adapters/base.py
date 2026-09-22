"""DecisionModel is a Protocol: user objects do not inherit from jev-xai."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DecisionModel(Protocol):
    """Structural type for a decision model.

    Optional members, detected with ``hasattr`` rather than declared here:
    ``apredict``, ``apredict_proba``, ``predict_batch``, ``apredict_batch``.
    """

    def predict(self, x: Mapping[str, Any]) -> Any:
        """Return a label, a ``(label, probability)`` pair, or a prediction dict."""

    def predict_proba(self, x: Mapping[str, Any]) -> Mapping[str, float]:
        """Return class probabilities. Optional in practice; probed at runtime."""

    def metadata(self) -> Mapping[str, Any]:
        """Return provider, model_name, model_version, and optionally fingerprint."""
