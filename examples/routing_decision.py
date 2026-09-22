"""Routing example: agent A, agent B, or a human."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def predict(instance: Mapping[str, Any]) -> dict[str, Any]:
    risk = float(instance.get("risk_score") or 0)
    if risk >= 0.7:
        label = "human"
    elif risk >= 0.4:
        label = "agent_b"
    else:
        label = "agent_a"
    score = max(0.01, min(0.99, 1 - risk))
    return {"label": label, "probability": score, "probabilities": {label: score}}


def build_model() -> Any:
    from jev_xai import CallableAdapter

    return CallableAdapter(
        predict,
        metadata={
            "provider": "example",
            "model_name": "router",
            "model_version": "1",
            "fingerprint": "example-router-v1",
        },
        proba_fn=lambda item: predict(item)["probabilities"],
    )


if __name__ == "__main__":
    print(build_model().predict({"risk_score": 0.8}))
