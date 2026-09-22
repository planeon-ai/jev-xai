"""Safety-classifier example used by the CLI and by readers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def predict(instance: Mapping[str, Any]) -> dict[str, Any]:
    text = str(instance.get("prompt") or instance.get("note") or "")
    verified = bool(instance.get("verified_user", False))
    amount = float(instance.get("transaction_amount") or 0)
    unsafe = any(token in text.lower() for token in ("ignore", "exfiltrate", "bypass"))
    score = 0.84
    if unsafe:
        score -= 0.5
    if not verified:
        score -= 0.15
    score -= min(amount / 40000, 0.2)
    score = max(0.01, min(0.99, score))
    label = "SAFE" if score >= 0.5 else "UNSAFE"
    return {
        "label": label,
        "probability": score,
        "probabilities": {"SAFE": score, "UNSAFE": 1 - score},
    }


def build_model() -> Any:
    from jev_xai import CallableAdapter

    return CallableAdapter(
        predict,
        metadata={
            "provider": "example",
            "model_name": "safety",
            "model_version": "1",
            "fingerprint": "example-safety-v1",
        },
        proba_fn=lambda item: predict(item)["probabilities"],
    )


if __name__ == "__main__":
    print(build_model().predict({"prompt": "hello", "verified_user": True}))
