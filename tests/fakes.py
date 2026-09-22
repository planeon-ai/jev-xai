"""Deterministic and scripted models for tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _clip(value: float) -> float:
    return max(0.01, min(0.99, value))


class ThresholdModel:
    """SAFE when verification outweighs sanctions and amount."""

    def predict(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        amount = float(instance.get("transaction_amount") or 0)
        verified = bool(instance.get("verified_user", False))
        sanctions = bool(instance.get("sanctions_match", False))
        note = str(instance.get("note") or "")
        score = 0.55
        if verified:
            score += 0.3
        if sanctions:
            score -= 0.4
        score -= min(amount / 50000, 0.25)
        if "bypass" in note:
            score -= 0.2
        score = _clip(score)
        label = "SAFE" if score >= 0.5 else "UNSAFE"
        return {
            "label": label,
            "probability": score,
            "probabilities": {"SAFE": score, "UNSAFE": 1 - score},
        }

    def predict_proba(self, instance: Mapping[str, Any]) -> Mapping[str, float]:
        probs: Mapping[str, float] = self.predict(instance)["probabilities"]
        return probs

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "toy",
            "model_name": "threshold",
            "model_version": "1",
            "fingerprint": "toy-v1",
        }


class FlippedModel(ThresholdModel):
    def predict(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(super().predict(instance))
        result["label"] = "UNSAFE" if result["label"] == "SAFE" else "SAFE"
        return result

    def metadata(self) -> dict[str, Any]:
        meta = dict(super().metadata())
        meta["model_version"] = "2"
        meta["fingerprint"] = "toy-v2"
        return meta


class LabelOnlyModel:
    def predict(self, instance: Mapping[str, Any]) -> str:
        return "SAFE" if instance.get("ok") else "UNSAFE"

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "toy",
            "model_name": "labels",
            "model_version": "1",
            "fingerprint": "labels-v1",
        }


class NoPredictModel:
    def metadata(self) -> dict[str, str]:
        return {"provider": "none", "model_name": "none", "model_version": "0"}


class ScriptedNoise:
    """First five calls on the zero input follow a fixed noisy script."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        self.calls += 1
        tiny = float(instance.get("tiny") or 0)
        risk = float(instance.get("risk") or 0)
        if self.calls <= 5 and tiny == 0 and risk == 0:
            probability = [0.75, 0.77, 0.20, 0.74, 0.76][self.calls - 1]
        else:
            probability = 0.75 + 0.0001 * tiny - 0.4 * risk
        probability = _clip(probability)
        label = "SAFE" if probability >= 0.5 else "UNSAFE"
        return {
            "label": label,
            "probability": probability,
            "probabilities": {"SAFE": probability, "UNSAFE": 1 - probability},
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "toy",
            "model_name": "noise",
            "model_version": "1",
            "fingerprint": "noise-v1",
        }


class BatchModel:
    def __init__(self) -> None:
        self.batch_calls = 0

    def predict(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        del instance
        return {"label": "SAFE", "probability": 0.6, "probabilities": {"SAFE": 0.6}}

    def predict_batch(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        self.batch_calls += 1
        return [self.predict(row) for row in rows]

    def supports_batch(self) -> bool:
        return True

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "toy",
            "model_name": "batch",
            "model_version": "1",
            "fingerprint": "batch-v1",
        }


class AsyncEcho:
    def predict(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        del instance
        return {"label": "SAFE", "probability": 0.8, "probabilities": {"SAFE": 0.8}}

    async def apredict(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        return self.predict(instance)

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "toy",
            "model_name": "async",
            "model_version": "1",
            "fingerprint": "async-v1",
        }


class RetryModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        del instance
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("temporary")
        return {"label": "SAFE", "probability": 0.7}

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "toy",
            "model_name": "retry",
            "model_version": "1",
            "fingerprint": "retry-v1",
        }


class SlowModel:
    def predict(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        import time

        del instance
        time.sleep(0.3)
        return {"label": "SAFE", "probability": 0.5}

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "toy",
            "model_name": "slow",
            "model_version": "1",
            "fingerprint": "slow-v1",
        }
