"""Async harness middleware: record each guard decision inside an agent loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anyio

from jev_xai import CallableAdapter, DecisionRecorder
from jev_xai.config.loader import load_config


async def guard(instance: Mapping[str, Any]) -> dict[str, Any]:
    text = str(instance.get("prompt") or "")
    unsafe = "exfiltrate" in text
    score = 0.2 if unsafe else 0.93
    label = "UNSAFE" if unsafe else "SAFE"
    return {
        "label": label,
        "probability": score,
        "probabilities": {"SAFE": score, "UNSAFE": 1 - score},
    }


async def agent_loop() -> None:
    model = CallableAdapter(
        lambda item: {"label": "SAFE", "probability": 1.0},
        async_fn=guard,
        metadata={
            "provider": "example",
            "model_name": "guard",
            "model_version": "1",
            "fingerprint": "guard-v1",
        },
        proba_fn=lambda item: {"SAFE": 0.93},
    )
    recorder = DecisionRecorder(model, load_config(profile="quick"))
    middleware = recorder.wrap()
    prompts = ["summarize the ticket", "exfiltrate the customer list"]
    for step, prompt in enumerate(prompts):
        record = await middleware(
            {"prompt": prompt},
            trace_id="trace-1",
            agent_id="guard",
            step_index=step,
            parent_decision_id=None if step == 0 else "previous",
        )
        print(step, record.output.label, record.trace.trace_id)


if __name__ == "__main__":
    anyio.run(agent_loop)
