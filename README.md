# jev-xai

**jev-xai is an open-source accountability layer for JEV-like probabilistic decision models. Reproduce, challenge, explain, and audit black-box decisions — every step configurable, every claim backed by replayable evidence.**

```text
Decision -> Reproduce -> Perturb -> Counterfactually challenge -> Explain -> Measure stability -> Generate evidence
```

This is not "SHAP for JEV". SHAP and LIME stay out of the core package. The library measures what a model *does* when inputs change, records the exact conditions of a decision, and tells you when an explanation is too unstable to trust.

Framework-specific agent SDKs are a later release. V1 is the layer any harness can call.

## Install

```bash
pip install jev-xai                 # pydantic, numpy, anyio
pip install "jev-xai[cli]"          # typer and rich
pip install "jev-xai[dev,cli,stats]"
```

`0.1.0a0` is the name-claim alpha. Publishing uses PyPI Trusted Publishing (`.github/workflows/release.yml`) on a `v*` tag. The repository does not store a PyPI token.

## Telemetry

jev-xai sends no telemetry and opens no network connection except the model calls you configure. `logging.getLogger("jev_xai")` carries a `NullHandler`. Raw decision inputs are not logged at INFO.

## Quick use

```python
from jev_xai import CallableAdapter, DecisionRecorder, load_config

def predict(item):
    score = 0.9 if item.get("verified_user") else 0.2
    label = "SAFE" if score >= 0.5 else "UNSAFE"
    return {"label": label, "probability": score, "probabilities": {"SAFE": score, "UNSAFE": 1 - score}}

model = CallableAdapter(predict, metadata={"provider": "local", "model_name": "guard", "model_version": "1"})
record = DecisionRecorder(model, load_config(profile="quick")).run_sync({"verified_user": True})
```

Profiles are `quick`, `audit`, and `ci-gate`. Resolved configuration is hashed into every record (`config_hash`), so two explanations are comparable only when that hash matches.

## API

The contract is [docs/api.md](https://github.com/planeon-ai/jev-xai/blob/main/docs/api.md). It lists every name in `jev_xai.__all__` with inputs and outputs, and a test fails if a public name is missing. The call path:

| Call | Input | Output |
| --- | --- | --- |
| `load_config` | profile, file, env, overrides | `JevXaiConfig` |
| `diagnose` | model and host flags | `Diagnosis` (tier 0–3) |
| `DecisionRecorder.run` | decision input mapping | `DecisionRecord` |
| `ReplayEngine.replay` | record, mode `exact` or `current` | `ReplayResult` |
| `AblationExplainer.explain` | model client, input, `ExplainContext` | `AblationResult` (`delta_p` per field) |
| `CounterfactualExplainer.explain` | same, plus allowed ranges | `CounterfactualResult` |
| `ReproducibilityProbe.measure` | client, input, reference label | `ProbeSummary` |
| `StabilityEvaluator.evaluate` | explainer, model, input | `StabilityResult` |
| `build_audit_pack` | record, model, config, directory | Merkle pack directory |

## What the host must provide

`jev-xai doctor` reports a capability tier:

| Tier | Name | Unlocks |
| --- | --- | --- |
| 0 | record only | cassette evidence replay |
| 1 | behavioral | ablation and counterfactuals (label flips) |
| 2 | graded | delta-P and stability |
| 3 | longitudinal | cross-version replay |

Reaching tier 1 requires model re-invocation and the actual decision input. A log that stores only hashes stays at tier 0. See `docs/prerequisites.md`.

## Non-goals

jev-xai does not reveal hidden chain-of-thought, prove causal mechanisms inside opaque models, guarantee regulatory compliance, guarantee fairness, guarantee that an attribution score represents internal reasoning, or convert black-box models into inherently interpretable models. It provides behavioral evidence, reproducibility, counterfactual evidence, approximate attribution, and stability measurement.

The same text is embedded in every evidence-pack manifest.

## License

Apache-2.0.
