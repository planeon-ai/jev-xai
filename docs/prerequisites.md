# Prerequisites

The library cannot explain an input it cannot see, and it cannot perturb a model it cannot call again.

| Prerequisite | If missing |
| --- | --- |
| Model re-invocation | ablation, counterfactuals, and stability stay off |
| Reachable input (not only a hash) | only evidence replay of the stored record |
| `FeatureSpec` | ablation is one whole-input mask |
| Calibrated probabilities | delta-P is omitted; label flips remain |
| Model fingerprint and a record corpus | cross-version replay stays off |

`hash_only` redaction and reachable inputs pull against each other. A team that stores only hashes is capped at tier 0. That is reported by `jev-xai doctor`, not discovered mid-loop. Engines raise `CapabilityError` with the missing prerequisite and how to supply it.

## Capture paths

- **Inline.** `DecisionRecorder` or its async `wrap()` middleware records the call and the cassette. This is the highest-fidelity path.
- **Post-hoc.** `JsonlSource` / `jev-xai import` reads a JSONL decision log from an existing harness. Imported records are tier 0 until a live model is attached. OpenTelemetry and vendor trace backends are later plugins on `jev_xai.sources`.
