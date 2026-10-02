# Architecture

The import name is `jev_xai`. The distribution name is `jev-xai`. The roadmap text that says `jev-explain` / `jev_explain` means this package. Do not rename it back.

```text
JevXaiConfig
    -> ModelClient (batching, cassette, cost)
        -> AblationExplainer
        -> CounterfactualExplainer
        -> AnchorExplainer
        -> PermutationExplainer
        -> ShapExplainer (optional extra)
        -> LimeExplainer (optional extra)
            -> StabilityEvaluator
DecisionRecorder -> EvidenceStore -> audit pack
ReplayEngine: evidence replay | behavioral reproduction | cross-version diff
```

`DecisionModel` is a Protocol at the user boundary. Explainers are ABCs so they share cost envelopes and seed derivation. Every explainer talks only to `ModelClient`.

Plugins:

- `jev_xai.explainers`
- `jev_xai.adapters`
- `jev_xai.sources`

SHAP and LIME are optional in-tree extras (`jev-xai[shap]`, `jev-xai[lime]`). OpenTelemetry importers stay out-of-tree plugins against those groups.

## Sync and async

Engines are async. Sync methods call `run_sync`, which raises `JevXaiUsageError` if an event loop is already running. Inside a harness, await `client.predict` or the recorder middleware.

## Cassette

Calls are keyed by `blake2b(canonical(fingerprint, input))`. Persisting the cassette makes evidence replay free of model calls and makes replay tests runnable offline.
