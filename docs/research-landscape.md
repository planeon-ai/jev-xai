# Research landscape

Phase 0 exit gates. These are decisions, not open questions.

## JEV API surface

No JEV SDK is published in this repository or as an install dependency. Hosted JEV-like models are treated as black boxes with this contract:

- `predict(x)` returns a label, a `(label, probability)` pair, or `{label, probability, probabilities}`.
- `predict_proba` is optional. Without it, explanations report label flips and omit delta-P.
- `apredict` / `predict_batch` are optional. The client uses them when present and otherwise threads sync calls with a concurrency limit.
- `metadata()` should include `provider`, `model_name`, `model_version`, and ideally `fingerprint`.

`JEVAdapter` wraps a caller-supplied client. It does not import a vendor SDK. When an official SDK exists, a new adapter can register on the `jev_xai.adapters` entry point without changing explainers.

## Determinism gate

**Chosen branch: hosted JEV-like models are nondeterministic unless a live probe says otherwise.**

- Evidence replay reads the cassette and makes zero model calls. It proves what was recorded.
- Behavioral reproduction re-invokes the model. Labels must match. Probabilities must fall within `probability_tolerance` (default `1e-6`) only when the probe measures a zero sigma. Otherwise the probe's noise floor qualifies every delta.
- The probe runs again at decision time. A Phase 0 note does not freeze a provider's behavior.

## Query-cost methodology

`benchmarks/run.py` calls each reference task, records `n_model_calls`, `wall_ms`, sparsity, flip rate, and stability, and writes JSON. Defaults:

- ablation scans each declared feature once, plus at most `max_spans` text spans
- counterfactual search stops at `call_budget` (200 in the audit profile, 30 in quick)
- stability runs are 20 in audit, 2 in quick, 3 in ci-gate

CI runs `python benchmarks/run.py --quick`. Full runs are manual. The qualitative comparison table in the original spec is not copied into the docs as fact; measured rows replace it.

## Explainability libraries

DiCE, Alibi, and CARLA are not core dependencies. Ablation, greedy counterfactual search, anchors, and permutation importance are in-tree so masking policy, the noise floor, async calls, and call budgets stay under one config object. SHAP and LIME stay out of the core install. The optional extras `jev-xai[shap]` and `jev-xai[lime]` are thin adapters on the same `ModelClient` budget and `FeatureSpec` context. Third-party engines belong on `jev_xai.explainers` as separate packages.

## Adapter decision

Ship `DecisionModel` as a Protocol, `CallableAdapter` for harness callables, and `JEVAdapter` as the thin wrapper above. Do not block the library on an unpublished SDK.
