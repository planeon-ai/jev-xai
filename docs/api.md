# API

The public surface is `jev_xai.__all__`. This page is the contract: each callable's inputs, the object it returns, and the fields on that object. Signatures at the bottom are rendered from the source. `tests/golden/test_api_doc.py` fails if a public name is missing from this file.

A model call may return a label string, a `(label, probability)` pair, or a dict `{"label", "probability", "probabilities"}`. `ModelClient` normalizes all three into a `Prediction`.

## Constants

| Name | Type | Value |
| --- | --- | --- |
| `__version__` | `str` | package version, currently `0.1.0a0` |
| `SCHEMA_VERSION` | `str` | `"1.0.0"`. Readers accept any `1.x.x` record |
| `LIMITATIONS` | `str` | non-goals copied into every pack manifest |

## Configure

### `load_config(*, profile=None, path=None, environ=None, cli=None, overrides=None) -> JevXaiConfig`

Later sources win: defaults, named profile (`quick`, `audit`, `ci-gate`), project TOML `path`, `JEV_XAI_*` environment (`__` nests keys), `cli` dict, then `overrides`.

**Returns** `JevXaiConfig`: `seed`, `threshold`, `policy_version`, and nested `model`, `ablation`, `counterfactual`, `stability`, `reproducibility`, `replay`, `evidence`.

### `config_hash(config: JevXaiConfig) -> str`

blake2b of the canonical JSON of `config`. Stored on the decision record and the pack manifest.

## Adapt a model

The host object does not subclass anything. It matches `DecisionModel` if it has:

| Method | Input | Output |
| --- | --- | --- |
| `predict` | `x: Mapping` decision input | label, `(label, probability)`, or prediction dict |
| `predict_proba` | `x: Mapping` | `Mapping[str, float]` class probabilities. Optional |
| `metadata` | none | `provider`, `model_name`, `model_version`, optional `fingerprint` and `artifact_hash` |

Optional, detected with `hasattr`: `apredict`, `apredict_proba`, `predict_batch`, `apredict_batch`.

### `CallableAdapter(predict_fn, *, metadata=None, proba_fn=None, async_fn=None, batch_fn=None, async_batch_fn=None)`

Wraps a harness function.

| Method | Input | Output |
| --- | --- | --- |
| `predict` | `x` | whatever `predict_fn` returns |
| `predict_proba` | `x` | `proba_fn(x)`, or `CapabilityError` if `proba_fn` was omitted |
| `metadata` | none | the metadata mapping, with a derived `fingerprint` when one was not passed |
| `supports_proba` / `supports_batch` | none | `bool` |
| `apredict` | `x` | `async_fn(x)`, else `predict_fn(x)` |
| `predict_batch` / `apredict_batch` | sequence of inputs | sequence of predictions |

### `JEVAdapter(client, *, provider="jev", model_name="jev", model_version="unspecified", artifact_hash=None)`

Wraps a caller-supplied JEV-like client. The client must be callable or expose `predict`. No vendor SDK is imported. Same methods as `CallableAdapter`. `JevXaiUsageError` if the client has neither.

## Check what the host can support

### `diagnose(model=None, *, input_reachable=True, context=None, has_corpus=False, source_only=False) -> Diagnosis`

**Returns** `Diagnosis`:

| Field | Meaning |
| --- | --- |
| `tier` | `0` record only, `1` behavioral, `2` graded, `3` longitudinal |
| `tier_name` | `record_only`, `behavioral`, `graded`, `longitudinal` |
| `prerequisites` | bools for reinvocation, reachable input, feature spec, probabilities, fingerprint, corpus |
| `explainers` | list of `ExplainerAvailability` (`name`, `enabled`, `missing`, `fix`, `note`) |
| `notes` | human-readable gaps, such as missing `FeatureSpec` |

## Record

### `DecisionRecorder(model, config=None, *, cassette=None)`

| Method | Input | Output |
| --- | --- | --- |
| `run` | `instance: Mapping`. Optional `trace: TraceInfo`, `decision_id: str`, `timestamp: str` | `DecisionRecord` |
| `run_sync` | same arguments | `DecisionRecord`. Raises `JevXaiUsageError` inside a running event loop |
| `wrap` | none | `async middleware(instance, **trace) -> DecisionRecord` |

`run` calls the model once, writes the cassette entry, and when `repeat_probe_runs > 0` measures reproducibility before returning.

**`DecisionRecord` fields:** `schema_version`, `decision_id`, `timestamp`, `model` (`provider`, `model_name`, `model_version`, `artifact_hash`, `fingerprint`), `input`, `input_hash`, `input_reachable`, `input_externalized`, `output` (`label`, `probability`, `probabilities`), `threshold`, `policy_version`, `runtime`, `explainer_config` (includes `config_hash`), `explanations`, `cost` (`n_model_calls`, `cache_hits`, `wall_ms`), `trace` (`trace_id`, `span_id`, `agent_id`, `step_index`, `parent_decision_id`), `cassette_key`, `reproducibility` (`ProbeSummary` or null).

## Replay

### `ReplayEngine(config, *, cassette=None).replay(record, *, model=None, mode=None) -> ReplayResult`

`mode` defaults to `config.replay.mode`.

| `mode` | Requires | Claim on the result |
| --- | --- | --- |
| `exact` | cassette or the stored output | `evidence_replay`. Zero live model calls |
| `current` | live `model` and a reachable dict input | `behavioral_reproduction` |

`cross` is a corpus operation: use `cross_version_diff`. `counterfactual` confirmation is `replay_confirmed` on each candidate, not a replay mode.

**`ReplayResult` fields:** `claim`, `mode`, `matched`, `mismatch_reason`, `original_label`, `replay_label`, `original_probability`, `replay_probability`, `cost`, `config_hash_matches`, `note`.

`mismatch_reason` is one of `model_version_changed`, `environment_changed`, `config_changed`, `input_nonstationary`, `model_nondeterministic`, or null when the replay matched.

### `cross_version_diff(records, model, config) -> DiffReport`

Re-invokes `model` on every reachable record.

**Returns** `n_records`, `n_flipped`, `rows` (`decision_id`, `original_label`, `replay_label`, `flipped`, `mismatch_reason`, `config_hash_matches`, probabilities), `config_hash_warning`, `from_fingerprint`, `to_fingerprint`.

## Explain

Shared input for every explainer: a `ModelClient`, the decision `instance: Mapping`, and an optional `ExplainContext`.

**`FeatureSpec`:** `name`, `kind` (`numeric`, `categorical`, `boolean`, `text`), `mutable`, `allowed_values`, `allowed_range`, `baseline`.

**`ExplainContext`:** `features`, `groups` (name to feature names), `target_label` (desired counterfactual label), `background`, `text_field`. `feature(name) -> FeatureSpec | None`.

### `AblationExplainer(config).explain(client, instance, context=None) -> AblationResult`

**Returns** `rows` and `suppressed_rows` of `AblationRow`, plus `masking_policy`.

**`AblationRow`:** `feature`, `scope` (`single`, `group`, `text_span`, `whole_input`), `masking_strategy`, `baseline_probability`, `ablated_probability`, `delta_p` (P(original label) minus P(ablated)), `label_flipped`, `below_noise_floor`, `original_label`, `ablated_label`.

### `CounterfactualExplainer(config, *, seed=None).explain(client, instance, context=None) -> CounterfactualResult`

Search stays inside `allowed_values`, `allowed_range`, and `config.counterfactual`. Immutable features are not changed.

**Returns** `original_label`, `desired_label`, `candidates`, `success_rate`, `algorithm`.

**`CounterfactualCandidate`:** `changes` (`feature`, `before`, `after`), `label`, `probability`, `distance`, `sparsity`, `margin`, `flipped`, `below_noise_floor`, `replay_confirmed`.

### `ReproducibilityProbe(config).measure(client, instance, *, reference_label) -> ProbeSummary`

Fresh calls (`use_cache=False`). Sets `client.noise_floor`.

**Returns** `runs`, `reproduction_rate`, `probability_sigma`, `noise_floor`, `labels`, `probabilities`.

### `StabilityEvaluator(config).evaluate(explainer, model, instance, *, runs=None, context=None, client=None) -> StabilityResult`

`evaluate_sequential` is the same measurement without a task group. `explain_with_stability(explainer, model, instance, config, *, runs=None, context=None)` is the short form and returns the same object.

**`StabilityResult`:** `runs`, `n_failed_runs`, `failures`, `rank_correlation`, `feature_overlap`, `attribution_variance`, `counterfactual_consistency`, `stability_score` (`stability_score_v1`), `formula`, `explanations`.

### `assert_stable(result: StabilityResult, min_score=0.8) -> None`

Raises `AssertionError` when `stability_score` is below `min_score`.

## Call the model

### `ModelClient(model, config: ModelClientConfig, *, fingerprint, seed=0, cassette=None)`

| Method | Input | Output |
| --- | --- | --- |
| `predict` | `instance`, `use_cache=True` | `Prediction` (`label`, `probability`, `probabilities`) |
| `predict_sync` | `instance` | `Prediction`. Raises `JevXaiUsageError` inside a running loop |
| `predict_many` | sequence of instances | `list[Prediction]` |
| `cache_key` | `instance` | cassette/cache key |
| `cost` | none | `CostEnvelope` |
| `force_store` | key, `Prediction` | none. Writes the cassette even when cache mode is off |

`max_calls` raises `BudgetExceededError`. Timeout and exhausted retries raise `ModelCallError`.

## Audit

### `build_audit_pack(record, model, config, directory, *, context=None) -> Path`

Requires `record.input` to be a dict. Runs ablation, counterfactual search, stability (at most 3 runs), evidence replay, and behavioral replay. Writes `decision.json`, `explanation.json`, `ablation.json`, `counterfactuals.json`, `stability.json`, `replay.json`, `manifest.json`, `report.md`, and `report.html` under `directory`. Returns that directory. `manifest.json` carries `schema_version`, `jev_xai_version`, `config_hash`, member hashes, `merkle_root`, and `LIMITATIONS`.

## Errors

| Exception | When |
| --- | --- |
| `JevXaiError` | base class |
| `JevXaiUsageError` | sync API called inside a running loop, or a client the adapter cannot bind |
| `CapabilityError` | a prerequisite is missing. Fields: `prerequisite`, `fix` |
| `ModelCallError` | the model call failed, timed out, or returned an unrecognized prediction |
| `ReplayMismatchError` | behavioral replay was asked for without a model or a reachable input, or `raise_on_mismatch` saw `matched=False`. Field: `reason` |
| `SchemaVersionError` | `schema_version` is not `1.x.x` |
| `BudgetExceededError` | `model.max_calls` was hit |

## Command line

Install `jev-xai[cli]`. The console script is `jev-xai`. Global options on the commands that load config: `--profile`, `--config`.

| Command | Input | Output |
| --- | --- | --- |
| `doctor` | `--model module:callable`, optional `--source` JSONL, `--format` | `Diagnosis` as json, md, html, or table |
| `import` | JSONL path, `--out` directory | one `DecisionRecord` JSON per decision, plus the tier name |
| `record` | `--model`, `--input` JSON object, `--out`, optional `--cassette` | path of the written `DecisionRecord` |
| `explain` | `--model`, `--input`, `--explainer` `ablation` or `counterfactual` | `AblationResult` or `CounterfactualResult` |
| `replay` | `decision.json`, `--mode` `exact` or `current`, optional `--model` and `--cassette` | `ReplayResult` |
| `diff` | `--records` directory of JSON, `--model` | `DiffReport` |
| `audit` | `decision.json`, `--model`, `--out` | pack directory. Prints `verified` and `config_hash` |
| `gate` | `--records`, `--min-stability`, `--min-reproduction-rate` | exit 0, or exit 1 with the failing files |
| `bench` | `--out`, `--quick` | JSON metrics path |
| `plugins list` | `--format` | entry points in `jev_xai.explainers`, `jev_xai.adapters`, `jev_xai.sources` |

## Signatures

::: jev_xai.config.loader.load_config

::: jev_xai.config.loader.config_hash

::: jev_xai.adapters.callable.CallableAdapter

::: jev_xai.adapters.jev.JEVAdapter

::: jev_xai.adapters.capabilities.diagnose

::: jev_xai.adapters.capabilities.Diagnosis

::: jev_xai.replay.recorder.DecisionRecorder

::: jev_xai.evidence.schema.DecisionRecord

::: jev_xai.replay.replay.ReplayEngine

::: jev_xai.replay.replay.ReplayResult

::: jev_xai.replay.diff.cross_version_diff

::: jev_xai.explainers.ablation.AblationExplainer

::: jev_xai.explainers.counterfactual.CounterfactualExplainer

::: jev_xai.replay.repeatability.ReproducibilityProbe

::: jev_xai.stability.evaluator.StabilityEvaluator

::: jev_xai.stability.evaluator.explain_with_stability

::: jev_xai.stability.evaluator.assert_stable

::: jev_xai.model.client.ModelClient

::: jev_xai.evidence.audit.build_audit_pack
