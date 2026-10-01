# Configuration

`JevXaiConfig` is the only configuration object. Precedence, later wins:

1. defaults
2. named profile (`quick`, `audit`, `ci-gate`)
3. project file (`jev-xai.toml`)
4. environment variables `JEV_XAI_*` with `__` for nesting (`JEV_XAI_MODEL__CONCURRENCY=4`)
5. CLI flags
6. explicit Python `overrides`

`config_hash` is blake2b of the canonical JSON of the resolved config. It is stored on the decision record and in the pack manifest.

## Knobs

| Group | Fields |
| --- | --- |
| model | `concurrency`, `batch_size`, `max_calls`, `timeout_s`, `retries`, `cache_mode` (`off` / `memory` / `cassette`) |
| ablation | `masking.structured`, `masking.text`, `feature_groups`, `text_span_granularity`, `max_spans` |
| counterfactual | `max_features_changed`, `grid_quantiles`, `max_candidates`, `distance_metric`, `immutable_features`, `allowed_ranges`, `categorical_values`, `numeric_grid`, `call_budget` |
| anchors | `precision`, `max_size`, `samples`, `coverage_samples`, `numeric_band`, `call_budget` |
| permutation | `repeats`, `call_budget` (per instance) |
| shap | `nsamples`, `call_budget` (optional extra) |
| lime | `num_features`, `num_samples`, `call_budget` (optional extra) |
| stability | `runs`, `top_k`, `rank_metric`, `failure_tolerance`, `score_weights` |
| reproducibility | `repeat_probe_runs`, `noise_floor_sigma_k`, `probability_tolerance`, `min_reproduction_rate`, `on_below_noise_floor` |
| evidence | `redaction_mode`, `redact_fields`, `max_inline_input_bytes`, `formats` |

::: jev_xai.config.schema.JevXaiConfig
