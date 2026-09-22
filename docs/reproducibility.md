# Reproducibility

Two claims, labeled differently in `replay.json`:

| Claim | Mode | What it proves |
| --- | --- | --- |
| `evidence_replay` | `exact` | The cassette (or the stored output) is what was recorded. Zero model calls. |
| `behavioral_reproduction` | `current` | The live model still produces that label, and the probability within tolerance. |

Cross-version replay is `jev-xai diff --records ./records --model module:factory`. It reports decision flips. If `config_hash` differs from the recorded config, the report says so instead of pretending the comparison is isolated.

## Repeat probe

`repeat_probe_runs` fresh calls (cache bypassed) produce:

- `reproduction_rate`: fraction matching the recorded label
- `probability_sigma`: sample standard deviation
- `noise_floor`: `noise_floor_sigma_k * probability_sigma` (k defaults to 2)

## Mismatch reasons

When behavioral reproduction fails, the reason is one of:

- `model_version_changed`
- `environment_changed`
- `config_changed`
- `input_nonstationary` (keys such as `timestamp`, `now`, `date`)
- `model_nondeterministic`

## Seeds

One root seed is stored on the record. Perturbation RNGs come from `numpy.random.SeedSequence(root).spawn()`. The global NumPy RNG is not used. Parallel and sequential stability runs with the same seeds match.
