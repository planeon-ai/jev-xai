# Explainability

Order of evidence, strongest first: counterfactuals and ablation, then stability of those results. Attribution libraries are optional plugins.

## Ablation

Delta P = P(original label) − P(label after the mask). The masking strategy is stored on every row because it changes the number:

- structured fields: `drop_key`, `null`, `neutral`, `background`
- text: `deletion`, `mask_token` (`[MASK]`), `neutral_token`

Without a `FeatureSpec`, ablation masks the whole input once.

Rows with `|delta|` below the probe noise floor are flagged `below_noise_floor` and rendered grey in the HTML report. `on_below_noise_floor` is `flag`, `drop`, or `error`.

## Counterfactuals

Algorithm: greedy coordinate descent over a per-feature quantile grid, then a sparsity polish that drops any change the decision does not need. Budgets: `max_calls` / `call_budget` and `max_features_changed`.

Plausibility in V1 means user-supplied ranges and categorical value sets. There is no density model. Immutable features are never changed.

## Anchors

A greedy search builds a short rule that keeps the original label. Each predicate is either `eq` (boolean, categorical, text) or `within` a numeric band. Precision is the fraction of seeded samples, with those predicates held, that still predict the original label. Coverage is how often a fully random perturbation satisfies the rule, and that estimate does not call the model.

Search stops at `anchors.precision`, `max_size`, or `call_budget`. `sufficient` is true only when precision reached the threshold. Otherwise the best rule found so far is still returned.

Immutable features never appear. A feature whose declared domain has only the instance value cannot move the prediction, so it is not a candidate.

## Permutation importance

For each mutable feature, `repeats` draws replace that feature with another value from its domain. Importance is the mean drop in P(original label). `label_flip_rate` is reported even when the model returns no probability; in that case `importance` is null.

`explain_dataset` averages those rows across instances. The call budget is per instance, not across the dataset. Immutable features are skipped.

## stability_score_v1

Weights live on `StabilityConfig.score_weights` and must sum to 1. Defaults:

| Component | Weight | Definition |
| --- | ---: | --- |
| rank correlation | 0.40 | `((rho + 1) / 2)` where rho is mean pairwise Spearman or Kendall tau-b |
| feature overlap | 0.30 | mean pairwise Jaccard of the top-k features |
| attribution stability | 0.20 | `1 / (1 + mean variance of deltas)` |
| counterfactual consistency | 0.10 | fraction of runs whose changed-feature set equals the modal set |

Decision reproducibility is **not** inside this score. It is `reproduction_rate` on the replay probe.

Failed runs are counted. If `n_failed / runs` exceeds `failure_tolerance`, evaluation raises.

`assert_stable(result, min_score=0.8)` and `jev-xai gate` turn the score into a CI check. To load the assertion helper as a pytest plugin, set `pytest_plugins = ["jev_xai.pytest_plugin"]` in your project. It is not auto-loaded, so it does not import jev-xai before coverage starts.
