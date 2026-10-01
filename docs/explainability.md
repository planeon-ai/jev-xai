# Explainability

Order of evidence, strongest first: counterfactuals and ablation, then stability of those results. Attribution libraries are optional plugins.

## Ablation

Delta P = P(original label) − P(label after the mask). The masking strategy is stored on every row because it changes the number:

- structured fields: `drop_key`, `null`, `neutral`, `background`
- text: `deletion`, `mask_token` (`[MASK]`), `neutral_token`

Without a `FeatureSpec`, ablation masks the whole input once.

Rows with `|delta|` below the probe noise floor are flagged `below_noise_floor` and rendered grey in the HTML report. `on_below_noise_floor` is `flag`, `drop`, or `error`.

A mask that leaves the input unchanged is `noop`. `delta_p` is null and the model is not called. That row is not evidence the feature has no effect: the mask value was already the instance value. A boolean that is already false, under the default neutral mask, is the usual case. No-op rows are left out of the stability ranking and out of `ablation_top`.

## Counterfactuals

Algorithm: greedy coordinate descent over a per-feature quantile grid, then a sparsity polish that drops any change the decision does not need. Budgets: `max_calls` / `call_budget` and `max_features_changed`.

Plausibility in V1 means user-supplied ranges and categorical value sets. There is no density model. Immutable features are never changed.

`label` and `flipped` come from the search call. `confirmed_label` is a second `predict` of that same candidate with the cache bypassed. `replay_confirmed` is true only when that second label still flips. A budget miss leaves `confirmed_label` null and `replay_confirmed` false. `success_rate` counts only confirmed flips. `jev-xai gate` fails a pack that contains an unconfirmed flip. This is not a replay mode.

## Anchors

A greedy search builds a short rule that keeps the original label. Each predicate is either `eq` (boolean, categorical, text) or `within` a numeric band. Precision is the fraction of seeded samples, with those predicates held, that still predict the original label. Coverage is how often a fully random perturbation satisfies the rule, and that estimate does not call the model.

Search stops at `anchors.precision`, `max_size`, or `call_budget`. `sufficient` is true only when precision reached the threshold. Otherwise the best rule found so far is still returned.

Immutable features never appear. A feature whose declared domain has only the instance value cannot move the prediction, so it is not a candidate.

An audit pack writes `anchors.json` and `permutation.json`. Pass `--context` (or `context=` in Python) with a `FeatureSpec` to fill them. Without that, the files record the missing prerequisite and the rest of the pack is still written. A call-budget miss is recorded the same way.

## Permutation importance

For each mutable feature, `repeats` draws replace that feature with another value from its domain. Importance is the mean drop in P(original label). `label_flip_rate` is reported even when the model returns no probability; in that case `importance` is null.

`explain_dataset` averages those rows across instances. The call budget is per instance, not across the dataset. Immutable features are skipped.

A feature that cannot take another value, or that the budget never samples, is `unmeasured`. Its importance is null. A zero there would not mean the feature had no effect. Unmeasured rows are left out of the stability ranking.

## Optional SHAP and LIME

These are not core dependencies. `pip install jev-xai[shap]` and `pip install jev-xai[lime]` register `ShapExplainer` and `LimeExplainer`.

Both need tabular columns and a background: `background_rows`, or one `background` dict. A single background row is a coarse baseline and is named on `note`. Text fields are skipped. Sampled calls go to `model.predict` and are counted on `cost`; they are not written to the cassette.

KernelSHAP uses the global NumPy RNG. The adapter saves and restores that state around the call. LIME takes `random_state` from the config seed.

## stability_score_v1

Weights live on `StabilityConfig.score_weights` and must sum to 1. Defaults:

| Component | Weight | Definition |
| --- | ---: | --- |
| rank correlation | 0.40 | `((rho + 1) / 2)` where rho is mean pairwise Spearman or Kendall tau-b |
| feature overlap | 0.30 | mean pairwise Jaccard of the top-k features |
| attribution stability | 0.20 | `1 / (1 + mean variance of deltas)` |
| counterfactual consistency | 0.10 | fraction of runs whose changed-feature set equals the modal set |

Decision reproducibility is **not** inside this score. It is `reproduction_rate` on the replay probe.

Each stability run calls the model. Those calls do not read or write the client cache or the cassette, so an audit that already explained the decision cannot turn a warm cache into a perfect score. The calls still count toward `max_calls`.

`measured_explainer` names the explainer that was repeated. An audit pack repeats `AblationExplainer` only. That score is not evidence that the anchors, permutation, or counterfactuals in the same pack were stable.

Failed runs are counted. If `n_failed / runs` exceeds `failure_tolerance`, evaluation raises.

`assert_stable(result, min_score=0.8)` and `jev-xai gate` turn the score into a CI check. The gate walks nested records and audit packs. It fails when stability or reproduction rate is below the threshold, when evidence or behavioral replay `matched` is false, when a counterfactual flipped without `replay_confirmed`, or when the directory has none of those fields. Copies under `store/` are not scored again. To load the assertion helper as a pytest plugin, set `pytest_plugins = ["jev_xai.pytest_plugin"]` in your project. It is not auto-loaded, so it does not import jev-xai before coverage starts.
