# Changelog

## Unreleased

- Anchors: greedy high-precision rules over `FeatureSpec` predicates, with precision from seeded model samples and coverage that does not call the model.
- Permutation importance: local sensitivity and a dataset average. Importance is the mean drop in P(original label). Label flip rate is still reported when the model returns no probability.
- Optional SHAP and LIME adapters (`jev-xai[shap]`, `jev-xai[lime]`). They are not core dependencies. Sampled calls are counted and are not written to the cassette.
- Replay mode `cross` re-invokes one record on the supplied model. `counterfactual` is rejected. Oversized inputs written to an evidence store can be loaded back for replay, diff, and audit.
- Ablation marks a mask that does not change the input as `noop` and does not call the model for it. `jev-xai gate` walks nested packs, fails on an unmatched replay, and fails closed when a directory has no stability, reproduction, or replay evidence.
- Audit packs include anchors and permutation when a `FeatureSpec` is supplied. Without one, or when the call budget is exhausted, those members record why they were skipped. Reports show `replay_confirmed` on counterfactual candidates.
- `replay_confirmed` is an uncached second call of the finished counterfactual. `success_rate` counts only that confirmation. `jev-xai gate` fails when a candidate flipped and the confirmation did not.

## 0.1.0a0

Initial alpha of the accountability layer: configurable ablation and counterfactual evidence, cassette-backed evidence replay, live behavioral reproduction, cross-version diff, stability scoring (`stability_score_v1`), content-addressed audit packs, and a CLI.

The `jev-xai` name on PyPI is claimed by publishing this alpha. Publishing uses the Trusted Publishing workflow in `.github/workflows/release.yml` (tag `v*`). No long-lived PyPI token is stored in the repository.
