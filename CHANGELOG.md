# Changelog

## Unreleased

- Anchors: greedy high-precision rules over `FeatureSpec` predicates, with precision from seeded model samples and coverage that does not call the model.
- Permutation importance: local sensitivity and a dataset average. Importance is the mean drop in P(original label). Label flip rate is still reported when the model returns no probability.
- Optional SHAP and LIME adapters (`jev-xai[shap]`, `jev-xai[lime]`). They are not core dependencies. Sampled calls are counted and are not written to the cassette.
- Replay mode `cross` re-invokes one record on the supplied model. `counterfactual` is rejected. Oversized inputs written to an evidence store can be loaded back for replay, diff, and audit.

## 0.1.0a0

Initial alpha of the accountability layer: configurable ablation and counterfactual evidence, cassette-backed evidence replay, live behavioral reproduction, cross-version diff, stability scoring (`stability_score_v1`), content-addressed audit packs, and a CLI.

The `jev-xai` name on PyPI is claimed by publishing this alpha. Publishing uses the Trusted Publishing workflow in `.github/workflows/release.yml` (tag `v*`). No long-lived PyPI token is stored in the repository.
