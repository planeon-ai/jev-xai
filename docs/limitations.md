# Limitations

jev-xai does not:

- reveal hidden chain-of-thought
- prove causal mechanisms inside opaque models
- guarantee regulatory compliance
- guarantee fairness
- guarantee that an attribution score represents internal reasoning
- convert black-box models into inherently interpretable models

It provides behavioral evidence, reproducibility, counterfactual evidence, approximate attribution, anchors, permutation importance, and stability measurement.

SHAP and LIME are optional extras (`jev-xai[shap]`, `jev-xai[lime]`), not core dependencies. Out of this release: PDF packs, JavaScript dashboards, distributed execution, OpenTelemetry export, model registries, and framework-specific agent SDKs. Anchors, permutation importance, and the optional attribution adapters are behavioral evidence, not a proof of internal reasoning.
