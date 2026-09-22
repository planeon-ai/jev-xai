"""Property checks for hashing and counterfactual constraints."""

from __future__ import annotations

import anyio
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.fakes import ThresholdModel

from jev_xai.canonical import content_hash
from jev_xai.config.loader import load_config
from jev_xai.explainers.context import ExplainContext, FeatureSpec
from jev_xai.explainers.counterfactual import CounterfactualExplainer
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=6, alphabet="ab"),
        st.integers(min_value=-5, max_value=5),
        max_size=4,
    )
)
@settings(max_examples=25)
def test_hash_stable_under_reordering(payload: dict[str, int]) -> None:
    flipped = dict(reversed(list(payload.items())))
    assert content_hash(payload) == content_hash(flipped)


def test_unicode_nfc_property() -> None:
    assert content_hash({"name": "é"}) == content_hash({"name": "e\u0301"})


def test_immutable_feature_never_changes() -> None:
    config = load_config(
        overrides={
            "reproducibility": {"repeat_probe_runs": 0},
            "model": {"retries": 0},
            "counterfactual": {
                "immutable_features": ["sanctions_match"],
                "call_budget": 20,
                "max_candidates": 1,
                "max_features_changed": 2,
                "allowed_ranges": {"transaction_amount": [0, 50000]},
            },
        }
    )
    model = ThresholdModel()
    client = ModelClient(model, config.model, fingerprint=model_fingerprint(model), seed=0)
    context = ExplainContext(
        features=[
            FeatureSpec(name="verified_user", kind="boolean"),
            FeatureSpec(name="sanctions_match", kind="boolean"),
            FeatureSpec(name="transaction_amount", kind="numeric", allowed_range=(0, 50000)),
        ],
        target_label="SAFE",
    )
    result = anyio.run(
        CounterfactualExplainer(config).explain,
        client,
        {"verified_user": False, "sanctions_match": True, "transaction_amount": 40000},
        context,
    )
    for candidate in result.candidates:
        assert all(change.feature != "sanctions_match" for change in candidate.changes)
        if candidate.flipped:
            assert candidate.label == "SAFE"
