"""Anchors and permutation importance on the threshold fixture."""

from __future__ import annotations

import anyio
import pytest
from tests.fakes import LabelOnlyModel, ThresholdModel

from jev_xai.adapters.capabilities import diagnose
from jev_xai.config.loader import load_config
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import CapabilityError, JevXaiUsageError
from jev_xai.explainers.anchors import AnchorExplainer
from jev_xai.explainers.context import ExplainContext, FeatureSpec
from jev_xai.explainers.permutation import PermutationExplainer
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint
from jev_xai.stability.evaluator import StabilityEvaluator

FEATURES = ExplainContext(
    features=[
        FeatureSpec(name="customer_id", kind="categorical", mutable=False, allowed_values=["acme"]),
        FeatureSpec(name="verified_user", kind="boolean"),
        FeatureSpec(name="sanctions_match", kind="boolean"),
        FeatureSpec(
            name="transaction_amount",
            kind="numeric",
            allowed_range=(0.0, 50000.0),
        ),
        FeatureSpec(name="note", kind="text"),
        FeatureSpec(name="channel", kind="categorical", allowed_values=["web", "pos"]),
        FeatureSpec(name="score_hint", kind="numeric"),
        FeatureSpec(name="flagged", kind="boolean", allowed_values=[False]),
    ]
)


def _cfg(**overrides: object) -> JevXaiConfig:
    base: dict[str, object] = {
        "seed": 7,
        "model": {"retries": 0, "cache_mode": "memory", "max_calls": 5000},
        "anchors": {
            "precision": 0.9,
            "samples": 24,
            "coverage_samples": 80,
            "max_size": 4,
            "numeric_band": 0.2,
            "call_budget": 2000,
        },
        "permutation": {"repeats": 4, "call_budget": 200},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged = dict(base[key])  # type: ignore[arg-type]
            merged.update(value)
            base[key] = merged
        else:
            base[key] = value
    return load_config(overrides=base)


def run(fn: object, *args: object, **kwargs: object) -> object:
    async def inner() -> object:
        return await fn(*args, **kwargs)  # type: ignore[operator]

    return anyio.run(inner)


def _client(model: object, config: JevXaiConfig) -> ModelClient:
    return ModelClient(
        model,
        config.model,
        fingerprint=model_fingerprint(model),
        seed=config.seed,
    )


def _safe() -> dict[str, object]:
    return {
        "customer_id": "acme",
        "verified_user": True,
        "sanctions_match": False,
        "transaction_amount": 0,
        "note": "",
        "channel": "web",
    }


def test_defaults_load_without_new_sections() -> None:
    config = load_config(overrides={"seed": 3})
    assert config.anchors.precision == 0.95
    assert config.permutation.repeats == 8


def test_anchor_needs_verified_and_sanctions() -> None:
    config = _cfg()
    result = anyio.run(
        AnchorExplainer(config).explain, _client(ThresholdModel(), config), _safe(), FEATURES
    )
    names = [predicate.feature for predicate in result.predicates]
    assert "verified_user" in names
    assert "sanctions_match" in names
    assert "customer_id" not in names
    assert "note" not in names
    assert result.sufficient
    assert result.precision >= 0.9
    assert 0.0 <= result.coverage <= 1.0
    assert result.label == "SAFE"
    again = anyio.run(
        AnchorExplainer(config).explain, _client(ThresholdModel(), config), _safe(), FEATURES
    )
    assert again.predicates == result.predicates
    assert again.precision == result.precision
    assert again.coverage == result.coverage


def test_anchor_requires_feature_spec_and_respects_budget() -> None:
    config = _cfg()
    model = ThresholdModel()
    with pytest.raises(CapabilityError):
        anyio.run(AnchorExplainer(config).explain, _client(model, config), _safe(), None)
    diagnosis = diagnose(model, context=FEATURES)
    assert next(item for item in diagnosis.explainers if item.name == "anchors").enabled
    bare = diagnose(model)
    assert not next(item for item in bare.explainers if item.name == "permutation").enabled
    tight = _cfg(anchors={"samples": 8, "call_budget": 1, "coverage_samples": 4})
    stopped = anyio.run(AnchorExplainer(tight).explain, _client(model, tight), _safe(), FEATURES)
    assert stopped.predicates == []
    assert stopped.sufficient is False


def test_permutation_ranks_verification_above_an_inert_note() -> None:
    config = _cfg()
    instance = _safe()
    instance["transaction_amount"] = 20000
    result = anyio.run(
        PermutationExplainer(config).explain,
        _client(ThresholdModel(), config),
        instance,
        FEATURES,
    )
    by_name = {row.feature: row for row in result.rows}
    assert "customer_id" not in by_name
    assert by_name["verified_user"].importance is not None
    assert by_name["note"].importance == 0.0
    assert by_name["verified_user"].importance > by_name["note"].importance
    assert by_name["verified_user"].label_flip_rate > by_name["note"].label_flip_rate
    dataset = anyio.run(
        PermutationExplainer(config).explain_dataset,
        _client(ThresholdModel(), config),
        [instance, instance],
        FEATURES,
    )
    assert dataset.scope == "dataset"
    verified = next(row for row in dataset.rows if row.feature == "verified_user")
    assert verified.importance is not None
    assert verified.importance > 0
    tight = _cfg(permutation={"repeats": 4, "call_budget": 1})
    stopped = anyio.run(
        PermutationExplainer(tight).explain,
        _client(ThresholdModel(), tight),
        instance,
        FEATURES,
    )
    assert any(row.importance is None and row.n_samples == 0 for row in stopped.rows)
    with pytest.raises(JevXaiUsageError):
        anyio.run(
            PermutationExplainer(config).explain_dataset,
            _client(ThresholdModel(), config),
            [],
            FEATURES,
        )


def test_permutation_reports_flips_without_probabilities() -> None:
    config = _cfg(permutation={"repeats": 2, "call_budget": 10})
    context = ExplainContext(features=[FeatureSpec(name="ok", kind="boolean")])
    result = anyio.run(
        PermutationExplainer(config).explain,
        _client(LabelOnlyModel(), config),
        {"ok": True},
        context,
    )
    assert result.rows[0].label_flip_rate == 1.0
    assert result.rows[0].importance is None


def test_stability_accepts_anchor_and_permutation_results() -> None:
    config = _cfg(
        anchors={"samples": 8, "coverage_samples": 8, "max_size": 2, "call_budget": 400},
        permutation={"repeats": 2, "call_budget": 40},
        stability={"runs": 2},
    )
    model = ThresholdModel()
    client = _client(model, config)
    anchored = run(
        StabilityEvaluator(config).evaluate,
        AnchorExplainer(config),
        model,
        _safe(),
        runs=2,
        context=FEATURES,
        client=client,
    )
    permuted = run(
        StabilityEvaluator(config).evaluate,
        PermutationExplainer(config),
        model,
        _safe(),
        runs=2,
        context=FEATURES,
        client=client,
    )
    assert 0.0 <= anchored.stability_score <= 1.0
    assert 0.0 <= permuted.stability_score <= 1.0
