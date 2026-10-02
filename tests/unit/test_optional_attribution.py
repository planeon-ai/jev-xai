"""Optional SHAP and LIME adapters, exercised with stand-in libraries."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import numpy as np
import pytest
from tests.fakes import ThresholdModel

from jev_xai.adapters.capabilities import diagnose
from jev_xai.config.loader import load_config
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import BudgetExceededError, CapabilityError, JevXaiError, JevXaiUsageError
from jev_xai.evidence.audit import build_audit_pack
from jev_xai.evidence.html import render_html
from jev_xai.evidence.report import render_markdown
from jev_xai.evidence.store import verify_pack
from jev_xai.explainers.context import ExplainContext, FeatureSpec
from jev_xai.explainers.lime import LimeExplainer, lime_weights
from jev_xai.explainers.shap import ShapExplainer, kernel_shap_values
from jev_xai.explainers.tabular import (
    DirectCalls,
    background_instances,
    decode_row,
    encode_row,
    extra_installed,
    load_extra,
    tabular_columns,
)
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint
from jev_xai.replay.recorder import DecisionRecorder
from jev_xai.stability.evaluator import StabilityEvaluator

FEATURES = ExplainContext(
    features=[
        FeatureSpec(name="verified_user", kind="boolean"),
        FeatureSpec(name="sanctions_match", kind="boolean"),
        FeatureSpec(
            name="transaction_amount",
            kind="numeric",
            allowed_range=(0.0, 50000.0),
        ),
        FeatureSpec(name="channel", kind="categorical", allowed_values=["web", "pos"]),
        FeatureSpec(name="region", kind="categorical"),
        FeatureSpec(name="note", kind="text"),
    ],
    background_rows=[
        {
            "verified_user": False,
            "sanctions_match": False,
            "transaction_amount": 0,
            "channel": "web",
            "note": "",
        },
        {
            "verified_user": True,
            "sanctions_match": True,
            "transaction_amount": 1000,
            "channel": "pos",
            "note": "",
        },
    ],
)


def _cfg(**overrides: object) -> JevXaiConfig:
    base: dict[str, object] = {
        "seed": 7,
        "model": {"retries": 0, "cache_mode": "off", "max_calls": 50},
        "shap": {"nsamples": 8, "call_budget": 40},
        "lime": {"num_features": 3, "num_samples": 8, "call_budget": 40},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged = dict(base[key])  # type: ignore[arg-type]
            merged.update(value)
            base[key] = merged
        else:
            base[key] = value
    return load_config(overrides=base)


def _client(config: JevXaiConfig) -> ModelClient:
    model = ThresholdModel()
    return ModelClient(
        model,
        config.model,
        fingerprint=model_fingerprint(model),
        seed=config.seed,
    )


def _instance() -> dict[str, object]:
    return {
        "verified_user": True,
        "sanctions_match": False,
        "transaction_amount": 0,
        "channel": "web",
        "note": "",
    }


class _Kernel:
    def __init__(self, predict_fn: object, background: np.ndarray) -> None:
        self.expected_value = np.array([0.62])
        predict_fn(background)  # type: ignore[operator]

    def shap_values(
        self, instance: np.ndarray, nsamples: int = 1, silent: bool = False
    ) -> np.ndarray:
        del nsamples
        if silent:
            raise TypeError("silent is not supported")
        return np.array([0.4, -0.3, 0.05, 0.0])


class _ShapModule:
    KernelExplainer = _Kernel


class _Explanation:
    def as_map(self) -> dict[int, list[tuple[int, float]]]:
        return {1: [(0, 0.5), (2, -0.1), (9, 1.0)]}


class _LimeTabular:
    def __init__(self, background: np.ndarray, feature_names: list[str], **kwargs: object) -> None:
        del background, kwargs
        self.feature_names = feature_names

    def explain_instance(
        self, instance: np.ndarray, predict_fn: object, num_features: int, num_samples: int
    ) -> _Explanation:
        del instance, num_features, num_samples
        predict_fn(np.zeros((2, len(self.feature_names))))  # type: ignore[operator]
        return _Explanation()


class _LimeModule:
    LimeTabularExplainer = _LimeTabular


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jev_xai.adapters.capabilities.extra_installed", lambda module: True)


def test_direct_calls_round_trip_into_the_model() -> None:
    specs, skipped = tabular_columns(FEATURES)
    assert skipped == ["region", "note"]
    encoded = encode_row({"transaction_amount": None, "channel": "missing"}, specs)
    assert encoded[2] == 0.0
    assert encoded[3] == -1.0
    assert background_instances(ExplainContext()) == ([], "")
    unchanged = decode_row(
        {"region": "eu"},
        [FeatureSpec(name="region", kind="categorical")],
        np.array([0.0]),
    )
    assert unchanged == {"region": "eu"}
    counter = DirectCalls(10)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 1.0, 20000.0, 1.2],
        ]
    )
    values = counter.probabilities(ThresholdModel(), _instance(), specs, matrix, "SAFE")
    assert counter.calls == 2
    assert values[0] > values[1]


def test_missing_extra_is_reported_before_a_model_call() -> None:
    if extra_installed("shap"):
        assert load_extra("shap", "shap") is not None
    else:
        with pytest.raises(JevXaiUsageError):
            load_extra("shap", "shap")
    diagnosis = diagnose(ThresholdModel(), context=FEATURES)
    shap = next(item for item in diagnosis.explainers if item.name == "shap")
    if not extra_installed("shap"):
        assert not shap.enabled
        assert shap.missing[0] == "shap_extra"
    config = _cfg()
    if not extra_installed("shap"):
        with pytest.raises(CapabilityError) as caught:
            anyio.run(ShapExplainer(config).explain, _client(config), _instance(), FEATURES)
        assert caught.value.prerequisite == "shap_extra"


def test_kernel_shap_maps_values_onto_feature_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    monkeypatch.setattr(
        "jev_xai.explainers.shap.kernel_shap_values",
        lambda predict_fn, background, instance, *, nsamples, seed, module=None: (
            [0.4, -0.3, 0.05, 0.0],
            0.62,
        ),
    )
    config = _cfg()
    result = anyio.run(ShapExplainer(config).explain, _client(config), _instance(), FEATURES)
    assert [row.feature for row in result.rows] == [
        "verified_user",
        "sanctions_match",
        "transaction_amount",
        "channel",
    ]
    assert result.rows[0].value == pytest.approx(0.4)
    assert result.base_value == pytest.approx(0.62)
    assert result.skipped == ["region", "note"]
    assert result.label == "SAFE"
    assert result.cost.n_model_calls >= 1
    direct = kernel_shap_values(
        lambda matrix: np.zeros(len(np.atleast_2d(matrix))),
        np.zeros((2, 4)),
        np.zeros(4),
        nsamples=4,
        seed=1,
        module=_ShapModule,
    )
    assert direct[0] == pytest.approx([0.4, -0.3, 0.05, 0.0])
    assert direct[1] == pytest.approx(0.62)


def test_lime_weights_skip_unknown_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    monkeypatch.setattr(
        "jev_xai.explainers.lime.lime_weights",
        lambda *args, **kwargs: [(0, 0.5), (2, -0.1), (9, 1.0)],
    )
    config = _cfg()
    context = FEATURES.model_copy(
        update={"background_rows": [], "background": FEATURES.background_rows[0]}
    )
    result = anyio.run(LimeExplainer(config).explain, _client(config), _instance(), context)
    assert [(row.feature, row.weight) for row in result.rows] == [
        ("verified_user", 0.5),
        ("transaction_amount", -0.1),
    ]
    assert "one-row background" in result.note
    weights = lime_weights(
        lambda matrix: np.zeros(len(np.atleast_2d(matrix))),
        np.zeros((2, 4)),
        np.zeros(4),
        ["a", "b", "c", "d"],
        num_features=2,
        num_samples=4,
        seed=1,
        module=_LimeModule,
    )
    assert weights[0] == (0, 0.5)
    assert (
        lime_weights(
            lambda matrix: np.zeros(1),
            np.zeros((1, 1)),
            np.zeros(1),
            ["a"],
            num_features=1,
            num_samples=1,
            seed=1,
            module=_EmptyLime(),
        )
        == []
    )


class _EmptyLime:
    class LimeTabularExplainer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def explain_instance(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return self

        def as_map(self) -> dict[int, list[tuple[int, float]]]:
            return {}


def test_attribution_requires_background_and_a_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)

    def explode(
        predict_fn: object, background: np.ndarray, *args: object, **kwargs: object
    ) -> object:
        del args, kwargs
        predict_fn(background)  # type: ignore[operator]
        return [0.0, 0.0, 0.0, 0.0], 0.0

    monkeypatch.setattr("jev_xai.explainers.shap.kernel_shap_values", explode)
    config = _cfg(shap={"nsamples": 4, "call_budget": 0})
    with pytest.raises(BudgetExceededError):
        anyio.run(ShapExplainer(config).explain, _client(config), _instance(), FEATURES)
    bare = FEATURES.model_copy(update={"background_rows": [], "background": None})
    with pytest.raises(CapabilityError) as caught:
        anyio.run(ShapExplainer(_cfg()).explain, _client(_cfg()), _instance(), bare)
    assert caught.value.prerequisite == "background_data"
    text_only = ExplainContext(
        features=[FeatureSpec(name="note", kind="text")],
        background={"note": ""},
    )
    with pytest.raises(JevXaiError):
        anyio.run(ShapExplainer(_cfg()).explain, _client(_cfg()), {"note": ""}, text_only)


def test_stability_scores_shap_and_lime(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    monkeypatch.setattr(
        "jev_xai.explainers.shap.kernel_shap_values",
        lambda *args, **kwargs: ([0.4, -0.3, 0.05, 0.0], 0.62),
    )
    monkeypatch.setattr(
        "jev_xai.explainers.lime.lime_weights",
        lambda *args, **kwargs: [(0, 0.5), (1, -0.2)],
    )
    config = _cfg(stability={"runs": 2})
    model = ThresholdModel()
    client = _client(config)
    shap = anyio.run(
        _evaluate,
        ShapExplainer(config),
        model,
        client,
        config,
    )
    lime = anyio.run(
        _evaluate,
        LimeExplainer(config),
        model,
        client,
        config,
    )
    assert shap.stability_score > 0.9
    assert lime.stability_score > 0.9
    assert shap.measured_explainer == "shap"
    assert lime.measured_explainer == "lime"


def test_audit_pack_includes_shap_and_lime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    monkeypatch.setattr(
        "jev_xai.explainers.shap.kernel_shap_values",
        lambda *args, **kwargs: ([0.4, -0.3, 0.05, 0.0], 0.62),
    )
    monkeypatch.setattr(
        "jev_xai.explainers.lime.lime_weights",
        lambda *args, **kwargs: [(0, 0.5), (1, -0.2)],
    )
    config = _audit_config()

    async def build() -> Path:
        record = await DecisionRecorder(ThresholdModel(), config).run(
            _instance(),
            decision_id="audit-shap",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        return await build_audit_pack(
            record,
            ThresholdModel(),
            config,
            tmp_path / "audit",
            context=FEATURES,
        )

    pack = anyio.run(build)
    shap = json.loads((pack / "shap.json").read_text(encoding="utf-8"))
    lime = json.loads((pack / "lime.json").read_text(encoding="utf-8"))
    explanation = json.loads((pack / "explanation.json").read_text(encoding="utf-8"))
    assert shap.get("skipped") is not True
    assert lime.get("skipped") is not True
    assert shap["rows"][0]["feature"] == "verified_user"
    assert shap["rows"][0]["value"] == pytest.approx(0.4)
    assert shap["base_value"] == pytest.approx(0.62)
    assert shap["skipped"] == ["region", "note"]
    assert lime["rows"][0]["feature"] == "verified_user"
    assert explanation["shap_top"][0] == "verified_user"
    assert explanation["lime_top"][0] == "verified_user"
    report = (pack / "report.md").read_text(encoding="utf-8")
    html = (pack / "report.html").read_text(encoding="utf-8")
    assert "## SHAP" in report and "## LIME" in report
    assert "Columns left out: region, note" in report
    assert "Columns left out: region, note" in html
    assert "Base value 0.62." in report
    assert verify_pack(pack)


def test_audit_skips_attribution_without_a_tabular_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("attribution should not run")

    monkeypatch.setattr("jev_xai.explainers.shap.kernel_shap_values", explode)
    monkeypatch.setattr("jev_xai.explainers.lime.lime_weights", explode)
    config = _audit_config()
    context = ExplainContext(
        features=[FeatureSpec(name="note", kind="text")],
        background={"note": ""},
    )

    async def build() -> Path:
        record = await DecisionRecorder(ThresholdModel(), config).run(
            {"note": "hello"},
            decision_id="audit-text",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        return await build_audit_pack(
            record,
            ThresholdModel(),
            config,
            tmp_path / "audit",
            context=context,
        )

    pack = anyio.run(build)
    shap = json.loads((pack / "shap.json").read_text(encoding="utf-8"))
    lime = json.loads((pack / "lime.json").read_text(encoding="utf-8"))
    assert shap["skipped"] is True
    assert shap["prerequisite"] == "tabular_features"
    assert lime["prerequisite"] == "tabular_features"
    report = (pack / "report.md").read_text(encoding="utf-8")
    assert "Skipped (tabular_features)" in report
    assert verify_pack(pack)


def test_report_names_a_one_row_background() -> None:
    pack = {
        "decision": {"output": {}},
        "shap": {
            "rows": [{"feature": "verified_user", "value": 0.4}],
            "skipped": [],
            "note": "one-row background; attributions are relative to that single baseline",
            "base_value": 0.5,
        },
        "lime": {"rows": [], "skipped": [], "note": ""},
    }
    page = render_markdown(pack)
    html = render_html(pack)
    assert "one-row background; attributions are relative to that single baseline" in page
    assert "one-row background; attributions are relative to that single baseline" in html
    assert "| — | — |" in page


def _audit_config() -> JevXaiConfig:
    return load_config(
        overrides={
            "seed": 7,
            "reproducibility": {"repeat_probe_runs": 0},
            "model": {"retries": 0, "cache_mode": "off"},
            "counterfactual": {"call_budget": 40, "max_candidates": 2},
            "ablation": {"max_spans": 2},
            "anchors": {"samples": 4, "max_size": 2, "coverage_samples": 8, "call_budget": 40},
            "permutation": {"repeats": 2, "call_budget": 20},
            "stability": {"runs": 2},
        }
    )


async def _evaluate(
    explainer: object, model: object, client: ModelClient, config: JevXaiConfig
) -> object:
    return await StabilityEvaluator(config).evaluate(
        explainer,  # type: ignore[arg-type]
        model,
        _instance(),
        runs=2,
        context=FEATURES,
        client=client,
    )
