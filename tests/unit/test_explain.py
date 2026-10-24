"""Ablation, counterfactuals, stability, replay, and evidence packs."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from tests.fakes import FlippedModel, NoPredictModel, ScriptedNoise, ThresholdModel

from jev_xai.adapters.capabilities import Diagnosis, diagnose
from jev_xai.config.loader import config_hash, load_config
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import (
    BudgetExceededError,
    CapabilityError,
    JevXaiError,
    JevXaiUsageError,
    ReplayMismatchError,
    SchemaVersionError,
)
from jev_xai.evidence.audit import _optional, build_audit_pack
from jev_xai.evidence.redaction import prepare_input
from jev_xai.evidence.schema import (
    DecisionRecord,
    ModelInfo,
    OutputInfo,
    TraceInfo,
    read_record,
)
from jev_xai.evidence.store import EvidenceStore, verify_pack
from jev_xai.explainers.ablation import AblationExplainer
from jev_xai.explainers.anchors import AnchorExplainer
from jev_xai.explainers.context import ExplainContext, FeatureSpec
from jev_xai.explainers.counterfactual import CounterfactualExplainer
from jev_xai.model.cassette import Cassette
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint
from jev_xai.replay.diff import cross_version_diff
from jev_xai.replay.environment import runtime_fingerprint
from jev_xai.replay.recorder import DecisionRecorder
from jev_xai.replay.repeatability import ReproducibilityProbe
from jev_xai.replay.replay import ReplayEngine, raise_on_mismatch
from jev_xai.stability.evaluator import StabilityEvaluator, assert_stable, explain_with_stability
from jev_xai.stability.metrics import kendall_tau_b, spearman

FEATURES = ExplainContext(
    features=[
        FeatureSpec(name="verified_user", kind="boolean", baseline=False),
        FeatureSpec(name="sanctions_match", kind="boolean", baseline=False),
        FeatureSpec(
            name="transaction_amount",
            kind="numeric",
            allowed_range=(0.0, 50000.0),
            baseline=0.0,
        ),
        FeatureSpec(name="note", kind="text"),
    ],
    groups={"identity": ["verified_user", "sanctions_match"]},
    target_label="SAFE",
)


def run(fn: object, *args: object, **kwargs: object) -> object:
    async def inner() -> object:
        return await fn(*args, **kwargs)  # type: ignore[operator]

    return anyio.run(inner)


def _cfg(**overrides: object) -> JevXaiConfig:
    base: dict[str, object] = {
        "reproducibility": {"repeat_probe_runs": 0},
        "model": {"retries": 0, "cache_mode": "memory"},
        "counterfactual": {
            "call_budget": 40,
            "max_candidates": 2,
            "max_features_changed": 3,
            "allowed_ranges": {"transaction_amount": [0, 50000]},
        },
        "ablation": {"max_spans": 2},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged = dict(base[key])  # type: ignore[arg-type]
            merged.update(value)
            base[key] = merged
        else:
            base[key] = value
    return load_config(overrides=base)


def _client(model: object, config: JevXaiConfig, cassette: Cassette | None = None) -> ModelClient:
    return ModelClient(
        model,
        config.model,
        fingerprint=model_fingerprint(model),
        seed=config.seed,
        cassette=cassette,
    )


def test_rank_correlations_match_reference_values() -> None:
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert kendall_tau_b([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_ablation_delta_and_call_count() -> None:
    config = _cfg()
    model = ThresholdModel()
    client = _client(model, config)
    result = anyio.run(AblationExplainer(config).explain, client, _safe_input(), FEATURES)
    by_name = {row.feature: row for row in result.rows}
    assert by_name["verified_user"].delta_p is not None
    assert by_name["verified_user"].delta_p > 0
    assert by_name["verified_user"].noop is False
    sanctions = by_name["sanctions_match"]
    assert sanctions.noop is True
    assert sanctions.delta_p is None
    assert sanctions.ablated_probability is None
    assert any(row.scope == "group" for row in result.rows)
    assert any(row.scope == "text_span" for row in result.rows)
    assert result.masking_policy["structured"] == "neutral"
    calls = result.cost.n_model_calls
    again = anyio.run(AblationExplainer(config).explain, client, _safe_input(), FEATURES)
    assert again.cost.n_model_calls == 0
    assert calls > 0


def test_ablation_requires_a_live_model() -> None:
    config = _cfg()
    client = _client(NoPredictModel(), config)
    with pytest.raises(CapabilityError) as caught:
        anyio.run(AblationExplainer(config).explain, client, {"ok": True}, None)
    assert caught.value.prerequisite == "model_reinvocation"


def test_noise_floor_flags_tiny_deltas() -> None:
    config = _cfg(
        model={"cache_mode": "off", "retries": 0},
        reproducibility={"repeat_probe_runs": 5, "noise_floor_sigma_k": 2},
    )
    model = ScriptedNoise()
    client = _client(model, config)
    instance = {"tiny": 0, "risk": 0}

    async def scenario() -> tuple[float, float, bool]:
        probe = await ReproducibilityProbe(config).measure(client, instance, reference_label="SAFE")
        context = ExplainContext(
            features=[
                FeatureSpec(name="tiny", kind="numeric", baseline=1),
                FeatureSpec(name="risk", kind="numeric", baseline=0),
            ]
        )
        ablation = await AblationExplainer(config).explain(client, instance, context)
        flagged = [row for row in ablation.rows if row.feature == "tiny" and row.below_noise_floor]
        return probe.reproduction_rate, probe.noise_floor, bool(flagged)

    rate, floor, flagged = anyio.run(scenario)
    assert rate < 1.0
    assert floor > 0
    assert flagged


def test_counterfactual_respects_immutable_features_and_can_flip() -> None:
    blocked = _cfg(counterfactual={"immutable_features": ["sanctions_match"]})
    unsafe = {
        "verified_user": False,
        "sanctions_match": True,
        "transaction_amount": 40000,
        "note": "",
    }
    client = _client(ThresholdModel(), blocked)
    held = anyio.run(CounterfactualExplainer(blocked).explain, client, unsafe, FEATURES)
    for candidate in held.candidates:
        assert all(change.feature != "sanctions_match" for change in candidate.changes)

    free = _cfg()
    client = _client(ThresholdModel(), free)
    found = anyio.run(CounterfactualExplainer(free).explain, client, unsafe, FEATURES)
    confirmed = [candidate for candidate in found.candidates if candidate.replay_confirmed]
    assert confirmed
    assert all(candidate.confirmed_label == candidate.label for candidate in confirmed)
    assert found.success_rate > 0
    assert found.cost.n_model_calls <= free.counterfactual.call_budget


def test_second_call_can_refuse_a_flip() -> None:
    class Reverts:
        def __init__(self) -> None:
            self.seen: dict[str, int] = {}

        def predict(self, instance: dict[str, object]) -> dict[str, object]:
            key = json.dumps(instance, sort_keys=True, default=str)
            count = self.seen.get(key, 0) + 1
            self.seen[key] = count
            verified = bool(instance.get("verified_user"))
            sanctions = bool(instance.get("sanctions_match"))
            probability = 0.2 if count >= 2 or not verified or sanctions else 0.9
            label = "SAFE" if probability >= 0.5 else "UNSAFE"
            return {
                "label": label,
                "probability": probability,
                "probabilities": {"SAFE": probability, "UNSAFE": 1 - probability},
            }

        def metadata(self) -> dict[str, str]:
            return {
                "provider": "toy",
                "model_name": "reverts",
                "model_version": "1",
                "fingerprint": "reverts-v1",
            }

    config = _cfg()
    found = anyio.run(
        CounterfactualExplainer(config).explain,
        _client(Reverts(), config),
        {
            "verified_user": False,
            "sanctions_match": True,
            "transaction_amount": 40000,
            "note": "",
        },
        FEATURES,
    )
    refused = [
        candidate
        for candidate in found.candidates
        if candidate.flipped and not candidate.replay_confirmed
    ]
    assert refused
    assert all(candidate.confirmed_label not in (None, candidate.label) for candidate in refused)
    assert found.success_rate == 0


def test_confirmation_is_not_claimed_when_the_model_budget_is_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ModelClient.predict

    async def wrapped(
        self: ModelClient, instance: dict[str, object], *, use_cache: bool = True
    ) -> object:
        if not use_cache:
            raise BudgetExceededError("model call budget exhausted")
        return await original(self, instance, use_cache=use_cache)

    monkeypatch.setattr(ModelClient, "predict", wrapped)
    config = _cfg()
    found = anyio.run(
        CounterfactualExplainer(config).explain,
        _client(ThresholdModel(), config),
        {
            "verified_user": False,
            "sanctions_match": True,
            "transaction_amount": 40000,
            "note": "",
        },
        FEATURES,
    )
    missed = [
        candidate
        for candidate in found.candidates
        if candidate.flipped and not candidate.replay_confirmed
    ]
    assert missed
    assert all(candidate.confirmed_label is None for candidate in missed)
    assert found.success_rate == 0


def test_stability_parallel_matches_sequential_and_formula() -> None:
    config = _cfg(stability={"runs": 3, "failure_tolerance": 0})
    model = ThresholdModel()
    parallel_client = _client(model, config)
    sequential_client = _client(model, config)
    evaluator = StabilityEvaluator(config)
    explainer = AblationExplainer(config)
    instance = _safe_input()

    async def both() -> tuple[object, object]:
        parallel = await evaluator.evaluate(
            explainer, model, instance, context=FEATURES, client=parallel_client
        )
        sequential = await evaluator.evaluate_sequential(
            explainer, model, instance, context=FEATURES, client=sequential_client
        )
        return parallel, sequential

    parallel, sequential = anyio.run(both)
    assert parallel.stability_score == pytest.approx(sequential.stability_score)
    assert [item["rows"] for item in parallel.explanations] == [
        item["rows"] for item in sequential.explanations
    ]
    assert parallel.formula == "stability_score_v1"
    assert parallel.explainer == "stability"
    assert parallel.measured_explainer == "ablation"
    assert sequential.measured_explainer == "ablation"
    assert_stable(parallel, min_score=0.8)
    helper = run(explain_with_stability, explainer, model, instance, config, context=FEATURES)
    assert helper.stability_score == pytest.approx(parallel.stability_score)
    assert helper.measured_explainer == "ablation"


def test_stability_measures_the_model_not_the_warm_cache() -> None:
    class Counting(ThresholdModel):
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, instance: dict[str, object]) -> dict[str, object]:
            self.calls += 1
            return super().predict(instance)

    config = _cfg(stability={"runs": 2, "failure_tolerance": 0})
    model = Counting()
    client = _client(model, config)
    instance = _safe_input()
    run(AblationExplainer(config).explain, client, instance, FEATURES)
    warmed = model.calls
    assert warmed > 0
    run(
        StabilityEvaluator(config).evaluate,
        AblationExplainer(config),
        model,
        instance,
        runs=1,
        context=FEATURES,
        client=client,
    )
    first_run = model.calls - warmed
    assert first_run > 0
    run(
        StabilityEvaluator(config).evaluate,
        AblationExplainer(config),
        model,
        instance,
        runs=1,
        context=FEATURES,
        client=client,
    )
    assert model.calls - warmed - first_run == first_run
    cached = model.calls
    run(client.predict, instance)
    assert model.calls == cached


def test_stability_partial_failure() -> None:
    config = _cfg(stability={"runs": 4, "failure_tolerance": 0.0})

    class Flaky:
        name = "ablation"

        def __init__(self, config: JevXaiConfig, seed: int = 0) -> None:
            self.config = config
            self.seed = seed
            self.inner = AblationExplainer(config)

        async def explain(
            self, client: ModelClient, instance: dict[str, object], context: object = None
        ) -> object:
            if self.seed % 2 == 0:
                raise RuntimeError("nope")
            return await self.inner.explain(client, instance, context)  # type: ignore[arg-type]

    with pytest.raises(JevXaiError):
        run(
            StabilityEvaluator(config).evaluate,
            Flaky(config),
            ThresholdModel(),
            _safe_input(),
            context=FEATURES,
        )


def test_record_evidence_replay_and_behavioral_reproduction(tmp_path: Path) -> None:
    config = _cfg()
    cassette = Cassette(tmp_path / "cassette")
    model = ThresholdModel()
    recorder = DecisionRecorder(model, config, cassette=cassette)
    record = run(
        recorder.run, _safe_input(), decision_id="dec-1", timestamp="2026-01-01T00:00:00+00:00"
    )
    assert record.input_hash
    assert record.explainer_config["config_hash"] == config_hash(config)
    assert record.schema_version == "1.0.0"
    engine = ReplayEngine(config, cassette=cassette)
    evidence = run(engine.replay, record, mode="exact")
    assert evidence.claim == "evidence_replay"
    assert evidence.cost.n_model_calls == 0
    assert evidence.replay_label == record.output.label
    live = run(engine.replay, record, model=model, mode="current")
    assert live.claim == "behavioral_reproduction"
    assert live.matched
    raise_on_mismatch(live)


def test_cross_replay_and_externalized_input(tmp_path: Path) -> None:
    config = _cfg(evidence={"max_inline_input_bytes": 16})
    model = ThresholdModel()
    bare = run(DecisionRecorder(model, config).run, _safe_input(), decision_id="inline-hash")
    assert bare.input_externalized is True
    assert bare.input_reachable is False
    store = EvidenceStore(tmp_path / "store")
    record = run(
        DecisionRecorder(model, config, store=store).run,
        _safe_input(),
        decision_id="stored",
    )
    assert record.input_externalized is True
    assert record.input_reachable is True
    assert isinstance(record.input, dict)
    digest = str(record.input["external_hash"])
    assert store.get_object(digest)["verified_user"] is True
    live = run(ReplayEngine(config, store=store).replay, record, model=model, mode="cross")
    assert live.mode == "cross"
    assert live.claim == "behavioral_reproduction"
    assert live.matched
    with pytest.raises(ReplayMismatchError):
        run(ReplayEngine(config).replay, record, model=model, mode="current")
    with pytest.raises(JevXaiUsageError):
        run(ReplayEngine(config).replay, record, model=model, mode="counterfactual")


def test_mismatch_classifies_model_version_change(tmp_path: Path) -> None:
    config = _cfg()
    record = anyio.run(DecisionRecorder(ThresholdModel(), config).run, _safe_input())
    report = anyio.run(cross_version_diff, [record], FlippedModel(), config)
    assert report.n_flipped == 1
    assert report.rows[0].mismatch_reason == "model_version_changed"


def test_nonstationary_input_and_config_warning() -> None:
    config = _cfg()
    runtime = runtime_fingerprint(config.seed)
    model = ThresholdModel()
    record = DecisionRecord(
        decision_id="ns",
        timestamp="2026-01-01T00:00:00+00:00",
        model=ModelInfo(
            provider="toy",
            model_name="threshold",
            model_version="1",
            fingerprint=model_fingerprint(model),
        ),
        input={"timestamp": "now", "verified_user": True},
        input_hash="abc",
        output=OutputInfo(label="UNSAFE", probability=0.1),
        runtime=runtime,
        explainer_config={"config_hash": config_hash(config)},
    )
    result = run(ReplayEngine(config).replay, record, model=model, mode="current")
    assert result.mismatch_reason == "input_nonstationary"
    other = _cfg(seed=99)
    warned = anyio.run(cross_version_diff, [record], model, other)
    assert warned.config_hash_warning
    with pytest.raises(ReplayMismatchError):
        run(ReplayEngine(config).replay, record, mode="current")


def test_audit_pack_merkle_and_hash_only(tmp_path: Path) -> None:
    config = _cfg(stability={"runs": 2})
    record = run(
        DecisionRecorder(ThresholdModel(), config).run,
        _safe_input(),
        decision_id="audit-1",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    pack = anyio.run(build_audit_pack, record, ThresholdModel(), config, tmp_path / "audit")
    assert (pack / "report.md").is_file()
    assert (pack / "report.html").is_file()
    assert "limitations" in (pack / "manifest.json").read_text(
        encoding="utf-8"
    ).lower() or "does not reveal" in (pack / "manifest.json").read_text(encoding="utf-8")
    assert verify_pack(pack)
    skipped = json.loads((pack / "anchors.json").read_text(encoding="utf-8"))
    assert skipped["skipped"] is True
    assert skipped["prerequisite"] == "feature_spec"
    report = (pack / "report.md").read_text(encoding="utf-8")
    assert "Skipped (feature_spec)" in report
    stability = json.loads((pack / "stability.json").read_text(encoding="utf-8"))
    assert stability["measured_explainer"] == "ablation"
    assert "stability_score_v1 = " in report and "on ablation" in report
    shap = json.loads((pack / "shap.json").read_text(encoding="utf-8"))
    lime = json.loads((pack / "lime.json").read_text(encoding="utf-8"))
    bare = diagnose(ThresholdModel(), input_reachable=True)
    assert shap["skipped"] is True
    assert lime["skipped"] is True
    assert shap["prerequisite"] == _missing(bare, "shap")
    assert lime["prerequisite"] == _missing(bare, "lime")
    assert f"Skipped ({shap['prerequisite']})" in report
    assert "## LIME" in report
    html = (pack / "report.html").read_text(encoding="utf-8")
    assert "on ablation" in html
    assert "<script" not in html
    decision = json.loads((pack / "decision.json").read_text(encoding="utf-8"))
    decision["output"]["label"] = "TAMPERED"
    (pack / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
    assert verify_pack(pack) is False

    stored, reachable, externalized = prepare_input(
        {"prompt": "secret"},
        load_config(overrides={"evidence": {"redaction_mode": "hash_only"}}).evidence,
    )
    assert stored is None and reachable is False and externalized is False
    redacted, _, _ = prepare_input(
        {"prompt": "secret", "ok": True},
        load_config(
            overrides={"evidence": {"redaction_mode": "redacted", "redact_fields": ["prompt"]}}
        ).evidence,
    )
    assert redacted["prompt"] == "[REDACTED]"
    store = EvidenceStore(tmp_path / "store")
    key = store.put_object({"a": 1})
    assert store.get_object(key) == {"a": 1}


def test_audit_pack_includes_rules_when_features_exist(tmp_path: Path) -> None:
    config = _cfg(
        stability={"runs": 2},
        anchors={"samples": 4, "max_size": 2, "coverage_samples": 8, "call_budget": 40},
        permutation={"repeats": 2, "call_budget": 20},
    )
    record = run(
        DecisionRecorder(ThresholdModel(), config).run,
        _safe_input(),
        decision_id="audit-rules",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    pack = run(
        build_audit_pack,
        record,
        ThresholdModel(),
        config,
        tmp_path / "audit",
        context=FEATURES,
    )
    anchors = json.loads((pack / "anchors.json").read_text(encoding="utf-8"))
    permutation = json.loads((pack / "permutation.json").read_text(encoding="utf-8"))
    explanation = json.loads((pack / "explanation.json").read_text(encoding="utf-8"))
    assert anchors.get("skipped") is not True
    assert "verified_user" in {row["feature"] for row in permutation["rows"]}
    assert explanation["anchor_sufficient"] in (True, False)
    assert "verified_user" in explanation["permutation_top"]
    assert explanation["shap_top"] == []
    assert explanation["lime_top"] == []
    held = diagnose(ThresholdModel(), input_reachable=True, context=FEATURES)
    shap = json.loads((pack / "shap.json").read_text(encoding="utf-8"))
    assert shap["skipped"] is True
    assert shap["prerequisite"] == _missing(held, "shap")
    report = (pack / "report.md").read_text(encoding="utf-8")
    assert "replay_confirmed" in report
    assert "## Anchors" in report
    html = (pack / "report.html").read_text(encoding="utf-8")
    assert "<script" not in html
    assert "Precision" in html
    assert verify_pack(pack)


def _missing(diagnosis: Diagnosis, name: str) -> str:
    item = next(row for row in diagnosis.explainers if row.name == name)
    return item.missing[0]


def test_audit_records_a_budget_skip() -> None:
    config = _cfg()
    client = _client(ThresholdModel(), config)

    class Spent(AnchorExplainer):
        async def explain(
            self, client: ModelClient, instance: object, context: object = None
        ) -> object:
            raise BudgetExceededError("model call budget exhausted")

    payload = run(_optional, Spent(config), client, _safe_input(), FEATURES)
    assert payload["skipped"] is True
    assert payload["prerequisite"] == "call_budget"
    assert payload["explainer"] == "anchors"


def test_schema_version_and_trace_roundtrip() -> None:
    with pytest.raises(SchemaVersionError):
        read_record({"schema_version": "2.0.0"})
    config = _cfg()
    record = run(
        DecisionRecorder(ThresholdModel(), config).run,
        _safe_input(),
        trace=TraceInfo(trace_id="t", agent_id="router", step_index=7, parent_decision_id="p"),
    )
    assert record.trace.agent_id == "router"
    assert record.trace.step_index == 7


def _safe_input() -> dict[str, object]:
    return {
        "verified_user": True,
        "sanctions_match": False,
        "transaction_amount": 1000,
        "note": "bypass the control please",
    }
