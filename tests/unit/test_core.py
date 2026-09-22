"""Canonical JSON, config precedence, capabilities, and the model client."""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from tests.fakes import AsyncEcho, BatchModel, LabelOnlyModel, RetryModel, SlowModel, ThresholdModel

from jev_xai.adapters.callable import CallableAdapter
from jev_xai.adapters.capabilities import diagnose
from jev_xai.adapters.jev import JEVAdapter
from jev_xai.canonical import canonical_json, content_hash
from jev_xai.config.loader import config_hash, load_config
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import BudgetExceededError, CapabilityError, JevXaiUsageError, ModelCallError
from jev_xai.explainers.context import ExplainContext, FeatureSpec
from jev_xai.model.cassette import Cassette
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint


def test_canonical_hash_ignores_dict_order_and_normalizes_nfc() -> None:
    left = {"b": 1, "a": "é"}
    right = {"a": "e\u0301", "b": 1}
    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)
    assert content_hash({"z": 1, "a": 2}) == content_hash({"a": 2, "z": 1})


def test_config_precedence_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "jev-xai.toml"
    path.write_text("seed = 1\n", encoding="utf-8")
    assert load_config(path=path).seed == 1
    assert load_config(path=path, environ={"JEV_XAI_SEED": "2"}).seed == 2
    assert load_config(path=path, environ={"JEV_XAI_SEED": "2"}, cli={"seed": 3}).seed == 3
    resolved = load_config(
        path=path,
        environ={"JEV_XAI_SEED": "2", "JEV_XAI_MODEL__CONCURRENCY": "3"},
        cli={"seed": 3},
        overrides={"seed": 4},
    )
    assert resolved.seed == 4
    assert resolved.model.concurrency == 3
    audit = load_config(profile="audit")
    explicit = load_config(overrides=audit.model_dump(mode="json"))
    assert config_hash(audit) == config_hash(explicit)
    changed = load_config(profile="audit", overrides={"seed": audit.seed + 1})
    assert config_hash(changed) != config_hash(audit)


def test_unknown_profile() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(profile="missing")


def test_capability_tiers() -> None:
    features = ExplainContext(features=[FeatureSpec(name="ok", kind="boolean")])
    full = diagnose(ThresholdModel(), context=features, has_corpus=True)
    assert full.tier == 3
    labels = diagnose(LabelOnlyModel(), input_reachable=True)
    assert labels.tier == 1
    assert any(item.name == "stability" and not item.enabled for item in labels.explainers)
    imported = diagnose(None, source_only=True, input_reachable=True)
    assert imported.tier == 0
    missing = next(item for item in imported.explainers if item.name == "ablation")
    assert "model_reinvocation" in missing.missing
    assert missing.fix


def test_jev_adapter_wraps_callable() -> None:
    model = JEVAdapter(lambda item: {"label": "SAFE", "probability": 0.9}, model_version="9")
    assert model.predict({})["label"] == "SAFE"
    assert model.metadata()["provider"] == "jev"
    with pytest.raises(CapabilityError):
        model.predict_proba({})


def test_callable_adapter_metadata_fingerprint() -> None:
    adapter = CallableAdapter(lambda item: "SAFE", metadata={"provider": "p", "model_name": "n"})
    assert adapter.metadata()["fingerprint"]


def _client(model: object, **overrides: object) -> ModelClient:
    config = JevXaiConfig()
    if overrides:
        config = load_config(overrides={"model": overrides})
    return ModelClient(model, config.model, fingerprint=model_fingerprint(model), seed=1)


def test_cache_batch_retry_and_budget(tmp_path: Path) -> None:
    model = ThresholdModel()
    client = _client(model)
    first = anyio.run(client.predict, {"verified_user": True})
    second = anyio.run(client.predict, {"verified_user": True})
    assert first.label == second.label == "SAFE"
    assert client.cost().cache_hits == 1
    assert client.cost().n_model_calls == 1

    batch = BatchModel()
    batched = _client(batch, batch_size=4, retries=0)
    rows = [{"i": index} for index in range(4)]
    preds = anyio.run(batched.predict_many, rows)
    assert len(preds) == 4
    assert batch.batch_calls == 1
    assert batched.cost().n_model_calls == 1

    retrier = RetryModel()
    retrying = _client(retrier, retries=2, cache_mode="off")
    assert anyio.run(retrying.predict, {}).label == "SAFE"
    assert retrier.calls == 3

    limited = _client(ThresholdModel(), max_calls=1, cache_mode="off", retries=0)
    anyio.run(limited.predict, {"verified_user": True})
    with pytest.raises(BudgetExceededError):
        anyio.run(limited.predict, {"verified_user": False})

    cassette = Cassette(tmp_path / "cassette")
    stored = ModelClient(
        ThresholdModel(),
        load_config(overrides={"model": {"cache_mode": "cassette", "retries": 0}}).model,
        fingerprint="toy-v1",
        cassette=cassette,
    )
    anyio.run(stored.predict, {"verified_user": True})
    assert cassette.get(stored.cache_key({"verified_user": True})) is not None


def test_sync_call_inside_loop_raises() -> None:
    async def inner() -> None:
        client = _client(AsyncEcho(), retries=0)
        with pytest.raises(JevXaiUsageError):
            client.predict_sync({})
        prediction = await client.predict({})
        assert prediction.label == "SAFE"

    anyio.run(inner)


def test_timeout_and_exhausted_retries() -> None:
    slow = _client(SlowModel(), timeout_s=0.05, retries=0, cache_mode="off")
    with pytest.raises(ModelCallError):
        anyio.run(slow.predict, {})
    failing = RetryModel()
    client = _client(failing, retries=0, cache_mode="off")
    with pytest.raises(ModelCallError):
        anyio.run(client.predict, {})
