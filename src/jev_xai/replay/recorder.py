"""Record a decision at call time. This is the inline capture path."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from jev_xai.async_compat import run_sync
from jev_xai.canonical import content_hash
from jev_xai.config.loader import config_hash
from jev_xai.config.schema import JevXaiConfig
from jev_xai.evidence.redaction import prepare_input
from jev_xai.evidence.schema import (
    SCHEMA_VERSION,
    DecisionRecord,
    ModelInfo,
    OutputInfo,
    TraceInfo,
)
from jev_xai.model.cassette import Cassette
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint
from jev_xai.replay.environment import runtime_fingerprint
from jev_xai.replay.repeatability import ReproducibilityProbe


class DecisionRecorder:
    """Wrap a model call and return a schema-valid record plus a cassette entry."""

    def __init__(
        self,
        model: Any,
        config: JevXaiConfig | None = None,
        *,
        cassette: Cassette | None = None,
    ) -> None:
        self.model = model
        self.config = config or JevXaiConfig()
        self.fingerprint = model_fingerprint(model)
        self.cassette = cassette
        self.client = ModelClient(
            model,
            self.config.model,
            fingerprint=self.fingerprint,
            seed=self.config.seed,
            cassette=cassette,
        )

    async def run(
        self,
        instance: Mapping[str, Any],
        *,
        trace: TraceInfo | None = None,
        decision_id: str | None = None,
        timestamp: str | None = None,
    ) -> DecisionRecord:
        payload = dict(instance)
        prediction = await self.client.predict(payload)
        key = self.client.cache_key(payload)
        self.client.force_store(key, prediction)
        probe = None
        if self.config.reproducibility.repeat_probe_runs > 0:
            probe = await ReproducibilityProbe(self.config).measure(
                self.client, payload, reference_label=prediction.label
            )
        meta = self.model.metadata() if callable(getattr(self.model, "metadata", None)) else {}
        stored_input, reachable, externalized = prepare_input(payload, self.config.evidence)
        return DecisionRecord(
            schema_version=SCHEMA_VERSION,
            decision_id=decision_id or str(uuid.uuid4()),
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            model=ModelInfo(
                provider=str(meta.get("provider", "unknown")),
                model_name=str(meta.get("model_name", "unknown")),
                model_version=str(meta.get("model_version", "unknown")),
                artifact_hash=meta.get("artifact_hash"),
                fingerprint=self.fingerprint,
            ),
            input=stored_input,
            input_hash=content_hash(payload),
            input_reachable=reachable,
            input_externalized=externalized,
            output=OutputInfo(
                label=prediction.label,
                probability=prediction.probability,
                probabilities=prediction.probabilities,
            ),
            threshold=self.config.threshold,
            policy_version=self.config.policy_version,
            runtime=runtime_fingerprint(self.config.seed),
            explainer_config={
                "config_hash": config_hash(self.config),
                "config": self.config.model_dump(mode="json"),
            },
            cost=self.client.cost(),
            trace=trace or TraceInfo(),
            cassette_key=key,
            reproducibility=probe,
        )

    def run_sync(self, instance: Mapping[str, Any], **kwargs: Any) -> DecisionRecord:
        return run_sync(self.run, instance, **kwargs)

    def wrap(self) -> Any:
        """Async middleware. ``await middleware(instance, trace_id=...)`` records the call."""

        recorder = self

        async def middleware(instance: Mapping[str, Any], **trace: Any) -> DecisionRecord:
            return await recorder.run(instance, trace=TraceInfo(**trace))

        return middleware
