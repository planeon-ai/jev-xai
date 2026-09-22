"""Build an audit pack from a recorded decision and a live model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jev_xai.config.loader import config_hash
from jev_xai.config.schema import JevXaiConfig
from jev_xai.evidence.html import render_html
from jev_xai.evidence.report import render_markdown
from jev_xai.evidence.schema import DecisionRecord
from jev_xai.evidence.store import EvidenceStore
from jev_xai.explainers.ablation import AblationExplainer
from jev_xai.explainers.context import ExplainContext
from jev_xai.explainers.counterfactual import CounterfactualExplainer
from jev_xai.model.cassette import Cassette
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint
from jev_xai.replay.replay import ReplayEngine
from jev_xai.stability.evaluator import StabilityEvaluator


async def build_audit_pack(
    record: DecisionRecord,
    model: Any,
    config: JevXaiConfig,
    directory: Path,
    *,
    context: ExplainContext | None = None,
) -> Path:
    """Write the V1 audit pack, including a verifiable manifest."""

    if not isinstance(record.input, dict):
        raise ValueError("audit requires a reachable dict input on the decision record")
    cassette = Cassette(directory / "cassette")
    client = ModelClient(
        model,
        config.model,
        fingerprint=model_fingerprint(model),
        seed=config.seed,
        cassette=cassette,
    )
    if record.reproducibility is not None:
        client.noise_floor = record.reproducibility.noise_floor
    ablation = await AblationExplainer(config).explain(client, record.input, context)
    counterfactual = await CounterfactualExplainer(config).explain(client, record.input, context)
    stability = await StabilityEvaluator(config).evaluate(
        AblationExplainer(config),
        model,
        record.input,
        runs=min(config.stability.runs, 3),
        context=context,
        client=client,
    )
    replay = await ReplayEngine(config, cassette=cassette).replay(record, model=model, mode="exact")
    behavioral = await ReplayEngine(config, cassette=cassette).replay(
        record, model=model, mode="current"
    )
    pack = {
        "decision": record.model_dump(mode="json"),
        "ablation": ablation.model_dump(mode="json"),
        "counterfactuals": counterfactual.model_dump(mode="json"),
        "stability": stability.model_dump(mode="json"),
        "replay": {
            "evidence": replay.model_dump(mode="json"),
            "behavioral": behavioral.model_dump(mode="json"),
            "claim": replay.claim,
            "matched": replay.matched,
            "mismatch_reason": behavioral.mismatch_reason,
            "note": replay.note,
        },
        "explanation": {
            "ablation_top": [row.feature for row in ablation.rows[:5]],
            "config_hash": config_hash(config),
        },
    }
    store = EvidenceStore(directory / "store")
    members: dict[str, Any] = {
        "decision.json": pack["decision"],
        "explanation.json": pack["explanation"],
        "counterfactuals.json": pack["counterfactuals"],
        "ablation.json": pack["ablation"],
        "stability.json": pack["stability"],
        "replay.json": pack["replay"],
    }
    report_md = render_markdown(pack)
    report_html = render_html(pack)
    members["report.md"] = report_md
    members["report.html"] = report_html
    written = store.write_pack(record.decision_id, members)
    # Also copy the pack to ``directory`` so ``jev-xai audit`` has a flat layout.
    flat = directory
    flat.mkdir(parents=True, exist_ok=True)
    for name in (
        "decision.json",
        "explanation.json",
        "counterfactuals.json",
        "ablation.json",
        "stability.json",
        "replay.json",
        "manifest.json",
        "report.md",
        "report.html",
    ):
        source = written / name
        target = flat / name
        if source.resolve() != target.resolve():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return flat
