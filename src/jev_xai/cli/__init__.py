"""jev-xai command line. Install the ``cli`` extra to use it."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import typer

from jev_xai.adapters.capabilities import diagnose
from jev_xai.async_compat import run_sync
from jev_xai.bench import write_benchmarks
from jev_xai.cli.render import render
from jev_xai.config.loader import config_hash, load_config
from jev_xai.config.schema import JevXaiConfig
from jev_xai.evidence.audit import build_audit_pack
from jev_xai.evidence.gate import gate_failures
from jev_xai.evidence.schema import read_record
from jev_xai.evidence.store import EvidenceStore, verify_pack
from jev_xai.explainers.ablation import AblationExplainer
from jev_xai.explainers.anchors import AnchorExplainer
from jev_xai.explainers.context import ExplainContext
from jev_xai.explainers.counterfactual import CounterfactualExplainer
from jev_xai.explainers.lime import LimeExplainer
from jev_xai.explainers.permutation import PermutationExplainer
from jev_xai.explainers.shap import ShapExplainer
from jev_xai.model.cassette import Cassette
from jev_xai.model.client import ModelClient
from jev_xai.model.fingerprint import model_fingerprint
from jev_xai.plugins import list_plugins
from jev_xai.replay.diff import cross_version_diff
from jev_xai.replay.recorder import DecisionRecorder
from jev_xai.replay.replay import ReplayEngine
from jev_xai.sources.jsonl import JsonlSource

app = typer.Typer(no_args_is_help=True, help="Accountability layer for JEV-like decision models.")
plugins_app = typer.Typer(help="Registered entry points.")
app.add_typer(plugins_app, name="plugins")


def main() -> None:
    app()


def _config(
    profile: str | None, config: Path | None, overrides: dict[str, Any] | None = None
) -> JevXaiConfig:
    return load_config(profile=profile, path=config, overrides=overrides)


def _model(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise typer.BadParameter("model must be module:callable")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory() if callable(factory) else factory


def _emit(payload: Any, fmt: str) -> None:
    typer.echo(render(payload, fmt))


@app.command()
def doctor(
    model: str | None = typer.Option(None, "--model", help="module:callable"),
    source: Path | None = typer.Option(None, "--source", help="JSONL decision log"),
    fmt: str = typer.Option("table", "--format"),
) -> None:
    """Report the capability tier and which explainers are available."""

    live = _model(model) if model else None
    records = JsonlSource(source).load() if source else []
    result = diagnose(
        live,
        input_reachable=bool(live) or any(record.input_reachable for record in records),
        has_corpus=len(records) > 0,
        source_only=live is None,
    )
    _emit(result, fmt)


@app.command(name="import")
def import_log(
    path: Path = typer.Argument(..., help="JSONL decision log"),
    out: Path = typer.Option(Path("records"), "--out"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    """Ingest a host-harness JSONL log into decision records."""

    source = JsonlSource(path)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for record in source.load():
        target = out / f"{record.decision_id}.json"
        target.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        written.append(str(target))
    diagnosis = diagnose(
        None, input_reachable=any(r.input_reachable for r in source.load()), source_only=True
    )
    _emit({"written": written, "tier": diagnosis.tier_name, "notes": diagnosis.notes}, fmt)


@app.command()
def record(
    model: str = typer.Option(..., "--model"),
    input: Path = typer.Option(..., "--input"),
    out: Path = typer.Option(Path("decision.json"), "--out"),
    profile: str | None = typer.Option(None, "--profile"),
    config: Path | None = typer.Option(None, "--config"),
    cassette_dir: Path | None = typer.Option(None, "--cassette"),
    store_dir: Path | None = typer.Option(None, "--store"),
) -> None:
    """Record one decision."""

    payload = json.loads(input.read_text(encoding="utf-8"))
    resolved = _config(profile, config)
    cassette = Cassette(cassette_dir) if cassette_dir else None
    store = EvidenceStore(store_dir) if store_dir else None
    recorder = DecisionRecorder(_model(model), resolved, cassette=cassette, store=store)
    result = run_sync(recorder.run, payload)
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(str(out))


@app.command()
def explain(
    model: str = typer.Option(..., "--model"),
    input: Path = typer.Option(..., "--input"),
    explainer: str = typer.Option("ablation", "--explainer"),
    context_path: Path | None = typer.Option(None, "--context", help="ExplainContext JSON"),
    profile: str | None = typer.Option(None, "--profile"),
    config: Path | None = typer.Option(None, "--config"),
    fmt: str = typer.Option("table", "--format"),
) -> None:
    """Run one explainer. Anchors, permutation, SHAP, and LIME need ``--context``."""

    payload = json.loads(input.read_text(encoding="utf-8"))
    resolved = _config(profile, config)
    live = _model(model)
    client = ModelClient(
        live, resolved.model, fingerprint=model_fingerprint(live), seed=resolved.seed
    )
    engines = {
        "ablation": AblationExplainer,
        "counterfactual": CounterfactualExplainer,
        "anchors": AnchorExplainer,
        "permutation": PermutationExplainer,
        "shap": ShapExplainer,
        "lime": LimeExplainer,
    }
    if explainer not in engines:
        raise typer.BadParameter(
            "explainer must be ablation, counterfactual, anchors, permutation, shap, or lime"
        )
    context = ExplainContext()
    if context_path is not None:
        context = ExplainContext.model_validate_json(context_path.read_text(encoding="utf-8"))
    result = run_sync(engines[explainer](resolved).explain, client, payload, context)
    _emit(result, fmt)


@app.command()
def replay(
    path: Path = typer.Argument(..., help="decision.json"),
    mode: str = typer.Option("exact", "--mode"),
    model: str | None = typer.Option(None, "--model"),
    profile: str | None = typer.Option(None, "--profile"),
    config: Path | None = typer.Option(None, "--config"),
    cassette_dir: Path | None = typer.Option(None, "--cassette"),
    store_dir: Path | None = typer.Option(None, "--store"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    """Replay a record. exact is evidence; current and cross re-invoke the model."""

    record = read_record(json.loads(path.read_text(encoding="utf-8")))
    resolved = _config(profile, config)
    cassette = Cassette(cassette_dir) if cassette_dir else None
    store = EvidenceStore(store_dir) if store_dir else None
    live = _model(model) if model else None
    result = run_sync(
        ReplayEngine(resolved, cassette=cassette, store=store).replay,
        record,
        model=live,
        mode=mode,
    )
    _emit(result, fmt)


@app.command()
def diff(
    records: Path = typer.Option(..., "--records"),
    model: str = typer.Option(..., "--model"),
    profile: str | None = typer.Option(None, "--profile"),
    config: Path | None = typer.Option(None, "--config"),
    store_dir: Path | None = typer.Option(None, "--store"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    """Cross-version decision-flip report."""

    paths = sorted(records.glob("*.json"))
    loaded = [read_record(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    resolved = _config(profile, config)
    store = EvidenceStore(store_dir) if store_dir else None
    report = run_sync(cross_version_diff, loaded, _model(model), resolved, store=store)
    _emit(report, fmt)


@app.command()
def audit(
    path: Path = typer.Argument(..., help="decision.json"),
    model: str = typer.Option(..., "--model"),
    out: Path = typer.Option(Path("audit"), "--out"),
    profile: str | None = typer.Option(None, "--profile"),
    config: Path | None = typer.Option(None, "--config"),
    store_dir: Path | None = typer.Option(None, "--store"),
) -> None:
    """Write an audit pack with a Merkle manifest, Markdown, and HTML."""

    record = read_record(json.loads(path.read_text(encoding="utf-8")))
    resolved = _config(profile, config)
    store = EvidenceStore(store_dir) if store_dir else None
    run_sync(build_audit_pack, record, _model(model), resolved, out, store=store)
    ok = verify_pack(out)
    typer.echo(f"{out} verified={ok} config_hash={config_hash(resolved)}")


@app.command()
def gate(
    records: Path = typer.Option(..., "--records"),
    min_stability: float = typer.Option(0.8, "--min-stability"),
    min_reproduction_rate: float = typer.Option(0.9, "--min-reproduction-rate"),
) -> None:
    """Fail when stability, reproduction, or replay evidence misses the threshold."""

    failures = gate_failures(
        records,
        min_stability=min_stability,
        min_reproduction_rate=min_reproduction_rate,
    )
    if failures:
        typer.echo("\n".join(failures))
        raise typer.Exit(code=1)
    typer.echo("gate passed")


@app.command()
def bench(
    out: Path = typer.Option(Path("benchmarks/results/latest.json"), "--out"),
    quick: bool = typer.Option(False, "--quick"),
) -> None:
    """Run the reference tasks and write measured metrics."""

    write_benchmarks(out, quick=quick)
    typer.echo(str(out))


@plugins_app.command("list")
def plugins_list(fmt: str = typer.Option("table", "--format")) -> None:
    """List registered explainers, adapters, and sources."""

    _emit(list_plugins(), fmt)
