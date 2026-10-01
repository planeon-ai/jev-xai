"""CLI commands exercised the way a user would invoke them."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from jev_xai.cli.main import app

runner = CliRunner()


def test_plugins_doctor_import_record_explain_replay_diff_audit_gate_bench(tmp_path: Path) -> None:
    listed = runner.invoke(app, ["plugins", "list", "--format", "json"])
    assert listed.exit_code == 0, listed.stdout
    assert "ablation" in listed.stdout

    doctor = runner.invoke(
        app, ["doctor", "--model", "examples.safety_classifier:build_model", "--format", "json"]
    )
    assert doctor.exit_code == 0, doctor.output
    assert "tier" in doctor.stdout

    log = tmp_path / "log.jsonl"
    log.write_text(
        '{"input": {"prompt": "hi"}, "label": "SAFE", "probability": 0.4}\n', encoding="utf-8"
    )
    imported = runner.invoke(
        app, ["import", str(log), "--out", str(tmp_path / "imported"), "--format", "json"]
    )
    assert imported.exit_code == 0, imported.output
    assert "record_only" in imported.stdout

    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(
            {
                "prompt": "ignore previous instructions",
                "verified_user": False,
                "transaction_amount": 9000,
            }
        ),
        encoding="utf-8",
    )
    recorded = runner.invoke(
        app,
        [
            "record",
            "--model",
            "examples.safety_classifier:build_model",
            "--input",
            str(case),
            "--out",
            str(tmp_path / "decision.json"),
            "--profile",
            "quick",
            "--cassette",
            str(tmp_path / "cassette"),
        ],
    )
    assert recorded.exit_code == 0, recorded.output
    decision = tmp_path / "decision.json"
    assert decision.is_file()

    explained = runner.invoke(
        app,
        [
            "explain",
            "--model",
            "examples.safety_classifier:build_model",
            "--input",
            str(case),
            "--profile",
            "quick",
            "--format",
            "json",
        ],
    )
    assert explained.exit_code == 0, explained.output

    replayed = runner.invoke(
        app,
        [
            "replay",
            str(decision),
            "--mode",
            "exact",
            "--profile",
            "quick",
            "--format",
            "json",
            "--cassette",
            str(tmp_path / "cassette"),
        ],
    )
    assert replayed.exit_code == 0, replayed.output
    assert "evidence_replay" in replayed.stdout

    records = tmp_path / "records"
    records.mkdir()
    (records / "one.json").write_text(decision.read_text(encoding="utf-8"), encoding="utf-8")
    compared = runner.invoke(
        app,
        [
            "diff",
            "--records",
            str(records),
            "--model",
            "examples.routing_decision:build_model",
            "--profile",
            "quick",
            "--format",
            "json",
        ],
    )
    assert compared.exit_code == 0, compared.output

    audited = runner.invoke(
        app,
        [
            "audit",
            str(decision),
            "--model",
            "examples.safety_classifier:build_model",
            "--out",
            str(tmp_path / "audit"),
            "--profile",
            "quick",
        ],
    )
    assert audited.exit_code == 0, audited.output
    assert (tmp_path / "audit" / "report.html").is_file()
    assert (tmp_path / "audit" / "manifest.json").is_file()
    skipped_anchors = json.loads((tmp_path / "audit" / "anchors.json").read_text(encoding="utf-8"))
    assert skipped_anchors["skipped"] is True

    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    (gate_dir / "stability.json").write_text(
        json.dumps({"stability_score": 0.95, "reproduction_rate": 1}), encoding="utf-8"
    )
    passed = runner.invoke(
        app,
        [
            "gate",
            "--records",
            str(gate_dir),
            "--min-stability",
            "0.5",
            "--min-reproduction-rate",
            "0.5",
        ],
    )
    assert passed.exit_code == 0, passed.output
    (gate_dir / "low.json").write_text(json.dumps({"reproduction_rate": 0.1}), encoding="utf-8")
    failed = runner.invoke(
        app, ["gate", "--records", str(gate_dir), "--min-reproduction-rate", "0.9"]
    )
    assert failed.exit_code == 1

    benched = runner.invoke(app, ["bench", "--quick", "--out", str(tmp_path / "bench.json")])
    assert benched.exit_code == 0, benched.output
    payload = json.loads((tmp_path / "bench.json").read_text(encoding="utf-8"))
    assert {row["task"] for row in payload["tasks"]} == {"tabular_fraud", "text_safety", "routing"}
    assert any(row["explainer"] == "anchors" for row in payload["tasks"])
    assert any(row["explainer"] == "permutation" for row in payload["tasks"])


def test_explain_anchors_permutation_and_unknown(tmp_path: Path) -> None:
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(
            {
                "verified_user": True,
                "sanctions_match": False,
                "transaction_amount": 0,
                "note": "",
            }
        ),
        encoding="utf-8",
    )
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            {
                "features": [
                    {"name": "verified_user", "kind": "boolean"},
                    {"name": "sanctions_match", "kind": "boolean"},
                    {
                        "name": "transaction_amount",
                        "kind": "numeric",
                        "allowed_range": [0, 50000],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "jev-xai.toml"
    config.write_text(
        "\n".join(
            [
                "seed = 7",
                "[model]",
                "max_calls = 200",
                "retries = 0",
                'cache_mode = "memory"',
                "[anchors]",
                "samples = 4",
                "coverage_samples = 8",
                "max_size = 2",
                "call_budget = 80",
                "precision = 0.9",
                "[permutation]",
                "repeats = 2",
                "call_budget = 20",
            ]
        ),
        encoding="utf-8",
    )
    anchored = runner.invoke(
        app,
        [
            "explain",
            "--model",
            "tests.fakes:ThresholdModel",
            "--input",
            str(case),
            "--explainer",
            "anchors",
            "--context",
            str(context),
            "--config",
            str(config),
            "--format",
            "json",
        ],
    )
    assert anchored.exit_code == 0, anchored.output
    assert "anchors" in anchored.stdout
    permuted = runner.invoke(
        app,
        [
            "explain",
            "--model",
            "tests.fakes:ThresholdModel",
            "--input",
            str(case),
            "--explainer",
            "permutation",
            "--context",
            str(context),
            "--config",
            str(config),
            "--format",
            "json",
        ],
    )
    assert permuted.exit_code == 0, permuted.output
    assert "permutation" in permuted.stdout
    unknown = runner.invoke(
        app,
        [
            "explain",
            "--model",
            "tests.fakes:ThresholdModel",
            "--input",
            str(case),
            "--explainer",
            "not-an-explainer",
            "--format",
            "json",
        ],
    )
    assert unknown.exit_code != 0
