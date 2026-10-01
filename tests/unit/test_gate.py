"""The CI gate walks packs and fails closed."""

from __future__ import annotations

import json
from pathlib import Path

from jev_xai.evidence.gate import gate_failures
from jev_xai.evidence.html import render_html
from jev_xai.evidence.report import render_markdown


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_nested_low_stability_fails(tmp_path: Path) -> None:
    _write(tmp_path / "pack-a" / "stability.json", {"stability_score": 0.99})
    _write(tmp_path / "pack-b" / "stability.json", {"stability_score": 0.2})
    failures = gate_failures(tmp_path, min_stability=0.8, min_reproduction_rate=0.9)
    assert failures == ["pack-b/stability.json: stability 0.2"]


def test_unmatched_replay_fails_and_store_copy_is_ignored(tmp_path: Path) -> None:
    _write(
        tmp_path / "replay.json",
        {
            "matched": True,
            "behavioral": {"matched": False, "mismatch_reason": "model_nondeterministic"},
        },
    )
    _write(tmp_path / "store" / "packs" / "abc" / "replay.json", {"matched": False})
    failures = gate_failures(tmp_path, min_stability=0.8, min_reproduction_rate=0.9)
    assert failures == ["replay.json: behavioral replay did not match (model_nondeterministic)"]


def test_decision_reproduction_rate_is_read(tmp_path: Path) -> None:
    _write(
        tmp_path / "decision.json",
        {"reproducibility": {"reproduction_rate": 0.4}},
    )
    failures = gate_failures(tmp_path, min_stability=0.8, min_reproduction_rate=0.9)
    assert failures == ["decision.json: reproduction_rate 0.4"]


def test_directory_without_evidence_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path / "explanation.json", {"ablation_top": []})
    failures = gate_failures(tmp_path, min_stability=0.8, min_reproduction_rate=0.9)
    assert any(
        "no stability, reproduction, replay, or counterfactual evidence" in item
        for item in failures
    )


def test_missing_directory_fails(tmp_path: Path) -> None:
    failures = gate_failures(tmp_path / "missing", min_stability=0.8, min_reproduction_rate=0.9)
    assert failures
    assert "not a directory" in failures[0]


def test_bad_evidence_is_named(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    _write(tmp_path / "stability.json", {"stability_score": "high"})
    _write(tmp_path / "replay.json", {"matched": False, "mismatch_reason": "model_version_changed"})
    failures = gate_failures(tmp_path, min_stability=0.8, min_reproduction_rate=0.9)
    assert "broken.json: invalid JSON" in failures
    assert "stability.json: stability_score is not a number" in failures
    assert "replay.json: replay did not match (model_version_changed)" in failures


def test_noop_ablation_is_labeled_in_the_report() -> None:
    pack = {
        "decision": {"output": {}},
        "ablation": {"rows": [{"feature": "verified_account", "delta_p": None, "noop": True}]},
        "counterfactuals": {"candidates": []},
        "stability": {},
    }
    assert "| verified_account | noop |" in render_markdown(pack)
    assert ">noop<" in render_html(pack)


def test_unconfirmed_counterfactual_fails(tmp_path: Path) -> None:
    _write(
        tmp_path / "counterfactuals.json",
        {
            "candidates": [
                {
                    "flipped": True,
                    "replay_confirmed": False,
                    "changes": [{"feature": "verified_user"}],
                }
            ]
        },
    )
    failures = gate_failures(tmp_path, min_stability=0.8, min_reproduction_rate=0.9)
    assert failures == [
        "counterfactuals.json: counterfactual [verified_user] flipped but was not replay_confirmed"
    ]


def test_confirmed_counterfactual_passes(tmp_path: Path) -> None:
    _write(
        tmp_path / "counterfactuals.json",
        {"candidates": [{"flipped": True, "replay_confirmed": True, "changes": []}]},
    )
    assert gate_failures(tmp_path, min_stability=0.8, min_reproduction_rate=0.9) == []


def test_numeric_anchor_is_rendered() -> None:
    pack = {
        "decision": {"output": {}},
        "ablation": {"rows": []},
        "counterfactuals": {"candidates": []},
        "anchors": {
            "predicates": [{"feature": "amount", "op": "within", "low": 1.0, "high": 3.0}],
            "precision": 0.95,
            "coverage": 0.2,
            "sufficient": True,
        },
        "permutation": {"rows": []},
        "stability": {},
    }
    text = render_markdown(pack)
    page = render_html(pack)
    assert "amount within [1.0, 3.0]" in text
    assert "amount within [1.0, 3.0]" in page
    assert "| — | — | — |" in text
