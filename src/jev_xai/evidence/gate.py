"""CI gate over decision records and audit packs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def gate_failures(
    root: Path,
    *,
    min_stability: float,
    min_reproduction_rate: float,
) -> list[str]:
    """Return one message per failed check.

    Walks ``root`` recursively. Copies under ``store/`` or ``objects/`` are skipped
    so a flat audit directory is not scored twice. A directory with no stability,
    reproduction, replay, or counterfactual field fails closed. A flip that was
    not replay-confirmed fails the gate.
    """

    if not root.is_dir():
        return [f"{root}: not a directory"]
    failures: list[str] = []
    saw_evidence = False
    for path in _json_files(root):
        label = path.name if path.parent == root else str(path.relative_to(root))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append(f"{label}: invalid JSON")
            continue
        if not isinstance(payload, dict):
            continue
        if _check(payload, label, failures, min_stability, min_reproduction_rate):
            saw_evidence = True
    if not saw_evidence and not failures:
        failures.append(f"{root}: no stability, reproduction, replay, or counterfactual evidence")
    return failures


def _json_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if "store" in relative.parts or "objects" in relative.parts:
            continue
        found.append(path)
    return found


def _check(
    payload: dict[str, Any],
    label: str,
    failures: list[str],
    min_stability: float,
    min_reproduction_rate: float,
) -> bool:
    saw = False
    if "stability_score" in payload:
        saw = True
        score = _number(payload.get("stability_score"))
        if score is None:
            failures.append(f"{label}: stability_score is not a number")
        elif score < min_stability:
            failures.append(f"{label}: stability {payload['stability_score']}")
    rate = payload.get("reproduction_rate")
    probe = payload.get("reproducibility")
    if rate is None and isinstance(probe, dict):
        rate = probe.get("reproduction_rate")
    if rate is not None or (isinstance(probe, dict) and "reproduction_rate" in probe):
        saw = True
        value = _number(rate)
        if value is None:
            failures.append(f"{label}: reproduction_rate is not a number")
        elif value < min_reproduction_rate:
            failures.append(f"{label}: reproduction_rate {rate}")
    behavioral = payload.get("behavioral")
    if isinstance(behavioral, dict) and "matched" in behavioral:
        saw = True
        if behavioral.get("matched") is False:
            reason = behavioral.get("mismatch_reason") or "unmatched"
            failures.append(f"{label}: behavioral replay did not match ({reason})")
    if "matched" in payload:
        saw = True
        if payload.get("matched") is False:
            reason = payload.get("mismatch_reason") or "unmatched"
            failures.append(f"{label}: replay did not match ({reason})")
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for index, item in enumerate(candidates):
            if not isinstance(item, dict) or "replay_confirmed" not in item:
                continue
            saw = True
            if item.get("flipped") is True and item.get("replay_confirmed") is not True:
                changes = item.get("changes") or []
                names = [
                    str(change.get("feature"))
                    for change in changes
                    if isinstance(change, dict) and change.get("feature") is not None
                ]
                feature = ", ".join(names) if names else str(index)
                failures.append(
                    f"{label}: counterfactual [{feature}] flipped but was not replay_confirmed"
                )
    return saw


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
