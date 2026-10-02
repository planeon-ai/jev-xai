"""Markdown evidence report."""

from __future__ import annotations

from typing import Any

from jev_xai.limitations import LIMITATIONS


def render_markdown(pack: dict[str, Any]) -> str:
    decision = pack.get("decision", {})
    output = decision.get("output", {})
    probe = decision.get("reproducibility") or {}
    lines = [
        "# Decision evidence",
        "",
        f"- Decision: `{decision.get('decision_id', '')}`",
        f"- Label: **{output.get('label', '')}**",
        f"- Probability: {output.get('probability')}",
        f"- Model: {decision.get('model', {}).get('model_name')} "
        f"({decision.get('model', {}).get('fingerprint')})",
        f"- Config hash: `{decision.get('explainer_config', {}).get('config_hash')}`",
        "",
        "## Reproducibility",
        "",
    ]
    if probe:
        lines.append(
            f"Reproduction rate {probe.get('reproduction_rate')} "
            f"(sigma {probe.get('probability_sigma')}, noise floor {probe.get('noise_floor')})."
        )
    else:
        lines.append("No repeat probe was recorded for this decision.")
    lines.extend(
        [
            "",
            "## Ablation",
            "",
            "| Feature | Delta P | Flipped | Below noise floor |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in pack.get("ablation", {}).get("rows", []):
        delta = "noop" if row.get("noop") else row.get("delta_p")
        lines.append(
            f"| {row.get('feature')} | {delta} | {row.get('label_flipped')} | {row.get('below_noise_floor')} |"
        )
    lines.extend(["", "## Counterfactuals", ""])
    for candidate in pack.get("counterfactuals", {}).get("candidates", []):
        changes = ", ".join(
            f"{change.get('feature')}: {change.get('before')} -> {change.get('after')}"
            for change in candidate.get("changes", [])
        )
        lines.append(
            f"- {changes or 'no change'} => {candidate.get('label')} "
            f"(confirmed {candidate.get('confirmed_label')}, "
            f"distance {candidate.get('distance')}, sparsity {candidate.get('sparsity')}, "
            f"flipped {candidate.get('flipped')}, "
            f"replay_confirmed {candidate.get('replay_confirmed')})"
        )
    lines.extend(["", "## Anchors", ""])
    lines.extend(_anchor_lines(pack.get("anchors") or {}))
    lines.extend(["", "## Permutation", ""])
    lines.extend(_permutation_lines(pack.get("permutation") or {}))
    lines.extend(["", "## SHAP", ""])
    lines.extend(_attribution_lines(pack.get("shap") or {}, "value", "Value"))
    lines.extend(["", "## LIME", ""])
    lines.extend(_attribution_lines(pack.get("lime") or {}, "weight", "Weight"))
    stability = pack.get("stability", {})
    lines.extend(
        [
            "",
            "## Stability",
            "",
            f"stability_score_v1 = {stability.get('stability_score')} "
            f"on {stability.get('measured_explainer') or 'unspecified explainer'}",
            f"rank correlation {stability.get('rank_correlation')}, "
            f"overlap {stability.get('feature_overlap')}, "
            f"failed runs {stability.get('n_failed_runs')}",
            "",
            "## Replay",
            "",
            f"Claim: {pack.get('replay', {}).get('claim')}",
            f"Matched: {pack.get('replay', {}).get('matched')}",
            f"Reason: {pack.get('replay', {}).get('mismatch_reason')}",
            "",
            "## Limitations",
            "",
            LIMITATIONS,
            "",
        ]
    )
    return "\n".join(lines)


def _anchor_lines(payload: dict[str, Any]) -> list[str]:
    if payload.get("skipped"):
        return [f"Skipped ({payload.get('prerequisite')}): {payload.get('fix')}"]
    predicates = payload.get("predicates") or []
    parts: list[str] = []
    for predicate in predicates:
        if predicate.get("op") == "within":
            parts.append(
                f"{predicate.get('feature')} within [{predicate.get('low')}, {predicate.get('high')}]"
            )
        else:
            parts.append(f"{predicate.get('feature')} = {predicate.get('value')}")
    rule = " AND ".join(parts) if parts else "no predicate"
    return [
        f"Rule: {rule}",
        f"Precision {payload.get('precision')}, coverage {payload.get('coverage')}, "
        f"sufficient {payload.get('sufficient')}.",
    ]


def _permutation_lines(payload: dict[str, Any]) -> list[str]:
    if payload.get("skipped"):
        return [f"Skipped ({payload.get('prerequisite')}): {payload.get('fix')}"]
    lines = [
        "| Feature | Importance | Label flip rate |",
        "| --- | ---: | ---: |",
    ]
    for row in payload.get("rows") or []:
        importance = "unmeasured" if row.get("unmeasured") else row.get("importance")
        lines.append(f"| {row.get('feature')} | {importance} | {row.get('label_flip_rate')} |")
    if len(lines) == 2:
        lines.append("| — | — | — |")
    return lines


def _attribution_lines(payload: dict[str, Any], value_key: str, heading: str) -> list[str]:
    if payload.get("skipped") is True:
        return [f"Skipped ({payload.get('prerequisite')}): {payload.get('fix')}"]
    lines = [
        f"| Feature | {heading} |",
        "| --- | ---: |",
    ]
    for row in payload.get("rows") or []:
        lines.append(f"| {row.get('feature')} | {row.get(value_key)} |")
    if len(lines) == 2:
        lines.append("| — | — |")
    omitted = payload.get("skipped") or []
    if isinstance(omitted, list) and omitted:
        lines.extend(["", "Columns left out: " + ", ".join(str(name) for name in omitted)])
    note = payload.get("note")
    if note:
        lines.extend(["", str(note)])
    base = payload.get("base_value")
    if base is not None:
        lines.append(f"Base value {base}.")
    return lines
