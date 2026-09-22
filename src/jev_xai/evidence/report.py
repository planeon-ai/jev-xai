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
        lines.append(
            f"| {row.get('feature')} | {row.get('delta_p')} | {row.get('label_flipped')} | {row.get('below_noise_floor')} |"
        )
    lines.extend(["", "## Counterfactuals", ""])
    for candidate in pack.get("counterfactuals", {}).get("candidates", []):
        changes = ", ".join(
            f"{change.get('feature')}: {change.get('before')} -> {change.get('after')}"
            for change in candidate.get("changes", [])
        )
        lines.append(
            f"- {changes or 'no change'} => {candidate.get('label')} "
            f"(distance {candidate.get('distance')}, sparsity {candidate.get('sparsity')}, "
            f"flipped {candidate.get('flipped')})"
        )
    stability = pack.get("stability", {})
    lines.extend(
        [
            "",
            "## Stability",
            "",
            f"stability_score_v1 = {stability.get('stability_score')}",
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
