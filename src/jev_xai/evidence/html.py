"""Single-file HTML report. No JavaScript, no external assets."""

from __future__ import annotations

import html
from typing import Any

from jev_xai.limitations import LIMITATIONS


def render_html(pack: dict[str, Any]) -> str:
    decision = pack.get("decision", {})
    output = decision.get("output", {})
    probe = decision.get("reproducibility") or {}
    rows = pack.get("ablation", {}).get("rows", [])
    magnitudes = [abs(row.get("delta_p") or 0.0) for row in rows]
    peak = max(magnitudes) if magnitudes else 1.0
    bars: list[str] = []
    for row in rows:
        delta = row.get("delta_p")
        width = 0 if delta is None or peak == 0 else abs(delta) / peak * 100
        muted = " muted" if row.get("below_noise_floor") or row.get("noop") else ""
        label = html.escape(str(row.get("feature")))
        if row.get("noop"):
            shown = "noop"
        else:
            shown = "n/a" if delta is None else f"{delta:.4f}"
        bars.append(
            f'<div class="row{muted}"><span class="name">{label}</span>'
            f'<span class="track"><span class="bar" style="width:{width:.1f}%"></span></span>'
            f'<span class="value">{shown}</span></div>'
        )
    changes: list[str] = []
    for candidate in pack.get("counterfactuals", {}).get("candidates", []):
        parts = [
            f"{html.escape(str(change.get('feature')))}: "
            f"{html.escape(str(change.get('before')))} → {html.escape(str(change.get('after')))}"
            for change in candidate.get("changes", [])
        ]
        changes.append(
            "<li>"
            + (", ".join(parts) or "no change")
            + f" → <strong>{html.escape(str(candidate.get('label')))}</strong>"
            + f" · confirmed {html.escape(str(candidate.get('confirmed_label')))}"
            + f" · flipped {html.escape(str(candidate.get('flipped')))}"
            + f" · replay_confirmed {html.escape(str(candidate.get('replay_confirmed')))}</li>"
        )
    anchors = _anchor_html(pack.get("anchors") or {})
    permutation = _permutation_html(pack.get("permutation") or {})
    shap = _attribution_html(pack.get("shap") or {}, "value")
    lime = _attribution_html(pack.get("lime") or {}, "weight")
    stability = pack.get("stability", {})
    replay = pack.get("replay", {})
    banner = ""
    if probe:
        banner = (
            f"<p class='banner'>Reproduction rate {probe.get('reproduction_rate')} · "
            f"noise floor {probe.get('noise_floor')}</p>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>jev-xai evidence {html.escape(str(decision.get("decision_id", "")))}</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 46rem; color: #1c1917; background: #faf7f2; }}
h1 {{ font-weight: 500; letter-spacing: -0.02em; }}
.banner {{ background: #f3e6c8; padding: 0.6rem 0.8rem; }}
.row {{ display: grid; grid-template-columns: 12rem 1fr 5rem; gap: 0.6rem; align-items: center; margin: 0.25rem 0; }}
.row.muted {{ color: #a8a29e; }}
.track {{ background: #e7e5e4; height: 0.7rem; }}
.bar {{ display: block; height: 0.7rem; background: #9a3412; }}
.muted .bar {{ background: #d6d3d1; }}
.panel {{ border-top: 1px solid #d6d3d1; margin-top: 1.2rem; padding-top: 0.6rem; }}
small {{ color: #57534e; }}
</style>
</head>
<body>
<h1>Decision {html.escape(str(output.get("label", "")))}</h1>
{banner}
<p>{html.escape(str(decision.get("model", {}).get("model_name", "")))}
· probability {html.escape(str(output.get("probability")))}</p>
<div class="panel"><h2>Ablation</h2>{"".join(bars) or "<p>No ablation rows.</p>"}</div>
<div class="panel"><h2>Counterfactuals</h2><ul>{"".join(changes) or "<li>None</li>"}</ul></div>
<div class="panel"><h2>Anchors</h2>{anchors}</div>
<div class="panel"><h2>Permutation</h2>{permutation}</div>
<div class="panel"><h2>SHAP</h2>{shap}</div>
<div class="panel"><h2>LIME</h2>{lime}</div>
<div class="panel"><h2>Stability</h2>
<p>stability_score_v1 {html.escape(str(stability.get("stability_score")))}
on {html.escape(str(stability.get("measured_explainer") or "unspecified explainer"))}</p></div>
<div class="panel"><h2>Replay</h2>
<p>{html.escape(str(replay.get("claim")))} · matched {html.escape(str(replay.get("matched")))}</p>
<small>{html.escape(str(replay.get("note", "")))}</small></div>
<div class="panel"><h2>Limitations</h2><small>{html.escape(LIMITATIONS)}</small></div>
</body>
</html>
"""


def _anchor_html(payload: dict[str, Any]) -> str:
    if payload.get("skipped"):
        reason = html.escape(str(payload.get("prerequisite")))
        fix = html.escape(str(payload.get("fix")))
        return f"<p>Skipped ({reason}): {fix}</p>"
    predicates = payload.get("predicates") or []
    parts: list[str] = []
    for predicate in predicates:
        feature = html.escape(str(predicate.get("feature")))
        if predicate.get("op") == "within":
            parts.append(f"{feature} within [{predicate.get('low')}, {predicate.get('high')}]")
        else:
            parts.append(f"{feature} = {html.escape(str(predicate.get('value')))}")
    rule = " AND ".join(parts) if parts else "no predicate"
    return (
        f"<p>{html.escape(rule)}</p>"
        f"<p>Precision {html.escape(str(payload.get('precision')))} · "
        f"coverage {html.escape(str(payload.get('coverage')))} · "
        f"sufficient {html.escape(str(payload.get('sufficient')))}</p>"
    )


def _permutation_html(payload: dict[str, Any]) -> str:
    if payload.get("skipped"):
        reason = html.escape(str(payload.get("prerequisite")))
        fix = html.escape(str(payload.get("fix")))
        return f"<p>Skipped ({reason}): {fix}</p>"
    items: list[str] = []
    for row in payload.get("rows") or []:
        feature = html.escape(str(row.get("feature")))
        raw = "unmeasured" if row.get("unmeasured") else row.get("importance")
        importance = html.escape(str(raw))
        rate = html.escape(str(row.get("label_flip_rate")))
        items.append(f"<li>{feature}: {importance} · flip rate {rate}</li>")
    return f"<ul>{''.join(items) or '<li>None</li>'}</ul>"


def _attribution_html(payload: dict[str, Any], value_key: str) -> str:
    if payload.get("skipped") is True:
        reason = html.escape(str(payload.get("prerequisite")))
        fix = html.escape(str(payload.get("fix")))
        return f"<p>Skipped ({reason}): {fix}</p>"
    items: list[str] = []
    for row in payload.get("rows") or []:
        feature = html.escape(str(row.get("feature")))
        value = html.escape(str(row.get(value_key)))
        items.append(f"<li>{feature}: {value}</li>")
    omitted = payload.get("skipped") or []
    extra = ""
    if isinstance(omitted, list) and omitted:
        names = ", ".join(html.escape(str(name)) for name in omitted)
        extra += f"<p>Columns left out: {names}</p>"
    note = payload.get("note")
    if note:
        extra += f"<p>{html.escape(str(note))}</p>"
    return f"<ul>{''.join(items) or '<li>None</li>'}</ul>{extra}"
