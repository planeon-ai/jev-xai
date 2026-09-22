"""Plain and rich renderers for CLI output."""

from __future__ import annotations

import json
from typing import Any


def render(payload: Any, fmt: str) -> str:
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    if fmt == "json":
        return json.dumps(data, indent=2, sort_keys=True)
    if fmt == "md":
        return _markdown(data)
    if fmt == "html":
        from jev_xai.evidence.html import render_html

        pack = data if isinstance(data, dict) and "decision" in data else {"decision": data}
        return render_html(pack)
    return _table(data)


def _markdown(data: Any) -> str:
    if isinstance(data, dict):
        lines = ["| key | value |", "| --- | --- |"]
        for key, value in data.items():
            lines.append(f"| {key} | {value} |")
        return "\n".join(lines)
    return str(data)


def _table(data: Any) -> str:
    try:
        from rich.console import Console
        from rich.pretty import Pretty
    except ImportError:  # pragma: no cover - cli extra provides rich
        return _markdown(data)
    console = Console(record=True, width=100)
    console.print(Pretty(data))
    return console.export_text()
