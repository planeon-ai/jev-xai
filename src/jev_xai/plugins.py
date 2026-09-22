"""Entry-point discovery for explainers, adapters, and evidence sources."""

from __future__ import annotations

from importlib.metadata import entry_points


def iter_plugins(group: str) -> list[tuple[str, str]]:
    selected = entry_points().select(group=group)
    return [(item.name, item.value) for item in selected]


def list_plugins() -> dict[str, list[str]]:
    groups = ("jev_xai.explainers", "jev_xai.adapters", "jev_xai.sources")
    return {group: [f"{name} = {value}" for name, value in iter_plugins(group)] for group in groups}
