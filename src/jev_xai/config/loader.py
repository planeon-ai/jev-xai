"""Load configuration with a single documented precedence chain.

Later sources win: defaults, then named profile, then project file,
then ``JEV_XAI_*`` environment variables, then CLI flags, then explicit
Python overrides.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from jev_xai.canonical import content_hash
from jev_xai.config.schema import JevXaiConfig

_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


def _read_toml(path: Path) -> dict[str, Any]:
    import tomli

    loaded: dict[str, Any] = tomli.loads(path.read_text(encoding="utf-8"))
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _profile_path(name: str) -> Path:
    filename = name.replace("-", "_") + ".toml"
    path = _PROFILE_DIR / filename
    if not path.is_file():
        known = sorted(p.stem for p in _PROFILE_DIR.glob("*.toml"))
        raise FileNotFoundError(f"unknown profile '{name}'. known profiles: {known}")
    return path


def _env_overlay(environ: dict[str, str]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    prefix = "JEV_XAI_"
    for key, raw in environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        cursor = root
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def config_hash(config: JevXaiConfig) -> str:
    """blake2b of the canonical JSON of the resolved config."""

    return content_hash(config.model_dump(mode="json"))


def load_config(
    *,
    profile: str | None = None,
    path: Path | str | None = None,
    environ: dict[str, str] | None = None,
    cli: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> JevXaiConfig:
    """Resolve a config. See the module docstring for precedence."""

    data: dict[str, Any] = {}
    if profile:
        data = _deep_merge(data, _read_toml(_profile_path(profile)))
    if path is not None:
        data = _deep_merge(data, _read_toml(Path(path)))
    env = os.environ if environ is None else environ
    data = _deep_merge(data, _env_overlay(dict(env)))
    if cli:
        data = _deep_merge(data, cli)
    if overrides:
        data = _deep_merge(data, overrides)
    return JevXaiConfig.model_validate(data)
