"""The API page stays aligned with the public export list."""

from __future__ import annotations

from pathlib import Path

import jev_xai

ROOT = Path(__file__).resolve().parents[2]


def test_api_doc_names_every_public_export() -> None:
    text = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    missing = [name for name in jev_xai.__all__ if name not in text]
    assert missing == []
