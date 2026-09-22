"""Content-addressed cassette of model inputs and outputs."""

from __future__ import annotations

from pathlib import Path

from jev_xai.canonical import canonical_json
from jev_xai.types import Prediction


class Cassette:
    """On-disk record of model I/O, keyed by the call cache key.

    Replaying a cassette entry makes zero model calls. It proves what the
    evidence was. It does not prove the model would decide the same way now.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, key: str) -> Path:
        return self.directory / key[:2] / key

    def put(self, key: str, prediction: Prediction) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(prediction.model_dump(mode="json"))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    def get(self, key: str) -> Prediction | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        return Prediction.model_validate_json(path.read_text(encoding="utf-8"))
