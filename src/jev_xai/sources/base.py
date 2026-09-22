"""EvidenceSource: post-hoc ingestion from a host harness."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from jev_xai.evidence.schema import DecisionRecord


@runtime_checkable
class EvidenceSource(Protocol):
    """Read decisions that were already logged somewhere else."""

    def load(self) -> list[DecisionRecord]:
        """Return records. Re-invocation is not implied."""


class RecordSource:
    """In-memory source used by tests and by the JSONL importer."""

    def __init__(self, records: list[DecisionRecord]) -> None:
        self._records = records

    def load(self) -> list[DecisionRecord]:
        return list(self._records)

    @property
    def path(self) -> Path | None:
        return None
